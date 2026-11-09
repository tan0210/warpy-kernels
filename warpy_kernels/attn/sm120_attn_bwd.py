import math

import cutlass
import torch
from cutlass import Float32, Int32, Int64, const_expr, cute
from cutlass.cute.runtime import from_dlpack

from ..sm120.ptx import (
    SMEM_LIMIT,
    bulk_reduce_add_f32,
    cp_async16,
    div_rn,
    ds_scale,
    e4m3x4_to_f32,
    ld_global_bf16,
    ld_shared_v2_f32,
    ldsm_x2_trans_b8,
    ldsm_x4,
    mma_fp8,
    pack_e4m3x4,
    pack_half2,
    red_add_v2_f32,
    red_max_pos_f32,
    st_global_b16,
    st_global_b32,
    st_global_zero,
    st_shared_e4m3_pair,
    st_shared_v2_f32,
    swizzle_offset,
)
from ..utils import (
    autotune,
    compile_cached,
    flat,
    pad_rows,
    round_up,
    tune_key,
    tune_lookup,
)
from .constants import LOG2E, P_OFFSET


class BwdPrep:
    ROWS_PER_CTA = 8

    def __init__(self, head_dim: int, zero_dq: bool = True, block_m: int = 32):
        self.d = head_dim
        self.per_lane = head_dim // 32
        self.zero_dq = zero_dq
        self.block_m = block_m

    def _row_coords(self, row, nq, h_q, h_kv):
        b = row // (h_q * nq)
        rem = row % (h_q * nq)
        hq = rem // nq
        return b, hq, rem % nq, hq // (h_q // h_kv)

    @cute.jit
    def __call__(self, mdO: cute.Tensor, mO: cute.Tensor, mPmax: cute.Tensor, mdO8: cute.Tensor,
                 mD: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDqAcc: cute.Tensor, rows: Int32,
                 nq: Int32, nq_pad: Int32, h_q: Int32, h_kv: Int32):
        nblk = (rows + (self.ROWS_PER_CTA - 1)) // self.ROWS_PER_CTA
        rows_pad = (rows // nq) * nq_pad
        nblk_pad = (rows_pad + (self.ROWS_PER_CTA - 1)) // self.ROWS_PER_CTA
        self.k_amax(mdO, mStats, mDqAcc, rows_pad, nq, nq_pad, h_q, h_kv).launch(
            grid=[nblk_pad, 1, 1], block=[256, 1, 1])
        self.k_quant(mdO, mO, mPmax, mdO8, mD, mTB, mStats, mDqAcc, rows, nq, nq_pad, h_q, h_kv).launch(
            grid=[nblk, 1, 1], block=[256, 1, 1])

    @cute.kernel
    def k_amax(self, mdO: cute.Tensor, mStats: cute.Tensor, mDqAcc: cute.Tensor, rows_pad: Int32, nq: Int32,
               nq_pad: Int32, h_q: Int32, h_kv: Int32):
        # real rows feed the `dO` amax, padded rows zero their `dQaccum` row
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        prow = bidx * self.ROWS_PER_CTA + tidx // 32
        b, hq, i, hkv = self._row_coords(prow, nq_pad, h_q, h_kv)
        row = (b * h_q + hq) * nq + i
        if const_expr(self.zero_dq):
            if prow < rows_pad:
                if i >= nq:
                    gz = Int64(mDqAcc.iterator.toint()) + (Int64(prow) * self.d + Int64(lane * self.per_lane)) * 4
                    st_global_zero(gz, self.per_lane)
        if prow < rows_pad and i < nq:
            m = Float32(0.0)
            xin = ld_global_bf16(Int64(mdO.iterator.toint()) + (Int64(row) * self.d + Int64(lane * self.per_lane)) * 2,
                                 self.per_lane)
            for e in cutlass.range_constexpr(self.per_lane):
                m = cute.arch.fmax(m, xin[e] if xin[e] > Float32(0.0) else -xin[e])
            m = cute.arch.warp_reduction_max(m, threads_in_group=32)
            if lane == 0:
                red_max_pos_f32(Int64(mStats.iterator.toint()) + Int64((b * h_kv + hkv) * 4) * 4, m)

    @cute.kernel
    def k_quant(self, mdO: cute.Tensor, mO: cute.Tensor, mPmax: cute.Tensor, mdO8: cute.Tensor,
                mD: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDqAcc: cute.Tensor, rows: Int32, nq: Int32,
                nq_pad: Int32, h_q: Int32, h_kv: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        row = bidx * self.ROWS_PER_CTA + tidx // 32
        if row < rows:
            b, hq, i, hkv = self._row_coords(row, nq, h_q, h_kv)
            amax = mStats[b, hkv, 0]
            s_do = amax * Float32(1.0 / 256.0) if amax > Float32(0.0) else Float32(1.0)
            inv = Float32(1.0) / s_do
            col0 = lane * self.per_lane
            xin = ld_global_bf16(Int64(mdO.iterator.toint()) + (Int64(row) * self.d + Int64(col0)) * 2, self.per_lane)
            oin = ld_global_bf16(Int64(mO.iterator.toint()) + (Int64(row) * self.d + Int64(col0)) * 2, self.per_lane)
            x = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
            for e in cutlass.range_constexpr(self.per_lane):
                x[e] = xin[e] * inv
            bits = pack_e4m3x4(x[0], x[1], x[2], x[3])
            prow = (b * h_q + hq) * nq_pad + i
            gdst = Int64(mdO8.iterator.toint()) + Int64(prow) * self.d + Int64(col0)
            if const_expr(self.per_lane == 4):
                st_global_b32(gdst, bits)
            else:
                st_global_b16(gdst, bits)
            if const_expr(self.zero_dq):
                gz = Int64(mDqAcc.iterator.toint()) + (Int64(prow) * self.d + Int64(col0)) * 4
                st_global_zero(gz, self.per_lane)
            deq = e4m3x4_to_f32(bits)
            dsum = Float32(0.0)
            nsum = Float32(0.0)
            for e in cutlass.range_constexpr(self.per_lane):
                v = deq[e] * s_do
                dsum = dsum + v * oin[e]
                nsum = nsum + v * v
            dsum = cute.arch.warp_reduction_sum(dsum, threads_in_group=32)
            nsum = cute.arch.warp_reduction_sum(nsum, threads_in_group=32)
            if lane == 0:
                mD[prow] = dsum
                # bound `|dS| <= 2 pmax |dO| max|V|`, max-reduced per query tile
                rb = Float32(2.0) * mPmax[row] * cute.math.sqrt(nsum, fastmath=True) * mStats[b, hkv, 2]
                tile = (b * h_q + hq) * (nq_pad // self.block_m) + i // self.block_m
                red_max_pos_f32(Int64(mTB.iterator.toint()) + Int64(tile) * 4, rb)


class AttnBwdSm120:
    BC = 128          # key padding granularity, 16 keys per warp
    # `setmaxnreg` targets: 8 x 224 + 4 x 56 equals 12 x 168, the launch allocation of 384 threads
    MMA_REGS = 224
    DQ_REGS = 56

    def __init__(self, head_dim: int, causal: bool, block_m: int, num_stages: int = 2, skip_dq: bool = False,
                 dq_bulk: bool = True, num_warps: int = 8, min_blocks: int = 0, kb_major: bool = False,
                 dq_wg: bool = False):
        assert head_dim in (64, 128) and block_m in (32, 64) and num_warps in (4, 8)
        assert not dq_wg or (num_warps == 8 and not skip_dq and min_blocks in (0, 1)), (
            "dq_wg needs 8 MMA warps, a dQ pass and one CTA per SM")
        self.dq_wg = dq_wg
        self.kb_major = kb_major
        self.skip_dq = skip_dq
        self.dq_bulk = dq_bulk
        self.num_warps = num_warps
        self.BC = 16 * num_warps
        self.NUM_THREADS = 32 * num_warps
        self.threads = self.NUM_THREADS + (128 if dq_wg else 0)  # the dQ warps are one extra warpgroup
        self.dq_warps = 4 if dq_wg else num_warps
        self.min_blocks = min_blocks if min_blocks > 0 else 8 // num_warps
        blocks = (block_m // 16) * (head_dim // 16)
        assert blocks % self.dq_warps == 0, "dQ split needs a whole number of (m16 x 16-col) blocks per warp"
        self.nb = blocks // self.dq_warps
        self.d = head_dim
        self.causal = causal
        self.BM = block_m
        self.nt = block_m // 8
        self.stages = num_stages
        self.kv_bytes = self.BC * head_dim
        self.q_tile_bytes = block_m * head_dim
        self.ds_bytes = block_m * self.BC
        self.k_off = 0
        self.v_off = self.kv_bytes
        self.q_off = 2 * self.kv_bytes
        self.do_off = self.q_off + num_stages * self.q_tile_bytes
        self.ds_off = self.do_off + num_stages * self.q_tile_bytes
        self.dq_stage_bytes = block_m * head_dim * 4
        self.ld_stage_bytes = 2 * block_m * 4   # `L` then `D` (fp32) for one query tile
        self.ld_off = self.ds_off + 2 * self.ds_bytes
        self.dq_off = self.ld_off + num_stages * self.ld_stage_bytes
        self.smem_bytes = self.dq_off + (2 * self.dq_stage_bytes if dq_bulk else 0)
        assert (self.smem_bytes + (1024 if dq_wg else 0)) * self.min_blocks <= SMEM_LIMIT, (
            f"shared memory does not fit: {self.smem_bytes} B x {self.min_blocks} CTA(s)")
        nbytes = self.smem_bytes

        if dq_wg:
            @cute.struct
            class SharedStorage:
                # dS full and free barriers, two buffers each
                mbar: cute.struct.MemRange[cutlass.Int64, 4]
                buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]
        else:
            @cute.struct
            class SharedStorage:
                buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        self.shared_storage = SharedStorage

    def _swz_off(self, row, chunk):
        return swizzle_offset(self.d, row, chunk)

    def _ds_off(self, row, chunk):
        # dS rows are `BC` bytes, one 16-byte chunk per warp
        return swizzle_offset(self.BC, row, chunk)

    def _load_rows(self, s_addr, g_addr, nrows, tidx):
        chunks = self.d // 16
        total = nrows * chunks
        for i in range(max(1, total // self.NUM_THREADS)):
            idx = tidx + i * self.NUM_THREADS
            if total < self.NUM_THREADS:
                idx = idx & (total - 1)  # extra threads repeat an identical 16-byte copy (`total` is a power of two)
            row = idx // chunks
            ch = idx % chunks
            cp_async16(s_addr + self._swz_off(row, ch), g_addr + Int64(row * self.d + ch * 16))

    @cute.jit
    def _q_row0(self, it: Int32, n_q: Int32, q_start: Int32, b: Int32, hkv: Int32, h_q: Int32, h_kv: Int32,
                nq_pad: Int32):
        rep = h_q // h_kv
        hq = hkv * rep + it // n_q
        qi = q_start + it % n_q
        return (b * h_q + hq) * nq_pad + qi * self.BM, qi

    @cute.jit
    def _load_q(self, it: Int32, total_it: Int32, n_q: Int32, q_start: Int32, b: Int32, hkv: Int32,
                h_q: Int32, h_kv: Int32, nq_pad: Int32, sbase: Int32, gq: Int64, gdo: Int64, gl: Int64, gd: Int64,
                tidx: Int32):
        if it < total_it:
            row0, qi = self._q_row0(it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
            st = it % self.stages
            self._load_rows(sbase + self.q_off + st * self.q_tile_bytes, gq + Int64(row0) * self.d, self.BM, tidx)
            self._load_rows(sbase + self.do_off + st * self.q_tile_bytes, gdo + Int64(row0) * self.d, self.BM, tidx)
            # `L` and `D` rows: `BM` floats each, in 16-byte pieces
            pieces = self.BM // 4
            if tidx < 2 * pieces:
                which = tidx // pieces
                gsrc = (gl if which == 0 else gd) + (Int64(row0) + Int64(4 * (tidx % pieces))) * 4
                cp_async16(sbase + self.ld_off + st * self.ld_stage_bytes + which * (self.BM * 4)
                           + (tidx % pieces) * 16, gsrc)

    @cute.jit
    def _iter(
        self, it: Int32, total_it: Int32, n_q: Int32, q_start: Int32, b: Int32, hkv: Int32,
        h_q: Int32, h_kv: Int32, nq_pad: Int32, seqlen_q: Int32, seqlen_k: Int32, kblock: Int32,
        sbase: Int32, gq: Int64, gdo: Int64, gdq: Int64, gl: Int64, gd: Int64, tidx: Int32, warp: Int32, lane: Int32,
        beta: Float32, kc: Float32, v_row: Int32,
        mL: cute.Tensor, mD: cute.Tensor, mTB: cute.Tensor,
        s_acc: cute.Tensor, dp_acc: cute.Tensor, pa_t: cute.Tensor, da_t: cute.Tensor,
        dk_acc: cute.Tensor, dv_acc: cute.Tensor, dq_acc: cute.Tensor, gcur: cute.Tensor,
    ):
        d = self.d
        g = lane // 4
        t = lane % 4

        cute.arch.cp_async_wait_group(self.stages - 2)
        # wait for the stage and for every MMA warp to finish the previous iteration
        if const_expr(self.dq_wg):
            cute.arch.barrier(barrier_id=1, number_of_threads=self.NUM_THREADS)
        else:
            cute.arch.sync_threads()
        if const_expr(self.dq_bulk and not self.skip_dq and not self.dq_wg):
            if tidx == 0:
                if it > 0:
                    # previous tile's staged dQ is complete, so issue its bulk reduce
                    prow0, _ = self._q_row0(it - 1, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
                    bulk_reduce_add_f32(gdq + Int64(prow0) * (d * 4),
                                        sbase + self.dq_off + ((it - 1) % 2) * self.dq_stage_bytes,
                                        self.dq_stage_bytes)
                    cute.arch.cp_async_bulk_commit_group()
                # buffer `it % 2` is rewritten below, so wait for its last reader
                cute.arch.cp_async_bulk_wait_group(1, read=True)
        self._load_q(it + (self.stages - 1), total_it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad, sbase, gq, gdo, gl, gd, tidx)
        cute.arch.cp_async_commit_group()

        st = it % self.stages
        qb = sbase + self.q_off + st * self.q_tile_bytes
        ob = sbase + self.do_off + st * self.q_tile_bytes
        kb = sbase + self.k_off
        vb = sbase + self.v_off
        row0, qi = self._q_row0(it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)

        hq = hkv * (h_q // h_kv) + it // n_q
        tbound = mTB[(b * h_q + hq) * (nq_pad // self.BM) + qi]
        gc = gcur[0]
        # power-of-two dS scale from the tile bound; all-zero tiles keep the old one
        g_t = ds_scale(tbound) if tbound > Float32(1e-30) else gc
        kappa = g_t * kc
        dgam = g_t * Float32(2.0 ** -P_OFFSET)

        # `S^T = K_w Q^T`, `dP^T = V_w dO^T`, rows are this warp's 16 keys
        s_acc.fill(0.0)
        dp_acc.fill(0.0)
        for kk in cutlass.range_constexpr(d // 32):
            arow = warp * 16 + (lane % 8) + ((lane // 8) % 2) * 8
            ach = lane // 16 + 2 * kk
            a_k = ldsm_x4(kb + self._swz_off(arow, ach))
            a_v = ldsm_x4(vb + self._swz_off(arow, ach))
            for npair in cutlass.range_constexpr(self.nt // 2):
                brow = (lane % 8) + (lane // 16) * 8 + 16 * npair
                bch = (lane // 8) % 2 + 2 * kk
                b_q = ldsm_x4(qb + self._swz_off(brow, bch))
                b_o = ldsm_x4(ob + self._swz_off(brow, bch))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_fp8(a_k, [b_q[2 * h], b_q[2 * h + 1]],
                                [s_acc[4 * nt], s_acc[4 * nt + 1], s_acc[4 * nt + 2], s_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        s_acc[4 * nt + x] = c[x]
                    c = mma_fp8(a_v, [b_o[2 * h], b_o[2 * h + 1]],
                                [dp_acc[4 * nt], dp_acc[4 * nt + 1], dp_acc[4 * nt + 2], dp_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        dp_acc[4 * nt + x] = c[x]


        # `P8 = 2^(s*beta - L + 8)`, `dS*g = P8 * (dP*kappa - D*g*2^-8)`
        key0 = kblock * self.BC + warp * 16 + g
        diag = seqlen_k - seqlen_q
        ldb = sbase + self.ld_off + st * self.ld_stage_bytes
        for nt in cutlass.range_constexpr(self.nt):
            l2 = ld_shared_v2_f32(ldb + (8 * nt + 2 * t) * 4)
            d2 = ld_shared_v2_f32(ldb + self.BM * 4 + (8 * nt + 2 * t) * 4)
            for e in cutlass.range_constexpr(2):
                q_loc = 8 * nt + 2 * t + e
                lq = l2[e]
                dq = d2[e] * dgam
                for r in cutlass.range_constexpr(2):
                    idx = 4 * nt + 2 * r + e
                    p8 = cute.math.exp2(s_acc[idx] * beta - lq + Float32(P_OFFSET), fastmath=True)
                    p8 = p8 if key0 + 8 * r < seqlen_k else Float32(0.0)
                    if const_expr(self.causal):
                        p8 = p8 if key0 + 8 * r <= qi * self.BM + q_loc + diag else Float32(0.0)
                    s_acc[idx] = p8
                    dp_acc[idx] = p8 * (dp_acc[idx] * kappa - dq)


        # dS to shared memory now; the fp32 S/dP fragments die once packed
        dsb = sbase + self.ds_off + (it % 2) * self.ds_bytes
        if const_expr(not self.skip_dq):
            if const_expr(self.dq_wg):
                # the dQ warps are done reading this buffer's previous tile
                cute.arch.mbarrier_wait(self._ds_free + it % 2, ((it // 2) & 1) ^ 1)
            for nt in cutlass.range_constexpr(self.nt):
                for e in cutlass.range_constexpr(2):
                    q_loc = 8 * nt + 2 * t + e
                    a_lo = dsb + self._ds_off(q_loc, warp) + g
                    st_shared_e4m3_pair(a_lo, a_lo + 8, dp_acc[4 * nt + e], dp_acc[4 * nt + 2 + e])
            if const_expr(self.dq_wg):
                cute.arch.mbarrier_arrive(self._ds_full + it % 2)
        for kk2 in cutlass.range_constexpr(self.BM // 32):
            t0 = 4 * kk2
            pa_t[4 * kk2] = pack_e4m3x4(s_acc[4 * t0], s_acc[4 * t0 + 1], s_acc[4 * t0 + 4], s_acc[4 * t0 + 5])
            pa_t[4 * kk2 + 1] = pack_e4m3x4(s_acc[4 * t0 + 2], s_acc[4 * t0 + 3], s_acc[4 * t0 + 6], s_acc[4 * t0 + 7])
            pa_t[4 * kk2 + 2] = pack_e4m3x4(s_acc[4 * t0 + 8], s_acc[4 * t0 + 9], s_acc[4 * t0 + 12], s_acc[4 * t0 + 13])
            pa_t[4 * kk2 + 3] = pack_e4m3x4(s_acc[4 * t0 + 10], s_acc[4 * t0 + 11], s_acc[4 * t0 + 14],
                                            s_acc[4 * t0 + 15])
            da_t[4 * kk2] = pack_e4m3x4(dp_acc[4 * t0], dp_acc[4 * t0 + 1], dp_acc[4 * t0 + 4], dp_acc[4 * t0 + 5])
            da_t[4 * kk2 + 1] = pack_e4m3x4(dp_acc[4 * t0 + 2], dp_acc[4 * t0 + 3], dp_acc[4 * t0 + 6],
                                            dp_acc[4 * t0 + 7])
            da_t[4 * kk2 + 2] = pack_e4m3x4(dp_acc[4 * t0 + 8], dp_acc[4 * t0 + 9], dp_acc[4 * t0 + 12],
                                            dp_acc[4 * t0 + 13])
            da_t[4 * kk2 + 3] = pack_e4m3x4(dp_acc[4 * t0 + 10], dp_acc[4 * t0 + 11], dp_acc[4 * t0 + 14],
                                            dp_acc[4 * t0 + 15])

        # rescale dK to the new scale (exact, powers of two)
        if g_t != gc:
            ratio = div_rn(g_t, gc)
            for e in cutlass.range_constexpr(4 * (d // 8)):
                dk_acc[e] = dk_acc[e] * ratio
            gcur[0] = g_t

        # `dV += P^T dO`, `dK += dS^T Q`; B via transposing 8-bit `ldmatrix`
        for kk2 in cutlass.range_constexpr(self.BM // 32):
            pa = [pa_t[4 * kk2], pa_t[4 * kk2 + 1], pa_t[4 * kk2 + 2], pa_t[4 * kk2 + 3]]
            da = [da_t[4 * kk2], da_t[4 * kk2 + 1], da_t[4 * kk2 + 2], da_t[4 * kk2 + 3]]
            qrow = v_row + 32 * kk2
            for npair in cutlass.range_constexpr(d // 16):
                bo = ldsm_x2_trans_b8(ob + self._swz_off(qrow, npair))
                bq = ldsm_x2_trans_b8(qb + self._swz_off(qrow, npair))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_fp8(pa, [bo[h], bo[2 + h]],
                                [dv_acc[4 * nt], dv_acc[4 * nt + 1], dv_acc[4 * nt + 2], dv_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        dv_acc[4 * nt + x] = c[x]
                    c = mma_fp8(da, [bq[h], bq[2 + h]],
                                [dk_acc[4 * nt], dk_acc[4 * nt + 1], dk_acc[4 * nt + 2], dk_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        dk_acc[4 * nt + x] = c[x]

        if const_expr(not self.skip_dq and not self.dq_wg):
            self._dq(it, row0, sbase, dsb, gdq, warp, lane, dq_acc)

    @cute.jit
    def _dq(self, it: Int32, row0: Int32, sbase: Int32, dsb: Int32, gdq: Int64, warp: Int32, lane: Int32,
            dq_acc: cute.Tensor):
        d = self.d
        g = lane // 4
        t = lane % 4
        kb = sbase + self.k_off
        # wait for every warp's dS; placed after the dV/dK GEMMs so they overlap
        if const_expr(not self.dq_wg):
            cute.arch.sync_threads()

        # each warp reduces `nb` blocks of 16 queries x 16 columns
        dchunks = d // 16
        sdq = sbase + self.dq_off + (it % 2) * self.dq_stage_bytes
        for i in cutlass.range_constexpr(self.nb):
            blk = warp + self.dq_warps * i
            mt = blk // dchunks
            ch = blk % dchunks
            dq_acc.fill(0.0)
            for ks in cutlass.range_constexpr(self.BC // 32):
                arow = mt * 16 + (lane % 8) + ((lane // 8) % 2) * 8
                a = ldsm_x4(dsb + self._ds_off(arow, lane // 16 + 2 * ks))
                bk = ldsm_x2_trans_b8(kb + self._swz_off(lane + 32 * ks, ch))
                for h in cutlass.range_constexpr(2):
                    c = mma_fp8(a, [bk[h], bk[2 + h]],
                                [dq_acc[4 * h], dq_acc[4 * h + 1], dq_acc[4 * h + 2], dq_acc[4 * h + 3]])
                    for x in cutlass.range_constexpr(4):
                        dq_acc[4 * h + x] = c[x]
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    if const_expr(self.dq_bulk):
                        # conflict-free fragment order, undone by `BwdDqPost`
                        slot = (((warp * self.nb + i) * 2 + h) * 2 + r) * 32 + lane
                        st_shared_v2_f32(sdq + slot * 8, dq_acc[4 * h + 2 * r], dq_acc[4 * h + 2 * r + 1])
                    else:
                        q_loc = mt * 16 + g + 8 * r
                        col = ch * 16 + 8 * h + 2 * t
                        addr = gdq + (Int64(row0 + q_loc) * d + Int64(col)) * 4
                        red_add_v2_f32(addr, dq_acc[4 * h + 2 * r], dq_acc[4 * h + 2 * r + 1])
        if const_expr(self.dq_bulk):
            # make staging writes visible to the bulk-copy proxy
            cute.arch.fence_view_async_shared()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mdO: cute.Tensor,
        mL: cute.Tensor, mD: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDesc: cute.Tensor,
        mdK: cute.Tensor, mdV: cute.Tensor, mdQacc: cute.Tensor, softmax_scale: Float32,
        seqlen_q: Int32, seqlen_k: Int32, nq_pad: Int32, nk_pad: Int32, h_q: Int32, h_kv: Int32, batch: Int32,
    ):
        self.kernel(
            mQ, mK, mV, mdO, mL, mD, mTB, mStats, mDesc, mdK, mdV, mdQacc, softmax_scale,
            seqlen_q, seqlen_k, nq_pad, nk_pad, h_q, h_kv,
        ).launch(
            # key block is the slowest axis, so long causal CTAs launch first
            grid=[h_kv * batch, nk_pad // self.BC, 1] if self.kb_major else [nk_pad // self.BC, h_kv, batch],
            block=[self.threads, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=self.min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mdO: cute.Tensor,
        mL: cute.Tensor, mD: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDesc: cute.Tensor,
        mdK: cute.Tensor, mdV: cute.Tensor, mdQacc: cute.Tensor, softmax_scale: Float32,
        seqlen_q: Int32, seqlen_k: Int32, nq_pad: Int32, nk_pad: Int32, h_q: Int32, h_kv: Int32,
    ):
        d = self.d
        tidx, _, _ = cute.arch.thread_idx()
        if const_expr(self.kb_major):
            lin, kblock, _ = cute.arch.block_idx()
            hkv = lin % h_kv
            b = lin // h_kv
        else:
            kblock, hkv, b = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32

        beta, kc, s_dk0, s_dv, _ = bwd_scales(mStats, mDesc, b, hkv, softmax_scale)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        sbase = Int32(storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint())

        k_row0 = (b * h_kv + hkv) * nk_pad + kblock * self.BC
        gk = Int64(mK.iterator.toint()) + Int64(k_row0) * d
        gv = Int64(mV.iterator.toint()) + Int64(k_row0) * d
        gq = Int64(mQ.iterator.toint())
        gdo = Int64(mdO.iterator.toint())
        gdq = Int64(mdQacc.iterator.toint())
        gl = Int64(mL.iterator.toint())
        gd = Int64(mD.iterator.toint())

        # query tiles this key block sees; causal keeps key `j` for query `i` iff `j <= i + Nk - Nq`
        nq_tiles = nq_pad // self.BM
        q_start = Int32(0)
        if const_expr(self.causal):
            first_q = kblock * self.BC + seqlen_q - seqlen_k
            qs = first_q // self.BM if first_q > 0 else Int32(0)
            q_start = min(nq_tiles, qs)
        n_q = nq_tiles - q_start
        total_it = (h_q // h_kv) * n_q

        if const_expr(self.dq_wg):
            self._ds_full = storage.mbar.data_ptr()
            self._ds_free = self._ds_full + 2
            if tidx == 0:
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(self._ds_full + i, self.NUM_THREADS)
                    cute.arch.mbarrier_init(self._ds_free + i, 128)
            cute.arch.mbarrier_init_fence()
            if warp < self.num_warps:
                self._prologue(sbase, gk, gv, gq, gdo, gl, gd, total_it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad, tidx)
                cute.arch.cp_async_wait_group(self.stages - 2)
            # K, V and the first query tile are visible to the dQ warps too
            cute.arch.sync_threads()
            if warp >= self.num_warps:
                cute.arch.setmaxregister_decrease(self.DQ_REGS)
                self._dq_loop(tidx, warp, lane, total_it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad, sbase, gdq)
            else:
                cute.arch.setmaxregister_increase(self.MMA_REGS)
                self._main(tidx, warp, lane, kblock, hkv, b, total_it, n_q, q_start, h_q, h_kv, nq_pad, seqlen_q,
                           seqlen_k, sbase, gq, gdo, gdq, gl, gd, beta, kc, s_dk0, s_dv, k_row0, mL, mD, mTB, mdK, mdV)
        else:
            self._prologue(sbase, gk, gv, gq, gdo, gl, gd, total_it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad, tidx)
            self._main(tidx, warp, lane, kblock, hkv, b, total_it, n_q, q_start, h_q, h_kv, nq_pad, seqlen_q,
                       seqlen_k, sbase, gq, gdo, gdq, gl, gd, beta, kc, s_dk0, s_dv, k_row0, mL, mD, mTB, mdK, mdV)

    @cute.jit
    def _prologue(self, sbase: Int32, gk: Int64, gv: Int64, gq: Int64, gdo: Int64, gl: Int64, gd: Int64,
                  total_it: Int32, n_q: Int32, q_start: Int32, b: Int32, hkv: Int32, h_q: Int32, h_kv: Int32,
                  nq_pad: Int32, tidx: Int32):
        self._load_rows(sbase + self.k_off, gk, self.BC, tidx)
        self._load_rows(sbase + self.v_off, gv, self.BC, tidx)
        cute.arch.cp_async_commit_group()
        for s in cutlass.range_constexpr(self.stages - 1):
            self._load_q(Int32(s), total_it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad, sbase, gq, gdo, gl, gd, tidx)
            cute.arch.cp_async_commit_group()

    @cute.jit
    def _dq_loop(self, tidx: Int32, warp: Int32, lane: Int32, total_it: Int32, n_q: Int32, q_start: Int32, b: Int32,
                 hkv: Int32, h_q: Int32, h_kv: Int32, nq_pad: Int32, sbase: Int32, gdq: Int64):
        d = self.d
        dwarp = warp - self.num_warps
        dtid = tidx - self.NUM_THREADS
        dq_acc = cute.make_rmem_tensor(8, Float32)
        for it in cutlass.range(total_it, unroll=1):
            if const_expr(self.dq_bulk):
                # every dQ warp has staged the previous tile
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if dtid == 0:
                    if it > 0:
                        prow0, _ = self._q_row0(it - 1, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
                        bulk_reduce_add_f32(gdq + Int64(prow0) * (d * 4),
                                            sbase + self.dq_off + ((it - 1) % 2) * self.dq_stage_bytes,
                                            self.dq_stage_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                    # buffer `it % 2` is rewritten below, so wait for its last reader
                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
            cute.arch.mbarrier_wait(self._ds_full + it % 2, (it // 2) & 1)
            row0, _ = self._q_row0(it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
            dsb = sbase + self.ds_off + (it % 2) * self.ds_bytes
            self._dq(it, row0, sbase, dsb, gdq, dwarp, lane, dq_acc)
            cute.arch.mbarrier_arrive(self._ds_free + it % 2)
        if const_expr(self.dq_bulk):
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if dtid == 0:
                if total_it > 0:
                    lrow0, _ = self._q_row0(total_it - 1, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
                    bulk_reduce_add_f32(gdq + Int64(lrow0) * (d * 4),
                                        sbase + self.dq_off + ((total_it - 1) % 2) * self.dq_stage_bytes,
                                        self.dq_stage_bytes)
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _main(self, tidx: Int32, warp: Int32, lane: Int32, kblock: Int32, hkv: Int32, b: Int32, total_it: Int32,
              n_q: Int32, q_start: Int32, h_q: Int32, h_kv: Int32, nq_pad: Int32, seqlen_q: Int32, seqlen_k: Int32,
              sbase: Int32, gq: Int64, gdo: Int64, gdq: Int64, gl: Int64, gd: Int64, beta: Float32, kc: Float32,
              s_dk0: Float32, s_dv: Float32, k_row0: Int32, mL: cute.Tensor, mD: cute.Tensor, mTB: cute.Tensor,
              mdK: cute.Tensor, mdV: cute.Tensor):
        d = self.d
        g = lane // 4
        t = lane % 4

        s_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dp_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dk_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        dv_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        dq_acc = cute.make_rmem_tensor(8, Float32)
        pa_t = cute.make_rmem_tensor(4 * (self.BM // 32), Int32)
        da_t = cute.make_rmem_tensor(4 * (self.BM // 32), Int32)
        dk_acc.fill(0.0)
        dv_acc.fill(0.0)
        gcur = cute.make_rmem_tensor(1, Float32)
        gcur.fill(1.0)

        li = lane % 16
        v_row = ((li % 4) // 2) * 8 + (li // 4) * 2 + (li % 2) + (lane // 16) * 16

        for it in cutlass.range(total_it, unroll=1):
            self._iter(it, total_it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad, seqlen_q, seqlen_k, kblock,
                       sbase, gq, gdo, gdq, gl, gd, tidx, warp, lane, beta, kc, v_row,
                       mL, mD, mTB, s_acc, dp_acc, pa_t, da_t, dk_acc, dv_acc, dq_acc, gcur)

        if const_expr(self.dq_bulk and not self.skip_dq and not self.dq_wg):
            cute.arch.sync_threads()
            if tidx == 0:
                if total_it > 0:
                    lrow0, _ = self._q_row0(total_it - 1, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
                    bulk_reduce_add_f32(gdq + Int64(lrow0) * (d * 4),
                                        sbase + self.dq_off + ((total_it - 1) % 2) * self.dq_stage_bytes,
                                        self.dq_stage_bytes)
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

        # `dK = alpha*sigma_Q/gcur * A^K`, `dV = sigma_dO*2^-8 * A^V`, bf16, one row per key
        s_dk = s_dk0 * div_rn(Float32(1.0), gcur[0])
        gdk = Int64(mdK.iterator.toint())
        gdv = Int64(mdV.iterator.toint())
        for r in cutlass.range_constexpr(2):
            key = kblock * self.BC + warp * 16 + g + 8 * r
            if key < seqlen_k:
                krow = Int64(k_row0 + warp * 16 + g + 8 * r)
                for nt in cutlass.range_constexpr(d // 8):
                    off = (krow * d + Int64(8 * nt + 2 * t)) * 2
                    st_global_b32(gdk + off, pack_half2(dk_acc[4 * nt + 2 * r] * s_dk, dk_acc[4 * nt + 2 * r + 1] * s_dk, "bf16"))
                    st_global_b32(gdv + off, pack_half2(dv_acc[4 * nt + 2 * r] * s_dv, dv_acc[4 * nt + 2 * r + 1] * s_dv, "bf16"))


@cute.jit
def bwd_scales(mStats: cute.Tensor, mDesc: cute.Tensor, b: Int32, hkv: Int32, softmax_scale: Float32):
    sig_q = mDesc[b, hkv, 0]
    sig_k = mDesc[b, hkv, 1]
    sig_v = mDesc[b, hkv, 2]
    amax = mStats[b, hkv, 0]
    s_do = amax * Float32(1.0 / 256.0) if amax > Float32(0.0) else Float32(1.0)
    # `(beta, kc, dK scale, dV scale, dQ scale)`; `kappa = kc * g_tile`
    return (
        softmax_scale * sig_q * sig_k * LOG2E,
        s_do * sig_v * Float32(2.0 ** -P_OFFSET),
        softmax_scale * sig_q,
        s_do * Float32(2.0 ** -P_OFFSET),
        softmax_scale * sig_k,
    )


class BwdDqPost:
    NUM_THREADS = 256

    def __init__(self, head_dim: int, block_m: int, staged: bool, num_warps: int = 8):
        self.d = head_dim
        self.BM = block_m
        self.staged = staged
        self.num_warps = num_warps
        self.nb = (block_m // 16) * (head_dim // 16) // num_warps
        self.tile_floats = block_m * head_dim
        self.pairs_per_thread = self.tile_floats // 2 // self.NUM_THREADS
        nfl = self.tile_floats

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, nfl], 16]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(self, mAcc: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDesc: cute.Tensor,
                 mdQ: cute.Tensor, softmax_scale: Float32, seqlen_q: Int32, nq_pad: Int32, h_q: Int32, h_kv: Int32,
                 batch: Int32):
        self.kernel(mAcc, mTB, mStats, mDesc, mdQ, softmax_scale, seqlen_q, nq_pad, h_q, h_kv).launch(
            grid=[nq_pad // self.BM, h_q, batch], block=[self.NUM_THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes() if const_expr(self.staged) else 0)

    @cute.kernel
    def kernel(self, mAcc: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDesc: cute.Tensor,
               mdQ: cute.Tensor, softmax_scale: Float32, seqlen_q: Int32, nq_pad: Int32, h_q: Int32, h_kv: Int32):
        d = self.d
        tidx, _, _ = cute.arch.thread_idx()
        tile, hq, b = cute.arch.block_idx()
        hkv = hq // (h_q // h_kv)
        _, _, _, _, s_dq0 = bwd_scales(mStats, mDesc, b, hkv, softmax_scale)
        # dS scale this tile was accumulated at, derived from the same bound
        s_dq = s_dq0 * div_rn(Float32(1.0), ds_scale(mTB[(b * h_q + hq) * (nq_pad // self.BM) + tile]))
        acc0 = ((b * h_q + hq) * nq_pad + tile * self.BM) * d
        gout = Int64(mdQ.iterator.toint())
        dchunks = d // 16
        if const_expr(self.staged):
            # coalesced copy into shared memory, then coalesced row-major writes
            storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
            sacc = storage.buf.get_tensor(cute.make_layout(self.tile_floats))
            for k in cutlass.range_constexpr(self.tile_floats // self.NUM_THREADS):
                idx = tidx + self.NUM_THREADS * k
                sacc[idx] = mAcc[acc0 + idx]
            cute.arch.sync_threads()
        for k in cutlass.range_constexpr(self.pairs_per_thread):
            p = tidx + self.NUM_THREADS * k
            q_loc = p // (d // 2)
            col = (p % (d // 2)) * 2
            if const_expr(self.staged):
                # inverse of `slot = ((((warp * nb + i) * 2 + h) * 2 + r) * 32 + lane)` in `_dq`
                blk = (q_loc // 16) * dchunks + col // 16
                warp = blk % self.num_warps
                i = blk // self.num_warps
                h = (col % 16) // 8
                r = (q_loc % 16) // 8
                lane = (q_loc % 8) * 4 + (col % 8) // 2
                slot = (((warp * self.nb + i) * 2 + h) * 2 + r) * 32 + lane
                x0 = sacc[2 * slot]
                x1 = sacc[2 * slot + 1]
            else:
                x0 = mAcc[acc0 + 2 * p]
                x1 = mAcc[acc0 + 2 * p + 1]
            qg = tile * self.BM + q_loc
            if qg < seqlen_q:
                addr = gout + (Int64((b * h_q + hq) * seqlen_q + qg) * d + Int64(col)) * 2
                st_global_b32(addr, pack_half2(x0 * s_dq, x1 * s_dq, "bf16"))


class AttnBwdDqSm120:
    BM = 128
    BN = 64
    NUM_THREADS = 256

    def __init__(self, head_dim: int, causal: bool, tb_block_m: int, num_stages: int = 2):
        assert head_dim in (64, 128)
        self.tb_block_m = tb_block_m
        self.d = head_dim
        self.causal = causal
        self.stages = num_stages
        self.nt = self.BN // 8
        self.q_bytes = self.BM * head_dim
        self.kv_tile_bytes = self.BN * head_dim
        self.q_off = 0
        self.do_off = self.q_bytes
        self.k_off = 2 * self.q_bytes
        self.v_off = self.k_off + num_stages * self.kv_tile_bytes
        self.smem_bytes = 2 * self.q_bytes + 2 * num_stages * self.kv_tile_bytes
        assert self.smem_bytes <= SMEM_LIMIT
        nbytes = self.smem_bytes

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        self.shared_storage = SharedStorage

    def _swz_off(self, row, chunk):
        return swizzle_offset(self.d, row, chunk)

    def _load_rows(self, s_addr, g_addr, nrows, tidx):
        chunks = self.d // 16
        for i in range(nrows * chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // chunks
            ch = idx % chunks
            cp_async16(s_addr + self._swz_off(row, ch), g_addr + Int64(row * self.d + ch * 16))

    @cute.jit
    def _load_tile(self, jn: Int32, n_blocks: Int32, sbase: Int32, gk: Int64, gv: Int64, tidx: Int32):
        if jn < n_blocks:
            sn = jn % self.stages
            self._load_rows(sbase + self.k_off + sn * self.kv_tile_bytes,
                            gk + Int64(jn) * (self.BN * self.d), self.BN, tidx)
            self._load_rows(sbase + self.v_off + sn * self.kv_tile_bytes,
                            gv + Int64(jn) * (self.BN * self.d), self.BN, tidx)

    @cute.jit
    def _tile(
        self, j: Int32, n_blocks: Int32, sbase: Int32, gk: Int64, gv: Int64, tidx: Int32, lane: Int32, t: Int32,
        beta: Float32, kappa: Float32, l0: Float32, l1: Float32, dk0: Float32, dk1: Float32,
        lim0: Int32, lim1: Int32, v_key: Int32,
        qf: cute.Tensor, of: cute.Tensor, s_acc: cute.Tensor, dp_acc: cute.Tensor, dq_acc: cute.Tensor,
    ):
        d = self.d
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()
        self._load_tile(j + (self.stages - 1), n_blocks, sbase, gk, gv, tidx)
        cute.arch.cp_async_commit_group()
        kbase = sbase + self.k_off + (j % self.stages) * self.kv_tile_bytes
        vbase = sbase + self.v_off + (j % self.stages) * self.kv_tile_bytes

        # `S = Q K^T`, `dP = dO V^T`, rows are this warp's 16 queries
        s_acc.fill(0.0)
        dp_acc.fill(0.0)
        for kk in cutlass.range_constexpr(d // 32):
            for npair in cutlass.range_constexpr(self.nt // 2):
                key = (lane % 8) + (lane // 16) * 8 + 16 * npair
                ch = (lane // 8) % 2 + 2 * kk
                bk = ldsm_x4(kbase + self._swz_off(key, ch))
                bv = ldsm_x4(vbase + self._swz_off(key, ch))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_fp8([qf[4 * kk], qf[4 * kk + 1], qf[4 * kk + 2], qf[4 * kk + 3]],
                                [bk[2 * h], bk[2 * h + 1]],
                                [s_acc[4 * nt], s_acc[4 * nt + 1], s_acc[4 * nt + 2], s_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        s_acc[4 * nt + x] = c[x]
                    c = mma_fp8([of[4 * kk], of[4 * kk + 1], of[4 * kk + 2], of[4 * kk + 3]],
                                [bv[2 * h], bv[2 * h + 1]],
                                [dp_acc[4 * nt], dp_acc[4 * nt + 1], dp_acc[4 * nt + 2], dp_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        dp_acc[4 * nt + x] = c[x]

        # `dS*g = P8 * (dP*kappa - D*g*2^-8)`; masked keys get `P8 = 0`
        kb = j * self.BN + t * 2
        for r in cutlass.range_constexpr(2):
            lq = l0 if r == 0 else l1
            dq = dk0 if r == 0 else dk1
            lim = lim0 if r == 0 else lim1
            for nt in cutlass.range_constexpr(self.nt):
                for e in cutlass.range_constexpr(2):
                    idx = 4 * nt + 2 * r + e
                    p8 = cute.math.exp2(s_acc[idx] * beta - lq + Float32(P_OFFSET), fastmath=True)
                    p8 = p8 if kb + (8 * nt + e) < lim else Float32(0.0)
                    dp_acc[idx] = p8 * (dp_acc[idx] * kappa - dq)

        # `dQ += dS K`; dS packed from the accumulators, K via transposing 8-bit `ldmatrix`
        for kk2 in cutlass.range_constexpr(self.BN // 32):
            t0 = 4 * kk2
            da = [
                pack_e4m3x4(dp_acc[4 * t0], dp_acc[4 * t0 + 1], dp_acc[4 * t0 + 4], dp_acc[4 * t0 + 5]),
                pack_e4m3x4(dp_acc[4 * t0 + 2], dp_acc[4 * t0 + 3], dp_acc[4 * t0 + 6], dp_acc[4 * t0 + 7]),
                pack_e4m3x4(dp_acc[4 * t0 + 8], dp_acc[4 * t0 + 9], dp_acc[4 * t0 + 12], dp_acc[4 * t0 + 13]),
                pack_e4m3x4(dp_acc[4 * t0 + 10], dp_acc[4 * t0 + 11], dp_acc[4 * t0 + 14], dp_acc[4 * t0 + 15]),
            ]
            key = v_key + 32 * kk2
            for npair in cutlass.range_constexpr(d // 16):
                bk = ldsm_x2_trans_b8(kbase + self._swz_off(key, npair))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_fp8(da, [bk[h], bk[2 + h]],
                                [dq_acc[4 * nt], dq_acc[4 * nt + 1], dq_acc[4 * nt + 2], dq_acc[4 * nt + 3]])
                    for x in cutlass.range_constexpr(4):
                        dq_acc[4 * nt + x] = c[x]

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mdO: cute.Tensor,
        mL: cute.Tensor, mD: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDesc: cute.Tensor,
        mdQ: cute.Tensor, softmax_scale: Float32, seqlen_q: Int32, seqlen_k: Int32, nq_pad: Int32, nk_pad: Int32,
        h_q: Int32, h_kv: Int32, batch: Int32,
    ):
        self.kernel(
            mQ, mK, mV, mdO, mL, mD, mTB, mStats, mDesc, mdQ, softmax_scale,
            seqlen_q, seqlen_k, nq_pad, nk_pad, h_q, h_kv,
        ).launch(
            grid=[nq_pad // self.BM, h_q, batch],
            block=[self.NUM_THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mdO: cute.Tensor,
        mL: cute.Tensor, mD: cute.Tensor, mTB: cute.Tensor, mStats: cute.Tensor, mDesc: cute.Tensor,
        mdQ: cute.Tensor, softmax_scale: Float32, seqlen_q: Int32, seqlen_k: Int32, nq_pad: Int32, nk_pad: Int32,
        h_q: Int32, h_kv: Int32,
    ):
        d = self.d
        kq_steps = d // 32
        tidx, _, _ = cute.arch.thread_idx()
        m_block, hq, b = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        g = lane // 4
        t = lane % 4
        hkv = hq // (h_q // h_kv)
        beta, kc, _, _, s_dq0 = bwd_scales(mStats, mDesc, b, hkv, softmax_scale)

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        sbase = Int32(storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint())

        q_row0 = (b * h_q + hq) * nq_pad + m_block * self.BM
        k_row0 = (b * h_kv + hkv) * nk_pad
        gq = Int64(mQ.iterator.toint()) + Int64(q_row0) * d
        gdo = Int64(mdO.iterator.toint()) + Int64(q_row0) * d
        gk = Int64(mK.iterator.toint()) + Int64(k_row0) * d
        gv = Int64(mV.iterator.toint()) + Int64(k_row0) * d

        n_blocks = nk_pad // self.BN
        if const_expr(self.causal):
            kmax = m_block * self.BM + seqlen_k - seqlen_q + (self.BM - 1)
            nb = (kmax + self.BN) // self.BN
            nb = nb if nb > 0 else Int32(0)
            n_blocks = min(n_blocks, nb)

        self._load_rows(sbase + self.q_off, gq, self.BM, tidx)
        self._load_rows(sbase + self.do_off, gdo, self.BM, tidx)
        cute.arch.cp_async_commit_group()
        for s in cutlass.range_constexpr(self.stages - 1):
            self._load_tile(Int32(s), n_blocks, sbase, gk, gv, tidx)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()

        qf = cute.make_rmem_tensor(4 * kq_steps, Int32)
        of = cute.make_rmem_tensor(4 * kq_steps, Int32)
        for kq in cutlass.range_constexpr(kq_steps):
            arow = (lane % 8) + ((lane // 8) % 2) * 8 + warp * 16
            ach = lane // 16 + 2 * kq
            qfrag = ldsm_x4(sbase + self.q_off + self._swz_off(arow, ach))
            ofrag = ldsm_x4(sbase + self.do_off + self._swz_off(arow, ach))
            for x in cutlass.range_constexpr(4):
                qf[4 * kq + x] = qfrag[x]
                of[4 * kq + x] = ofrag[x]

        lrow = q_row0 + warp * 16 + g
        l0 = mL[lrow]
        l1 = mL[lrow + 8]
        # the warp's 16 query rows share one prep tile, so the dS scale is fixed over the key loop
        tq = m_block * self.BM + warp * 16
        gq_s = ds_scale(mTB[(b * h_q + hq) * (nq_pad // self.tb_block_m) + tq // self.tb_block_m])
        kappa = gq_s * kc
        dgam = gq_s * Float32(2.0 ** -P_OFFSET)
        s_dq = s_dq0 * div_rn(Float32(1.0), gq_s)
        dk0 = mD[lrow] * dgam
        dk1 = mD[lrow + 8] * dgam
        q_idx0 = m_block * self.BM + warp * 16 + g
        lim0 = seqlen_k
        lim1 = seqlen_k
        if const_expr(self.causal):
            lr0 = q_idx0 + (seqlen_k - seqlen_q + 1)
            lr1 = lr0 + 8
            lim0 = min(seqlen_k, lr0)
            lim1 = min(seqlen_k, lr1)
        li = lane % 16
        v_key = ((li % 4) // 2) * 8 + (li // 4) * 2 + (li % 2) + (lane // 16) * 16

        s_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dp_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dq_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        dq_acc.fill(0.0)
        for j in cutlass.range(n_blocks, unroll=1):
            self._tile(j, n_blocks, sbase, gk, gv, tidx, lane, t, beta, kappa, l0, l1, dk0, dk1,
                       lim0, lim1, v_key, qf, of, s_acc, dp_acc, dq_acc)

        gdq = Int64(mdQ.iterator.toint())
        for r in cutlass.range_constexpr(2):
            if q_idx0 + 8 * r < seqlen_q:
                orow = Int64(lrow + 8 * r)
                for nt in cutlass.range_constexpr(d // 8):
                    addr = gdq + (orow * d + Int64(8 * nt + 2 * t)) * 2
                    st_global_b32(addr, pack_half2(dq_acc[4 * nt + 2 * r] * s_dq, dq_acc[4 * nt + 2 * r + 1] * s_dq, "bf16"))


def make_attn_bwd(q, k, v, out, lse, pmax, dout, q_descale=None, k_descale=None, v_descale=None,
                      causal=False, softmax_scale=None, num_stages=None, dq_bulk=None,
                      deterministic=False, num_warps=8, block_m=None, min_blocks=None, kb_major=None,
                      dq_wg=None):
    """Prepare the attention backward on SM120.

    Args:
        q: The forward's E4M3 ``q``, ``[B, Hq, Nq, d]``.
        k: The forward's E4M3 ``k``, ``[B, Hkv, Nk, d]``.
        v: The forward's E4M3 ``v``, ``[B, Hkv, Nk, d]``.
        out: The forward's ``out`` (bf16).
        lse: The forward's base-2 ``lse``.
        pmax: The forward's largest probability per row.
        dout: Gradient of ``out``, ``[B, Hq, Nq, d]``. Read as bf16; other dtypes are converted first.
        q_descale: ``[B, Hkv]`` float32 dequantization scale, or ``None`` for 1.
        k_descale: Same as ``q_descale``, for ``k``.
        v_descale: Same as ``q_descale``, for ``v``.
        causal: Whether to apply a bottom-right aligned causal mask.
        softmax_scale: Scale on the logits. Defaults to ``1 / sqrt(d)``.
        num_stages: Depth of the Q/dO shared-memory pipeline. ``None`` takes the autotuned or built-in choice.
            See ``autotune_attn_bwd``.
        dq_bulk: Whether to reduce dQ with one bulk TMA reduce-add per tile instead of per-element atomics.
            ``None`` takes the autotuned or built-in choice.
        deterministic: Whether to compute dQ in a separate kernel without atomics. This is bitwise
            reproducible but recomputes S and dP.
        num_warps: 8 for one 128-key CTA per SM, or 4 for two 64-key CTAs per SM (dQ then uses atomics).
        block_m: Queries per inner iteration, 32 or 64. ``None`` takes the autotuned or built-in choice.
        min_blocks: CTAs per SM to request from the compiler. 2 caps registers at 128 per thread.
        dq_wg: Whether four extra warps compute dQ, overlapping it with the next tile's MMAs. Needs 8 MMA warps and
            one CTA per SM, so ``min_blocks`` is forced to 1. Defaults to the tuned value, else ``False``.
        kb_major: Whether to launch key blocks slowest, so the longest causal CTAs go first. Defaults to
            ``True`` for causal attention and ``False`` otherwise.

    Returns:
        ``(launch, finish)``. ``launch()`` quantizes ``dout`` and runs the backward kernels. ``finish()`` then
        returns ``(dq, dk, dv)`` in bf16: gradients with respect to the dequantized ``q``, ``k`` and ``v``.
    """
    B, Hq, Nq, d = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    if kb_major is None:
        kb_major = causal
    dev = q.device
    ones = torch.ones(B, Hkv, dtype=torch.float32, device=dev)
    sq, sk, sv = [(x if x is not None else ones).float() for x in (q_descale, k_descale, v_descale)]
    desc = torch.stack([sq, sk, sv], dim=-1).contiguous()
    from_cache = num_stages is None and dq_bulk is None and block_m is None
    if num_stages is None or dq_bulk is None or block_m is None:
        tuned = tune_lookup(tune_key("bwd", d, causal, Nq, Nk, Hq // Hkv)) or dict(
            num_stages=2, dq_bulk=causal or d != 128, block_m=32 if d == 128 else 64)
        num_stages = tuned["num_stages"] if num_stages is None else num_stages
        dq_bulk = tuned["dq_bulk"] if dq_bulk is None else dq_bulk
        block_m = tuned["block_m"] if block_m is None else block_m
        if dq_wg is None and from_cache:
            dq_wg = tuned.get("dq_wg", False)
    if min_blocks is None:
        # never reuse a cached `min_blocks` with a config it was not tuned for
        tuned = (tune_lookup(tune_key("bwd", d, causal, Nq, Nk, Hq // Hkv)) or {}) if from_cache else {}
        min_blocks = tuned.get("min_blocks", 0)
    if deterministic or num_warps == 4:
        dq_bulk = False
    dq_wg = bool(dq_wg) and not deterministic and num_warps == 8
    if dq_wg:
        min_blocks = 1
    nq_pad = round_up(Nq, 128)
    nk_pad = round_up(Nk, AttnBwdSm120.BC)

    qp = pad_rows(q.view(torch.uint8), nq_pad)
    kp = pad_rows(k.view(torch.uint8), nk_pad)
    vp = pad_rows(v.view(torch.uint8), nk_pad)
    lp = pad_rows(lse.float(), nq_pad, value=math.inf)
    dout_c, out_c, pmax_c = dout.to(torch.bfloat16).contiguous(), out.to(torch.bfloat16).contiguous(), pmax.float().contiguous()
    do8 = (torch.zeros if nq_pad != Nq else torch.empty)(B, Hq, nq_pad, d, dtype=torch.uint8, device=dev)
    dvec = torch.zeros(B, Hq, nq_pad, dtype=torch.float32, device=dev)
    # per (batch, kv head) stats `[dO amax, unused, max |V|, unused]`, then per-query-tile dS bounds
    n_stats = B * Hkv * 4
    stats_buf = torch.zeros(n_stats + B * Hq * (nq_pad // block_m), dtype=torch.float32, device=dev)
    stats, tb = stats_buf[:n_stats].view(B, Hkv, 4), stats_buf[n_stats:]
    stats_init = torch.zeros_like(stats_buf)
    stats_init[:n_stats].view(B, Hkv, 4)[:, :, 2] = (v.float() * sv[:, :, None, None]).norm(dim=-1).amax(dim=2)
    dk_out = torch.empty(B, Hkv, nk_pad, d, dtype=torch.bfloat16, device=dev)
    dv_out = torch.empty(B, Hkv, nk_pad, d, dtype=torch.bfloat16, device=dev)

    run_main_dq = not deterministic
    dq_acc = torch.empty(B, Hq, nq_pad, d, dtype=torch.float32, device=dev) if run_main_dq else None
    prep_args = [flat(dout_c), flat(out_c), flat(pmax_c), flat(do8), flat(dvec), flat(tb), from_dlpack(stats),
                 flat(dq_acc if dq_acc is not None else dvec),
                 Int32(B * Hq * Nq), Int32(Nq), Int32(nq_pad), Int32(Hq), Int32(Hkv)]
    prep = compile_cached(BwdPrep, (d, dq_acc is not None, block_m), (str(dout_c.dtype), B, Hq, Hkv, Nq, nq_pad), *prep_args)

    if deterministic:
        dq_out = torch.empty(B, Hq, nq_pad, d, dtype=torch.bfloat16, device=dev)
        dqk_args = [flat(qp), flat(kp), flat(vp), flat(do8), flat(lp), flat(dvec), flat(tb),
                    from_dlpack(stats), from_dlpack(desc), flat(dq_out), Float32(softmax_scale),
                    Int32(Nq), Int32(Nk), Int32(nq_pad), Int32(nk_pad), Int32(Hq), Int32(Hkv), Int32(B)]
        dqk = compile_cached(AttnBwdDqSm120, (d, causal, block_m), (B, Hq, Hkv, nq_pad, nk_pad), *dqk_args)
    else:
        dq_out = torch.empty(B, Hq, Nq, d, dtype=torch.bfloat16, device=dev)
        post_args = [flat(dq_acc), flat(tb), from_dlpack(stats), from_dlpack(desc), flat(dq_out), Float32(softmax_scale),
                     Int32(Nq), Int32(nq_pad), Int32(Hq), Int32(Hkv), Int32(B)]
        post = compile_cached(BwdDqPost, (d, block_m, dq_bulk, 4 if dq_wg else num_warps), (B, Hq, Hkv, Nq, nq_pad),
                         *post_args)

    main_args = [flat(qp), flat(kp), flat(vp), flat(do8), flat(lp), flat(dvec), flat(tb), from_dlpack(stats),
                 from_dlpack(desc), flat(dk_out), flat(dv_out),
                 flat(dq_acc if dq_acc is not None else dvec), Float32(softmax_scale),
                 Int32(Nq), Int32(Nk), Int32(nq_pad), Int32(nk_pad), Int32(Hq), Int32(Hkv), Int32(B)]
    main = compile_cached(AttnBwdSm120, (d, causal, block_m, num_stages, not run_main_dq, dq_bulk, num_warps, min_blocks,
                                       kb_major, dq_wg),
                     (B, Hq, Hkv, nq_pad, nk_pad), *main_args)

    def launch():
        # prep zeroes all `dQaccum` rows; padded rows of a partial tile hold pieces of valid rows
        stats_buf.copy_(stats_init)
        prep(*prep_args)
        main(*main_args)
        if deterministic:
            dqk(*dqk_args)

    def launch_main_only():
        main(*main_args)
        if deterministic:
            dqk(*dqk_args)

    def finish():
        if deterministic:
            return dq_out[:, :, :Nq], dk_out[:, :, :Nk], dv_out[:, :, :Nk]
        post(*post_args)
        return dq_out, dk_out[:, :, :Nk], dv_out[:, :, :Nk]

    launch.main_only = launch_main_only
    return launch, finish


def attn_bwd(q, k, v, out, lse, pmax, dout, q_descale=None, k_descale=None, v_descale=None,
                 causal=False, softmax_scale=None, num_stages=None, dq_bulk=None, deterministic=False, num_warps=8,
                 block_m=None, min_blocks=None, kb_major=None, dq_wg=None):
    """Run the attention backward once.

    Args:
        q: As in ``make_attn_bwd``.
        k: As in ``make_attn_bwd``.
        v: As in ``make_attn_bwd``.
        out: As in ``make_attn_bwd``.
        lse: As in ``make_attn_bwd``.
        pmax: As in ``make_attn_bwd``.
        dout: As in ``make_attn_bwd``.
        q_descale: As in ``make_attn_bwd``.
        k_descale: As in ``make_attn_bwd``.
        v_descale: As in ``make_attn_bwd``.
        causal: As in ``make_attn_bwd``.
        softmax_scale: As in ``make_attn_bwd``.
        num_stages: As in ``make_attn_bwd``.
        dq_bulk: As in ``make_attn_bwd``.
        deterministic: As in ``make_attn_bwd``.
        num_warps: As in ``make_attn_bwd``.
        block_m: As in ``make_attn_bwd``.
        min_blocks: As in ``make_attn_bwd``.
        kb_major: As in ``make_attn_bwd``.
        dq_wg: As in ``make_attn_bwd``.

    Returns:
        ``(dq, dk, dv)`` in bf16.
    """
    launch, finish = make_attn_bwd(q, k, v, out, lse, pmax, dout, q_descale, k_descale, v_descale,
                                       causal, softmax_scale, num_stages, dq_bulk=dq_bulk,
                                       deterministic=deterministic, num_warps=num_warps, block_m=block_m,
                                       min_blocks=min_blocks, kb_major=kb_major, dq_wg=dq_wg)
    launch()
    return finish()


BWD_CANDIDATES = {
    128: [dict(block_m=32, num_stages=st, dq_bulk=bulk) for st in (2, 3) for bulk in (False, True)]
         + [dict(block_m=32, num_stages=st, dq_bulk=bulk, dq_wg=True) for st in (2, 3) for bulk in (False, True)],
    64: [dict(block_m=bm, num_stages=st, dq_bulk=bulk) for bm in (32, 64) for st in (2, 3) for bulk in (False, True)]
        + [dict(block_m=32, num_stages=2, dq_bulk=bulk, min_blocks=2) for bulk in (False, True)]
        + [dict(block_m=bm, num_stages=st, dq_bulk=bulk, dq_wg=True) for bm in (32, 64) for st in (2, 3)
           for bulk in (False, True)],
}


def autotune_attn_bwd(q, k, v, out, lse, pmax, dout, q_descale=None, k_descale=None, v_descale=None,
                          causal=False, softmax_scale=None, candidates=None, iters=10, verbose=True):
    """Time candidate backward configs and cache the fastest.

    Each candidate is timed as a full step: prep, kernels and the dQ postprocess.

    Args:
        q: As in ``make_attn_bwd``.
        k: As in ``make_attn_bwd``.
        v: As in ``make_attn_bwd``.
        out: As in ``make_attn_bwd``.
        lse: As in ``make_attn_bwd``.
        pmax: As in ``make_attn_bwd``.
        dout: As in ``make_attn_bwd``.
        q_descale: As in ``make_attn_bwd``.
        k_descale: As in ``make_attn_bwd``.
        v_descale: As in ``make_attn_bwd``.
        causal: As in ``make_attn_bwd``.
        softmax_scale: As in ``make_attn_bwd``.
        candidates: Config dicts with keys ``block_m``, ``num_stages`` and ``dq_bulk``. Defaults to
            ``BWD_CANDIDATES`` for the head dim.
        iters: Timed steps per candidate.
        verbose: Whether to print one line per candidate.

    Returns:
        ``(best_config, results)``, where ``results`` is a list of ``(config, milliseconds or error string)``.

    Raises:
        RuntimeError: If no candidate runs.
    """
    d = q.shape[-1]

    def build(cfg):
        launch, finish = make_attn_bwd(q, k, v, out, lse, pmax, dout, q_descale, k_descale, v_descale,
                                           causal, softmax_scale, **cfg)
        return lambda: (launch(), finish())

    return autotune("bwd", d, causal, q.shape[2], k.shape[2], q.shape[1] // k.shape[1],
                    candidates or BWD_CANDIDATES[d], build, iters, verbose)
    