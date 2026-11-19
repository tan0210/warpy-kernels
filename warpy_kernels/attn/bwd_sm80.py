# type: ignore

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, Int64, const_expr, cute

from ..arch import smem_limit
from ..ptx import (
    cp_async16,
    ld_shared_v2_f32,
    ldsm_x4_trans,
    mma_16816,
    pack_half2,
    red_add_f32,
    st_global_b32,
    st_shared_b32,
)
from ..utils.fragments import ldsm_a, ldsm_b
from ..utils.layout import PaddedTile, SmemTile
from .bwd_common import bwd_scales
from .constants import P_OFFSET


class AttnBwdSm80:
    """bf16 attention backward for SM80 and newer, one 128-key block per CTA and 16 keys per warp.

    K and V stay in shared memory while Q, dO, L and D tiles stream through a ``cp.async`` ring. Each key block adds
    its dQ to an fp32 accumulator with atomics, so dQ is not bitwise reproducible.
    """

    BC = 128  # key padding granularity, 16 keys per warp
    NUM_THREADS = 256
    WARPS = 8
    FP8 = False  # whether the class has an e4m3 path

    def __init__(
        self,
        head_dim: int,
        causal: bool,
        block_m: int,
        num_stages: int = 2,
        skip_dq: bool = False,
        kb_major: bool = False,
        arch: int = 80,
        dq_interleave: bool = False,
        bf16: bool = True,
        one_barrier: bool = False,
    ):
        assert bf16 or self.FP8, f"{type(self).__name__} needs bf16 inputs"
        assert head_dim in (64, 128) and block_m in ((16, 32, 64) if bf16 else (32, 64))
        assert num_stages >= 2, "the Q/dO pipeline needs at least 2 stages"
        assert not one_barrier or (num_stages >= 3 and not skip_dq), (
            "one barrier per tile needs 3 stages and the dQ pass"
        )
        blocks = (block_m // 16) * (head_dim // 16)
        assert blocks % self.WARPS == 0, (
            "dQ needs a whole number of m16 x 16-column blocks per warp"
        )
        self.arch = arch
        self.nb = blocks // self.WARPS
        self.d = head_dim
        self.causal = causal
        self.BM = block_m
        self.nt = block_m // 8
        self.m_steps = block_m // 16  # dV/dK k steps of 16 queries per tile
        self.stages = num_stages
        self.skip_dq = skip_dq
        self.kb_major = kb_major
        self.dq_interleave = dq_interleave
        # one barrier per tile, with dS double-buffered so the next tile can write it early
        self.one_barrier = one_barrier
        self.bf16 = bf16
        # bf16 P needs no offset against underflow
        self.p_off = 0.0 if bf16 else P_OFFSET
        self.row_bytes = head_dim * (2 if bf16 else 1)
        self.v_regs = (
            self.row_bytes <= 128
        )  # d128 V fragments would take 32 more registers
        self.kv_bytes = self.BC * self.row_bytes
        self.q_tile_bytes = block_m * self.row_bytes
        # `L`, `D` and for e4m3 the per-query dQ scale, fp32
        self.ld_stage_bytes = (2 if bf16 else 3) * block_m * 4
        self.k_off = 0
        self.v_off = self.kv_bytes
        self.q_off = 2 * self.kv_bytes
        self.do_off = self.q_off + num_stages * self.q_tile_bytes
        # e4m3 dO's residual tiles
        self.dolo_off = self.do_off + num_stages * self.q_tile_bytes
        self.ld_off = self.dolo_off + (0 if bf16 else num_stages * self.q_tile_bytes)
        self.ds_off = self.ld_off + num_stages * self.ld_stage_bytes
        # bf16 dS^T as `[key][query]`, e4m3 dS as `[query][key]`
        self.ds_shape = (block_m * 2, self.BC) if bf16 else (self.BC, block_m)
        self.ds_bytes = self.ds_shape[1] * (self.ds_shape[0] + 16)
        self.smem_bytes = self.ds_off + self.ds_bytes * (2 if one_barrier else 1)
        assert self.smem_bytes <= smem_limit(arch), (
            f"shared memory does not fit: {self.smem_bytes} B > {smem_limit(arch)} B "
            f"(d={head_dim}, block_m={block_m}, num_stages={num_stages})"
        )
        nbytes = self.smem_bytes

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        self.shared_storage = SharedStorage

    def _g_row(self, row0):
        return row0  # where the dQ scales of rows `row0 ..` start

    def _tile(self, sbase, off, rows):
        return SmemTile(sbase + off, self.row_bytes, rows)

    def _q_tile(self, sbase, st):
        return self._tile(sbase, self.q_off + st * self.q_tile_bytes, self.BM)

    def _do_tile(self, sbase, st):
        return self._tile(sbase, self.do_off + st * self.q_tile_bytes, self.BM)

    def _load_rows(self, dst, g_addr, tidx):
        total = dst.rows * dst.chunks
        for i in range(max(1, total // self.NUM_THREADS)):
            idx = tidx + i * self.NUM_THREADS
            if total < self.NUM_THREADS:
                idx = idx & (total - 1)  # extra threads repeat a copy
            row = idx // dst.chunks
            ch = idx % dst.chunks
            cp_async16(
                dst.addr(row, ch), g_addr + Int64(row * self.row_bytes + ch * 16)
            )

    @cute.jit
    def _q_row0(self, u: Int32, tile):
        n_q, q_start, b, hkv, h_q, h_kv, nq_pad, total_it, rot = tile
        # step `u` takes tile `u - kblock`, so a query tile reaches key block `kb - 1` one step before `kb`
        it = (u + total_it - rot) % total_it
        rep = h_q // h_kv
        hq = hkv * rep + it // n_q
        qi = q_start + it % n_q
        return (b * h_q + hq) * nq_pad + qi * self.BM, hq, qi

    @cute.jit
    def _load_q(
        self,
        it: Int32,
        total_it: Int32,
        tile,
        sbase: Int32,
        gq: Int64,
        gdo: Int64,
        gl: Int64,
        gd: Int64,
        tidx: Int32,
    ):
        if it < total_it:
            row0, _, _ = self._q_row0(it, tile)
            st = it % self.stages
            self._load_rows(
                self._q_tile(sbase, st), gq + Int64(row0) * self.row_bytes, tidx
            )
            self._load_rows(
                self._do_tile(sbase, st), gdo + Int64(row0) * self.row_bytes, tidx
            )
            if const_expr(not self.bf16):
                self._load_rows(
                    self._tile(sbase, self.dolo_off + st * self.q_tile_bytes, self.BM),
                    self._gdolo + Int64(row0) * self.row_bytes,
                    tidx,
                )
            # 16-byte pieces of one fp32 row of `L`, `D` or the dQ scale
            pieces = self.BM // 4
            if tidx < (2 if self.bf16 else 3) * pieces:
                which = tidx // pieces
                r0 = Int64(row0) if which < 2 else Int64(self._g_row(row0))
                gsrc = (gl if which == 0 else (gd if which == 1 else self._gg)) + (
                    r0 + Int64(4 * (tidx % pieces))
                ) * 4
                cp_async16(
                    sbase
                    + self.ld_off
                    + st * self.ld_stage_bytes
                    + which * (self.BM * 4)
                    + (tidx % pieces) * 16,
                    gsrc,
                )

    @cute.jit
    def _mma_sdp(self, sbase, st, warp, lane, vf, s_acc, dp_acc):
        ktile = self._tile(sbase, self.k_off, self.BC)
        vtile = self._tile(sbase, self.v_off, self.BC)
        qtile = self._q_tile(sbase, st)
        otile = self._do_tile(sbase, st)
        s_acc.fill(0.0)
        dp_acc.fill(0.0)
        # `S^T = K_w Q^T`, `dP^T = V_w dO^T`, rows are this warp's 16 keys
        for kk in cutlass.range_constexpr(self.row_bytes // 32):
            a_k = ldsm_a(ktile, warp * 16, kk, lane)
            if const_expr(self.v_regs):
                a_v = [vf[4 * kk + x] for x in range(4)]
            else:
                a_v = ldsm_a(vtile, warp * 16, kk, lane)
            for npair in cutlass.range_constexpr(self.nt // 2):
                b_q = ldsm_b(qtile, 16 * npair, kk, lane)
                b_o = ldsm_b(otile, 16 * npair, kk, lane)
                for h in cutlass.range_constexpr(2):
                    o = 4 * (2 * npair + h)
                    c = mma_16816(
                        a_k,
                        [b_q[2 * h], b_q[2 * h + 1]],
                        [s_acc[o + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        s_acc[o + x] = c[x]
                    c = mma_16816(
                        a_v,
                        [b_o[2 * h], b_o[2 * h + 1]],
                        [dp_acc[o + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dp_acc[o + x] = c[x]

    @cute.jit
    def _softmax_ds(
        self,
        sbase,
        st,
        qi,
        kblock,
        seqlen_q,
        seqlen_k,
        warp,
        lane,
        beta,
        kc,
        s_acc,
        dp_acc,
    ):
        g = lane // 4
        t = lane % 4
        key0 = kblock * self.BC + warp * 16 + g
        diag = seqlen_k - seqlen_q
        ldb = sbase + self.ld_off + st * self.ld_stage_bytes
        # `P = 2^(s*beta - L + p_off)`, `dS = P * (dP*kc - D*2^-p_off)`
        for nt in cutlass.range_constexpr(self.nt):
            l2 = ld_shared_v2_f32(ldb + (8 * nt + 2 * t) * 4)
            d2 = ld_shared_v2_f32(ldb + self.BM * 4 + (8 * nt + 2 * t) * 4)
            for e in cutlass.range_constexpr(2):
                q_loc = 8 * nt + 2 * t + e
                lq = l2[e]
                dq = d2[e]
                if const_expr(self.p_off != 0.0):
                    lq = lq - Float32(self.p_off)
                    dq = dq * Float32(2.0**-self.p_off)
                for r in cutlass.range_constexpr(2):
                    idx = 4 * nt + 2 * r + e
                    p = cute.math.exp2(s_acc[idx] * beta - lq, fastmath=True)
                    p = p if key0 + 8 * r < seqlen_k else Float32(0.0)
                    if const_expr(self.causal):
                        p = (
                            p
                            if key0 + 8 * r <= qi * self.BM + q_loc + diag
                            else Float32(0.0)
                        )
                    s_acc[idx] = p
                    dp_acc[idx] = p * (dp_acc[idx] * kc - dq)

    @cute.jit
    def _store_ds(self, sbase, st, dst, warp, lane, dp_acc):
        g = lane // 4
        t = lane % 4
        # dS^T as `[key][query]`, two adjacent queries per 4-byte store
        for nt in cutlass.range_constexpr(self.nt):
            for r in cutlass.range_constexpr(2):
                st_shared_b32(
                    dst.addr(warp * 16 + g + 8 * r, (8 * nt + 2 * t) * 2),
                    pack_half2(
                        dp_acc[4 * nt + 2 * r], dp_acc[4 * nt + 2 * r + 1], "bf16"
                    ),
                )

    @cute.jit
    def _mma_dvdk(self, sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc, reds=None):
        qtile = self._q_tile(sbase, st)
        otile = self._do_tile(sbase, st)
        for kk in cutlass.range_constexpr(self.m_steps):
            nt = 2 * kk
            # P^T and dS^T of 16 queries as A operands, keys g and g + 8 then queries + 8
            pa = [
                pack_half2(s_acc[4 * nt + 2 * x], s_acc[4 * nt + 2 * x + 1], "bf16")
                for x in range(4)
            ]
            da = [
                pack_half2(dp_acc[4 * nt + 2 * x], dp_acc[4 * nt + 2 * x + 1], "bf16")
                for x in range(4)
            ]
            qrow = 16 * kk + (lane % 8) + ((lane // 8) % 2) * 8
            # `dV += P^T dO`, `dK += dS^T Q`, B via transposing `ldmatrix`
            for npair in cutlass.range_constexpr(self.d // 16):
                ch = 2 * npair + lane // 16
                bo = ldsm_x4_trans(otile.addr(qrow, ch))
                bq = ldsm_x4_trans(qtile.addr(qrow, ch))
                for h in cutlass.range_constexpr(2):
                    o = 4 * (2 * npair + h)
                    c = mma_16816(
                        pa,
                        [bo[2 * h], bo[2 * h + 1]],
                        [dv_acc[o + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dv_acc[o + x] = c[x]
                    c = mma_16816(
                        da,
                        [bq[2 * h], bq[2 * h + 1]],
                        [dk_acc[o + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dk_acc[o + x] = c[x]
                if const_expr(reds is not None):
                    # spread the dQ atomics over the dV/dK steps
                    steps = self.m_steps * (self.d // 16)
                    per = (len(reds) + steps - 1) // steps
                    i = kk * (self.d // 16) + npair
                    self._red_dq(reds[i * per : (i + 1) * per])

    @cute.jit
    def _dq(self, sbase, st, ds, row0, gdq, warp, lane):
        out = []
        d = self.d
        g = lane // 4
        t = lane % 4
        ktile = self._tile(sbase, self.k_off, self.BC)
        dchunks = d // 16
        j = lane // 8
        for i in cutlass.range_constexpr(self.nb):
            blk = warp + self.WARPS * i
            q0 = (blk // dchunks) * 16  # this block's 16 queries and 16 columns
            c16 = blk % dchunks
            acc = [Float32(0.0)] * 8
            for ks in cutlass.range_constexpr(self.BC // 16):
                # dS of 16 queries x 16 keys as the A operand, read transposed from `[key][query]`
                a = ldsm_x4_trans(
                    ds.addr(16 * ks + (j // 2) * 8 + lane % 8, (q0 + (j % 2) * 8) * 2)
                )
                key = 16 * ks + (lane % 8) + ((lane // 8) % 2) * 8
                bk = ldsm_x4_trans(ktile.addr(key, 2 * c16 + lane // 16))
                for h in cutlass.range_constexpr(2):
                    c = mma_16816(
                        a, [bk[2 * h], bk[2 * h + 1]], acc[4 * h : 4 * h + 4], "bf16"
                    )
                    for x in cutlass.range_constexpr(4):
                        acc[4 * h + x] = c[x]
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    q_loc = q0 + g + 8 * r
                    col = c16 * 16 + 8 * h + 2 * t
                    addr = gdq + (Int64(row0 + q_loc) * d + Int64(col)) * 4
                    out.append((addr, acc[4 * h + 2 * r], acc[4 * h + 2 * r + 1]))
        return out

    def _red_dq(self, parts):
        for addr, v0, v1 in parts:
            red_add_f32(addr, v0)
            red_add_f32(addr + 4, v1)

    @cute.jit
    def _store_dkdv(
        self, warp, lane, kblock, seqlen_k, s_dk, s_dv, k_row0, dk_acc, dv_acc, mdK, mdV
    ):
        d = self.d
        g = lane // 4
        t = lane % 4
        gdk = Int64(mdK.iterator.toint())
        gdv = Int64(mdV.iterator.toint())
        for r in cutlass.range_constexpr(2):
            if kblock * self.BC + warp * 16 + g + 8 * r < seqlen_k:
                krow = Int64(k_row0 + warp * 16 + g + 8 * r)
                for nt in cutlass.range_constexpr(d // 8):
                    off = (krow * d + Int64(8 * nt + 2 * t)) * 2
                    st_global_b32(
                        gdk + off,
                        pack_half2(
                            dk_acc[4 * nt + 2 * r] * s_dk,
                            dk_acc[4 * nt + 2 * r + 1] * s_dk,
                            "bf16",
                        ),
                    )
                    st_global_b32(
                        gdv + off,
                        pack_half2(
                            dv_acc[4 * nt + 2 * r] * s_dv,
                            dv_acc[4 * nt + 2 * r + 1] * s_dv,
                            "bf16",
                        ),
                    )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdOlo: cute.Tensor,
        mL: cute.Tensor,
        mD: cute.Tensor,
        mG: cute.Tensor,
        mStats: cute.Tensor,
        mDesc: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQacc: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        nq_pad: Int32,
        nk_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
        batch: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(self.kb_major):
            grid = [
                h_kv * batch,
                nk_pad // self.BC,
                1,
            ]  # longest causal CTAs launch first
        else:
            grid = [nk_pad // self.BC, h_kv, batch]
        self.kernel(
            mQ,
            mK,
            mV,
            mdO,
            mdOlo,
            mL,
            mD,
            mG,
            mStats,
            mDesc,
            mdK,
            mdV,
            mdQacc,
            softmax_scale,
            seqlen_q,
            seqlen_k,
            nq_pad,
            nk_pad,
            h_q,
            h_kv,
        ).launch(
            grid=grid,
            block=[self.NUM_THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdOlo: cute.Tensor,
        mL: cute.Tensor,
        mD: cute.Tensor,
        mG: cute.Tensor,
        mStats: cute.Tensor,
        mDesc: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdQacc: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        nq_pad: Int32,
        nk_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
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

        beta, kc, s_dk, s_dv, _ = bwd_scales(
            mStats, mDesc, b, hkv, softmax_scale, self.p_off
        )

        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sbase = Int32(
            storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint()
        )
        ds = PaddedTile(sbase + self.ds_off, *self.ds_shape)

        k_row0 = (b * h_kv + hkv) * nk_pad + kblock * self.BC
        gq = Int64(mQ.iterator.toint())
        gdo = Int64(mdO.iterator.toint())
        gdq = Int64(mdQacc.iterator.toint())
        gl = Int64(mL.iterator.toint())
        gd = Int64(mD.iterator.toint())
        self._gdolo = Int64(mdOlo.iterator.toint())
        self._gg = Int64(mG.iterator.toint())

        # causal keeps key j for query i iff j <= i + Nk - Nq
        nq_tiles = nq_pad // self.BM
        q_start = Int32(0)
        if const_expr(self.causal):
            first_q = kblock * self.BC + seqlen_q - seqlen_k
            qs = first_q // self.BM if first_q > 0 else Int32(0)
            q_start = min(nq_tiles, qs)
        n_q = nq_tiles - q_start
        total_it = (h_q // h_kv) * n_q
        rot = kblock % max(total_it, Int32(1))
        tile = (n_q, q_start, b, hkv, h_q, h_kv, nq_pad, total_it, rot)

        self._load_rows(
            self._tile(sbase, self.k_off, self.BC),
            Int64(mK.iterator.toint()) + Int64(k_row0) * self.row_bytes,
            tidx,
        )
        self._load_rows(
            self._tile(sbase, self.v_off, self.BC),
            Int64(mV.iterator.toint()) + Int64(k_row0) * self.row_bytes,
            tidx,
        )
        cute.arch.cp_async_commit_group()
        for s in cutlass.range_constexpr(self.stages - 1):
            self._load_q(Int32(s), total_it, tile, sbase, gq, gdo, gl, gd, tidx)
            cute.arch.cp_async_commit_group()

        s_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dp_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dk_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        dv_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        dk_acc.fill(0.0)
        dv_acc.fill(0.0)
        vf = None
        if const_expr(self.v_regs):
            # V's A fragments are the same for every tile
            cute.arch.cp_async_wait_group(self.stages - 1)  # K and V resident
            cute.arch.sync_threads()
            vf = cute.make_rmem_tensor(4 * (self.row_bytes // 32), Int32)
            for kk in cutlass.range_constexpr(self.row_bytes // 32):
                a_v = ldsm_a(
                    self._tile(sbase, self.v_off, self.BC), warp * 16, kk, lane
                )
                for x in cutlass.range_constexpr(4):
                    vf[4 * kk + x] = a_v[x]

        if const_expr(self.one_barrier):
            cute.arch.cp_async_wait_group(self.stages - 2)  # K, V and tile 0 resident
            cute.arch.sync_threads()
            for it in cutlass.range(total_it, unroll=1):
                st = it % self.stages
                row0, _, qi = self._q_row0(it, tile)
                dsb = PaddedTile(
                    sbase + self.ds_off + (it % 2) * self.ds_bytes, *self.ds_shape
                )
                self._mma_sdp(sbase, st, warp, lane, vf, s_acc, dp_acc)
                self._softmax_ds(
                    sbase,
                    st,
                    qi,
                    kblock,
                    seqlen_q,
                    seqlen_k,
                    warp,
                    lane,
                    beta,
                    kc,
                    s_acc,
                    dp_acc,
                )
                self._store_ds(sbase, st, dsb, warp, lane, dp_acc)
                if const_expr(not self.dq_interleave):
                    self._mma_dvdk(sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc)
                cute.arch.cp_async_wait_group(self.stages - 3)  # tile `it + 1` resident
                # dS is written, and every warp is done with tile `it - 1`, whose slot the next load reuses
                cute.arch.sync_threads()
                self._load_q(
                    it + (self.stages - 1), total_it, tile, sbase, gq, gdo, gl, gd, tidx
                )
                cute.arch.cp_async_commit_group()
                if const_expr(self.dq_interleave):
                    parts = self._dq(sbase, st, dsb, row0, gdq, warp, lane)
                    self._mma_dvdk(
                        sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc, parts
                    )
                else:
                    self._red_dq(self._dq(sbase, st, dsb, row0, gdq, warp, lane))

        else:
            for it in cutlass.range(total_it, unroll=1):
                cute.arch.cp_async_wait_group(self.stages - 2)  # tile `it` resident
                cute.arch.sync_threads()  # and every warp is done with tile `it - 1` and its dS
                self._load_q(
                    it + (self.stages - 1), total_it, tile, sbase, gq, gdo, gl, gd, tidx
                )
                cute.arch.cp_async_commit_group()
                st = it % self.stages
                row0, _, qi = self._q_row0(it, tile)
                self._mma_sdp(sbase, st, warp, lane, vf, s_acc, dp_acc)
                self._softmax_ds(
                    sbase,
                    st,
                    qi,
                    kblock,
                    seqlen_q,
                    seqlen_k,
                    warp,
                    lane,
                    beta,
                    kc,
                    s_acc,
                    dp_acc,
                )
                if const_expr(not self.skip_dq):
                    self._store_ds(sbase, st, ds, warp, lane, dp_acc)
                if const_expr(self.skip_dq):
                    self._mma_dvdk(sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc)
                elif const_expr(self.dq_interleave):
                    cute.arch.sync_threads()  # all of dS is written
                    parts = self._dq(sbase, st, ds, row0, gdq, warp, lane)
                    self._mma_dvdk(
                        sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc, parts
                    )
                else:
                    self._mma_dvdk(sbase, st, lane, s_acc, dp_acc, dv_acc, dk_acc)
                    cute.arch.sync_threads()  # all of dS is written
                    self._red_dq(self._dq(sbase, st, ds, row0, gdq, warp, lane))

        self._store_dkdv(
            warp, lane, kblock, seqlen_k, s_dk, s_dv, k_row0, dk_acc, dv_acc, mdK, mdV
        )
