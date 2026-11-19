# type: ignore


import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, Int64, const_expr, cute

from ..arch import smem_limit
from ..ptx import (
    bulk_reduce_add_f32,
    bulk_reduce_add_u64,
    cp_async16,
    ds_scale,
    ld_shared_f32,
    ld_shared_v2_f32,
    ldsm_x2_trans_b8,
    ldsm_x4,
    ldsm_x4_trans,
    mma_16816,
    mma_mxf8,
    pack_e4m3x4,
    pack_fixed2,
    pack_half2,
    red_add_u64,
    red_add_v2_f32,
    st_global_b32,
    st_shared_b64,
    st_shared_e4m3_pair,
    st_shared_v2_f32,
    stsm_x4_trans,
    swizzle_offset,
    ue8m0_inv,
)
from .bwd_common import bwd_scales, g_index
from .constants import P_OFFSET


class AttnBwdSm120:
    BC = 128  # key padding granularity, 16 keys per warp
    NUM_THREADS = 256  # 8 MMA warps
    DQ_WARPS = 4  # the dQ warpgroup, which also loads the query tiles
    # 8 x 216 + 4 x 72 = 12 x 168, the register budget of 384 threads
    MMA_REGS = 216
    DQ_REGS = 72
    V_REGS = True  # keep V's A fragments in registers

    def __init__(
        self,
        head_dim: int,
        causal: bool,
        block_m: int,
        num_stages: int = 3,
        skip_dq: bool = False,
        dq_bulk: bool = True,
        kb_major: bool = False,
        bf16: bool = False,
        fixed_dq: bool = False,
    ):
        assert head_dim in (64, 128) and block_m in ((16, 32, 64) if bf16 else (32, 64))
        assert not (fixed_dq and bf16), "fixed-point dQ needs the e4m3 dQ scales"
        # dQ adds as int32 pairs, which makes it bitwise reproducible
        self.fixed_dq = fixed_dq
        assert num_stages >= 2, "the Q/dO pipeline needs at least 2 stages"
        self.bf16 = bf16
        # bf16 P needs no offset against underflow
        self.p_off = 0.0 if bf16 else P_OFFSET
        self.row_bytes = head_dim * (2 if bf16 else 1)
        # dV/dK k steps per query tile, 16 bf16 or 32 e4m3 queries each
        self.m_steps = block_m // (16 if bf16 else 32)
        # d128 bf16 V fragments would take 32 more registers
        self.v_regs = self.V_REGS and self.row_bytes <= 128
        self.kb_major = kb_major
        self.skip_dq = skip_dq
        self.dq_bulk = dq_bulk and not skip_dq
        blocks = (block_m // 16) * (head_dim // 16)
        assert blocks % self.DQ_WARPS == 0, (
            "dQ split needs a whole number of (m16 x 16-col) blocks per warp"
        )
        self.nb = blocks // self.DQ_WARPS
        self.d = head_dim
        self.causal = causal
        self.BM = block_m
        self.nt = block_m // 8
        self.stages = num_stages
        self.kv_bytes = self.BC * self.row_bytes
        self.q_tile_bytes = block_m * self.row_bytes
        self.ds_row = self.BC * (2 if bf16 else 1)
        self.ds_bytes = block_m * self.ds_row
        self.k_off = 0
        self.v_off = self.kv_bytes
        self.q_off = 2 * self.kv_bytes
        self.do_off = self.q_off + num_stages * self.q_tile_bytes
        # bf16 dO has no residual tile
        self.dolo_off = self.do_off + num_stages * self.q_tile_bytes
        self.ds_off = self.dolo_off + (0 if bf16 else num_stages * self.q_tile_bytes)
        self.dq_stage_bytes = block_m * head_dim * 4
        # `L`, `D`, the per-query dQ scale and with fixed-point dQ its scale `cq` (fp32) for one query tile
        self.ld_stage_bytes = (4 if fixed_dq else 3) * block_m * 4
        self.ld_off = self.ds_off + 2 * self.ds_bytes
        self.dq_off = self.ld_off + num_stages * self.ld_stage_bytes
        self.smem_bytes = self.dq_off + (self.dq_stage_bytes if self.dq_bulk else 0)
        assert self.smem_bytes + 1024 <= smem_limit(120), (
            f"shared memory does not fit: {self.smem_bytes} B"
        )
        nbytes = self.smem_bytes
        nbar = 4 + 2 * num_stages

        @cute.struct
        class SharedStorage:
            # dS full and free (two buffers each), then stage full and empty
            mbar: cute.struct.MemRange[cutlass.Int64, nbar]
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        self.shared_storage = SharedStorage

    def _bulk_add(self, gaddr, saddr, nbytes):
        if const_expr(self.fixed_dq):
            bulk_reduce_add_u64(gaddr, saddr, nbytes)
        else:
            bulk_reduce_add_f32(gaddr, saddr, nbytes)

    def _swz_off(self, row, chunk):
        if self.bf16:
            return swizzle_offset(self.row_bytes, row, chunk)
        if self.d == 128:
            # rows r and r + 8 get different chunks too, since the dV/dK loads read rows 0, 1, 8, 9, ... together
            return row * 128 + ((((row & 7) ^ (((row >> 3) & 1) << 2)) ^ chunk) << 4)
        return row * 64 + (((((row >> 1) & 3) ^ (((row >> 3) & 1) << 1)) ^ chunk) << 4)

    def _ds_off(self, row, chunk):
        # one dS row holds `BC` keys
        return swizzle_offset(self.ds_row, row, chunk)

    def _mma(self, a, b, c):
        if self.bf16:
            return mma_16816(a, b, c, "bf16")
        return mma_mxf8(a, b, c)

    def _load_rows(self, s_addr, g_addr, nrows, tidx, nthreads=None):
        nthreads = nthreads or self.NUM_THREADS
        chunks = self.row_bytes // 16
        total = nrows * chunks
        for i in range(max(1, total // nthreads)):
            idx = tidx + i * nthreads
            if total < nthreads:
                idx = idx & (total - 1)  # extra threads repeat a copy
            row = idx // chunks
            ch = idx % chunks
            cp_async16(
                s_addr + self._swz_off(row, ch),
                g_addr + Int64(row * self.row_bytes + ch * 16),
            )

    @cute.jit
    def _q_row0(
        self,
        it: Int32,
        n_q: Int32,
        q_start: Int32,
        b: Int32,
        hkv: Int32,
        h_q: Int32,
        h_kv: Int32,
        nq_pad: Int32,
    ):
        rep = h_q // h_kv
        hq = hkv * rep + it // n_q
        qi = q_start + it % n_q
        return (b * h_q + hq) * nq_pad + qi * self.BM, qi

    @cute.jit
    def _load_q(
        self,
        it: Int32,
        total_it: Int32,
        n_q: Int32,
        q_start: Int32,
        b: Int32,
        hkv: Int32,
        h_q: Int32,
        h_kv: Int32,
        nq_pad: Int32,
        sbase: Int32,
        gq: Int64,
        gdo: Int64,
        gl: Int64,
        gd: Int64,
        tidx: Int32,
        nthreads: cutlass.Constexpr = None,
    ):
        if it < total_it:
            row0, _ = self._q_row0(it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
            st = it % self.stages
            self._load_rows(
                sbase + self.q_off + st * self.q_tile_bytes,
                gq + Int64(row0) * self.row_bytes,
                self.BM,
                tidx,
                nthreads,
            )
            self._load_rows(
                sbase + self.do_off + st * self.q_tile_bytes,
                gdo + Int64(row0) * self.row_bytes,
                self.BM,
                tidx,
                nthreads,
            )
            if const_expr(not self.bf16):
                self._load_rows(
                    sbase + self.dolo_off + st * self.q_tile_bytes,
                    self._gdolo + Int64(row0) * self.d,
                    self.BM,
                    tidx,
                    nthreads,
                )
            # L, D and the e4m3 dQ scale rows in 16-byte pieces
            pieces = self.BM // 4
            if tidx < (2 if self.bf16 else (4 if self.fixed_dq else 3)) * pieces:
                which = tidx // pieces
                r0 = Int64(row0)
                if const_expr(self.fixed_dq):
                    r0 = (
                        Int64(row0)
                        if which < 2
                        else Int64(g_index(row0) + (which - 2) * 128)
                    )
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
    def _key_scales(self, dp_acc):
        m = [Float32(0.0), Float32(0.0)]
        for nt in cutlass.range_constexpr(self.nt):
            for r in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(2):
                    x = dp_acc[4 * nt + 2 * r + e]
                    m[r] = cute.arch.fmax(m[r], x if x > Float32(0.0) else -x)
        # dK scales of this thread's two keys, from the largest |dS| over the tile's queries
        return [
            ds_scale(cute.arch.warp_reduction_max(m[r], threads_in_group=4))
            for r in range(2)
        ]

    @cute.jit
    def _mma_sdp(self, sbase, qb, ob, olb, warp, lane, s_acc, dp_acc, vf=None):
        kb = sbase + self.k_off
        vb = sbase + self.v_off
        s_acc.fill(0.0)
        dp_acc.fill(0.0)
        # `S^T = K_w Q^T`, `dP^T = V_w dO^T`, rows are this warp's 16 keys
        for kk in cutlass.range_constexpr(self.row_bytes // 32):
            arow = warp * 16 + (lane % 8) + ((lane // 8) % 2) * 8
            ach = lane // 16 + 2 * kk
            a_k = ldsm_x4(kb + self._swz_off(arow, ach))
            if const_expr(vf is not None):
                a_v = [vf[4 * kk + x] for x in range(4)]
            else:
                a_v = ldsm_x4(vb + self._swz_off(arow, ach))
            for npair in cutlass.range_constexpr(self.nt // 2):
                brow = (lane % 8) + (lane // 16) * 8 + 16 * npair
                bch = (lane // 8) % 2 + 2 * kk
                b_q = ldsm_x4(qb + self._swz_off(brow, bch))
                b_o = ldsm_x4(ob + self._swz_off(brow, bch))
                if const_expr(not self.bf16):
                    b_l = ldsm_x4(olb + self._swz_off(brow, bch))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = self._mma(
                        a_k,
                        [b_q[2 * h], b_q[2 * h + 1]],
                        [
                            s_acc[4 * nt],
                            s_acc[4 * nt + 1],
                            s_acc[4 * nt + 2],
                            s_acc[4 * nt + 3],
                        ],
                    )
                    for x in cutlass.range_constexpr(4):
                        s_acc[4 * nt + x] = c[x]
                    c = self._mma(
                        a_v,
                        [b_o[2 * h], b_o[2 * h + 1]],
                        [
                            dp_acc[4 * nt],
                            dp_acc[4 * nt + 1],
                            dp_acc[4 * nt + 2],
                            dp_acc[4 * nt + 3],
                        ],
                    )
                    if const_expr(not self.bf16):
                        # ue8m0 0x7B reads the residual at 2^-4
                        c = mma_mxf8(a_v, [b_l[2 * h], b_l[2 * h + 1]], c, sfb=0x7B)
                    for x in cutlass.range_constexpr(4):
                        dp_acc[4 * nt + x] = c[x]

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
        g_t,
        s_acc,
        dp_acc,
    ):
        g = lane // 4
        t = lane % 4
        kappa = g_t * kc
        dgam = g_t * Float32(2.0**-self.p_off)
        key0 = kblock * self.BC + warp * 16 + g
        diag = seqlen_k - seqlen_q
        ldb = sbase + self.ld_off + st * self.ld_stage_bytes
        # `P8 = 2^(s*beta - L + 8)`, `dS*g = P8 * (dP*kappa - D*g*2^-8)`
        for nt in cutlass.range_constexpr(self.nt):
            l2 = ld_shared_v2_f32(ldb + (8 * nt + 2 * t) * 4)
            d2 = ld_shared_v2_f32(ldb + self.BM * 4 + (8 * nt + 2 * t) * 4)
            for e in cutlass.range_constexpr(2):
                q_loc = 8 * nt + 2 * t + e
                lq = l2[e]
                dq = d2[e] * dgam
                for r in cutlass.range_constexpr(2):
                    idx = 4 * nt + 2 * r + e
                    p8 = cute.math.exp2(
                        s_acc[idx] * beta - lq + Float32(self.p_off), fastmath=True
                    )
                    p8 = p8 if key0 + 8 * r < seqlen_k else Float32(0.0)
                    if const_expr(self.causal):
                        p8 = (
                            p8
                            if key0 + 8 * r <= qi * self.BM + q_loc + diag
                            else Float32(0.0)
                        )
                    s_acc[idx] = p8
                    dp_acc[idx] = p8 * (dp_acc[idx] * kappa - dq)

    @cute.jit
    def _store_ds(self, it, dsb, warp, lane, dp_acc, gq_base):
        # wait until the dQ warps free this buffer
        cute.arch.mbarrier_wait(self._ds_free + it % 2, ((it // 2) & 1) ^ 1)
        if const_expr(self.bf16):
            self._store_ds_bf16(dsb, warp, lane, dp_acc)
        else:
            self._store_ds_e4m3(dsb, warp, lane, dp_acc, gq_base)
        cute.arch.mbarrier_arrive(self._ds_full + it % 2)

    @cute.jit
    def _store_ds_bf16(self, dsb, warp, lane, dp_acc):
        for p in cutlass.range_constexpr(self.nt // 2):
            # matrix j is query block `2 * p + j // 2` and key half `j % 2`, stored as `[query][key]`
            row = 16 * p + 8 * (lane // 16) + lane % 8
            stsm_x4_trans(
                dsb + self._ds_off(row, 2 * warp + (lane // 8) % 2),
                *[
                    pack_half2(dp_acc[8 * p + 2 * j], dp_acc[8 * p + 2 * j + 1], "bf16")
                    for j in range(4)
                ],
            )

    @cute.jit
    def _store_ds_e4m3(self, dsb, warp, lane, dp_acc, gq_base):
        g = lane // 4
        t = lane % 4
        for nt in cutlass.range_constexpr(self.nt):
            gq = ld_shared_v2_f32(gq_base + (8 * nt + 2 * t) * 4)
            for e in cutlass.range_constexpr(2):
                q_loc = 8 * nt + 2 * t + e
                a_lo = dsb + self._ds_off(q_loc, warp) + g
                st_shared_e4m3_pair(
                    a_lo,
                    a_lo + 8,
                    dp_acc[4 * nt + e] * gq[e],
                    dp_acc[4 * nt + 2 + e] * gq[e],
                )

    @cute.jit
    def _pack(self, s_acc, dp_acc, pa_t, da_t, gk):
        if const_expr(self.bf16):
            for kk in cutlass.range_constexpr(self.m_steps):
                nt = 2 * kk
                # P^T and dS^T of 16 queries as A operands, keys g and g + 8 then queries + 8
                for src, dst in ((s_acc, pa_t), (dp_acc, da_t)):
                    for x in cutlass.range_constexpr(4):
                        dst[4 * kk + x] = pack_half2(
                            src[4 * nt + 2 * x], src[4 * nt + 2 * x + 1], "bf16"
                        )
        else:
            self._pack_e4m3(s_acc, dp_acc, pa_t, da_t, gk)

    @cute.jit
    def _pack_e4m3(self, s_acc, dp_acc, pa_t, da_t, gk):
        # entry i belongs to key row `(i // 2) % 2` of this thread
        ds = [dp_acc[i] * gk[(i // 2) % 2] for i in range(4 * self.nt)]
        for kk2 in cutlass.range_constexpr(self.BM // 32):
            t0 = 4 * kk2
            for src, dst in ((s_acc, pa_t), (ds, da_t)):
                dst[4 * kk2] = pack_e4m3x4(
                    src[4 * t0], src[4 * t0 + 1], src[4 * t0 + 4], src[4 * t0 + 5]
                )
                dst[4 * kk2 + 1] = pack_e4m3x4(
                    src[4 * t0 + 2], src[4 * t0 + 3], src[4 * t0 + 6], src[4 * t0 + 7]
                )
                dst[4 * kk2 + 2] = pack_e4m3x4(
                    src[4 * t0 + 8], src[4 * t0 + 9], src[4 * t0 + 12], src[4 * t0 + 13]
                )
                dst[4 * kk2 + 3] = pack_e4m3x4(
                    src[4 * t0 + 10],
                    src[4 * t0 + 11],
                    src[4 * t0 + 14],
                    src[4 * t0 + 15],
                )

    @cute.jit
    def _mma_dvdk(self, qb, ob, v_row, pa_t, da_t, dv_acc, dk_acc, sfa_dk=0x7F):
        if const_expr(self.bf16):
            self._mma_dvdk_bf16(qb, ob, pa_t, da_t, dv_acc, dk_acc)
        else:
            self._mma_dvdk_e4m3(qb, ob, v_row, pa_t, da_t, dv_acc, dk_acc, sfa_dk)

    @cute.jit
    def _mma_dvdk_bf16(self, qb, ob, pa_t, da_t, dv_acc, dk_acc):
        lane = cute.arch.thread_idx()[0] % 32
        for kk in cutlass.range_constexpr(self.m_steps):
            pa = [pa_t[4 * kk + x] for x in range(4)]
            da = [da_t[4 * kk + x] for x in range(4)]
            qrow = 16 * kk + (lane % 8) + ((lane // 8) % 2) * 8
            # `dV += P^T dO`, `dK += dS^T Q`, B via transposing b16 `ldmatrix`
            for npair in cutlass.range_constexpr(self.d // 16):
                ch = 2 * npair + lane // 16
                bo = ldsm_x4_trans(ob + self._swz_off(qrow, ch))
                bq = ldsm_x4_trans(qb + self._swz_off(qrow, ch))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_16816(
                        pa,
                        [bo[2 * h], bo[2 * h + 1]],
                        [dv_acc[4 * nt + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dv_acc[4 * nt + x] = c[x]
                    c = mma_16816(
                        da,
                        [bq[2 * h], bq[2 * h + 1]],
                        [dk_acc[4 * nt + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dk_acc[4 * nt + x] = c[x]

    @cute.jit
    def _mma_dvdk_e4m3(self, qb, ob, v_row, pa_t, da_t, dv_acc, dk_acc, sfa_dk):
        for kk2 in cutlass.range_constexpr(self.BM // 32):
            pa = [
                pa_t[4 * kk2],
                pa_t[4 * kk2 + 1],
                pa_t[4 * kk2 + 2],
                pa_t[4 * kk2 + 3],
            ]
            da = [
                da_t[4 * kk2],
                da_t[4 * kk2 + 1],
                da_t[4 * kk2 + 2],
                da_t[4 * kk2 + 3],
            ]
            qrow = v_row + 32 * kk2
            # `dV += P^T dO`, `dK += dS^T Q`, B via transposing 8-bit `ldmatrix`
            for npair in cutlass.range_constexpr(self.d // 16):
                bo = ldsm_x2_trans_b8(ob + self._swz_off(qrow, npair))
                bq = ldsm_x2_trans_b8(qb + self._swz_off(qrow, npair))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_mxf8(
                        pa,
                        [bo[h], bo[2 + h]],
                        [
                            dv_acc[4 * nt],
                            dv_acc[4 * nt + 1],
                            dv_acc[4 * nt + 2],
                            dv_acc[4 * nt + 3],
                        ],
                    )
                    for x in cutlass.range_constexpr(4):
                        dv_acc[4 * nt + x] = c[x]
                    c = mma_mxf8(
                        da,
                        [bq[h], bq[2 + h]],
                        [
                            dk_acc[4 * nt],
                            dk_acc[4 * nt + 1],
                            dk_acc[4 * nt + 2],
                            dk_acc[4 * nt + 3],
                        ],
                        sfa=sfa_dk,
                    )
                    for x in cutlass.range_constexpr(4):
                        dk_acc[4 * nt + x] = c[x]

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
            # longest causal CTAs launch first
            grid=[h_kv * batch, nk_pad // self.BC, 1]
            if self.kb_major
            else [nk_pad // self.BC, h_kv, batch],
            block=[self.NUM_THREADS + 32 * self.DQ_WARPS, 1, 1],
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
        tidx, _, _ = cute.arch.thread_idx()
        if const_expr(self.kb_major):
            lin, kblock, _ = cute.arch.block_idx()
            hkv = lin % h_kv
            b = lin // h_kv
        else:
            kblock, hkv, b = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32

        beta, kc, s_dk0, s_dv, _ = bwd_scales(
            mStats, mDesc, b, hkv, softmax_scale, self.p_off
        )

        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sbase = Int32(
            storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint()
        )

        k_row0 = (b * h_kv + hkv) * nk_pad + kblock * self.BC
        gk = Int64(mK.iterator.toint()) + Int64(k_row0) * self.row_bytes
        gv = Int64(mV.iterator.toint()) + Int64(k_row0) * self.row_bytes
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

        self._ds_full = storage.mbar.data_ptr()
        self._ds_free = self._ds_full + 2
        self._full = self._ds_full + 4
        self._empty = self._full + self.stages
        if tidx == 0:
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(self._ds_full + i, self.NUM_THREADS)
                cute.arch.mbarrier_init(self._ds_free + i, 128)
            for i in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(self._full + i, 128)
                cute.arch.mbarrier_init(self._empty + i, self.NUM_THREADS)
        cute.arch.mbarrier_init_fence()
        if warp < 8:
            self._load_rows(sbase + self.k_off, gk, self.BC, tidx)
            self._load_rows(sbase + self.v_off, gv, self.BC, tidx)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        # K and V are visible to the dQ warps too
        cute.arch.sync_threads()
        if warp >= 8:
            cute.arch.setmaxregister_decrease(self.DQ_REGS)
            self._dq_loop(
                tidx,
                warp,
                lane,
                total_it,
                n_q,
                q_start,
                b,
                hkv,
                h_q,
                h_kv,
                nq_pad,
                sbase,
                gq,
                gdo,
                gdq,
                gl,
                gd,
            )
        else:
            cute.arch.setmaxregister_increase(self.MMA_REGS)
            self._main_ws(
                warp,
                lane,
                kblock,
                hkv,
                b,
                total_it,
                n_q,
                q_start,
                h_q,
                h_kv,
                nq_pad,
                seqlen_q,
                seqlen_k,
                sbase,
                beta,
                kc,
                s_dk0,
                s_dv,
                k_row0,
                mdK,
                mdV,
            )

    @cute.jit
    def _dq_wg(
        self,
        it: Int32,
        row0: Int32,
        sbase: Int32,
        dsb: Int32,
        gdq: Int64,
        warp: Int32,
        lane: Int32,
        dq_acc: cute.Tensor,
    ):
        d = self.d
        g = lane // 4
        t = lane % 4
        kb = sbase + self.k_off
        dchunks = d // 16
        sdq = sbase + self.dq_off
        gq_base = (
            sbase
            + self.ld_off
            + (it % self.stages) * self.ld_stage_bytes
            + 2 * self.BM * 4
        )
        # block i covers query rows `16 * mts[i]` and column chunk `warp + chs[i]`
        mts = [(self.DQ_WARPS * i) // dchunks for i in range(self.nb)]
        chs = [(self.DQ_WARPS * i) % dchunks for i in range(self.nb)]
        rows = sorted(set(mts))
        cols = sorted(set(chs))
        dq_acc.fill(0.0)
        if const_expr(self.bf16):
            self._dq_mma_bf16(kb, dsb, warp, lane, mts, chs, rows, cols, dq_acc)
        else:
            self._dq_mma_e4m3(
                kb, dsb, gq_base, warp, lane, mts, chs, rows, cols, dq_acc
            )
        for i in cutlass.range_constexpr(self.nb):
            for h in cutlass.range_constexpr(2):
                for r in cutlass.range_constexpr(2):
                    o = 8 * i + 4 * h + 2 * r
                    if const_expr(self.fixed_dq):
                        q_loc = mts[i] * 16 + g + 8 * r
                        # `fq = cq * gq` puts this tile's dQ in fixed point
                        fq = ld_shared_f32(
                            gq_base + (self.BM + q_loc) * 4
                        ) * ld_shared_f32(gq_base + q_loc * 4)
                        v = pack_fixed2(dq_acc[o] * fq, dq_acc[o + 1] * fq)
                        if const_expr(self.dq_bulk):
                            slot = (((warp * self.nb + i) * 2 + h) * 2 + r) * 32 + lane
                            st_shared_b64(sdq + slot * 8, v)
                        else:
                            col = (warp + chs[i]) * 16 + 8 * h + 2 * t
                            red_add_u64(
                                gdq + (Int64(row0 + q_loc) * d + Int64(col)) * 4, v
                            )
                    elif const_expr(self.dq_bulk):
                        # conflict-free fragment order, undone by `BwdDqPost`
                        slot = (((warp * self.nb + i) * 2 + h) * 2 + r) * 32 + lane
                        st_shared_v2_f32(sdq + slot * 8, dq_acc[o], dq_acc[o + 1])
                    else:
                        q_loc = mts[i] * 16 + g + 8 * r
                        col = (warp + chs[i]) * 16 + 8 * h + 2 * t
                        addr = gdq + (Int64(row0 + q_loc) * d + Int64(col)) * 4
                        red_add_v2_f32(addr, dq_acc[o], dq_acc[o + 1])
        if const_expr(self.dq_bulk):
            # make staging writes visible to the bulk-copy proxy
            cute.arch.fence_view_async_shared()

    @cute.jit
    def _dq_mma_bf16(self, kb, dsb, warp, lane, mts, chs, rows, cols, dq_acc):
        for ks in cutlass.range_constexpr(self.BC // 16):
            a = [
                ldsm_x4(
                    dsb
                    + self._ds_off(
                        m * 16 + (lane % 8) + ((lane // 8) % 2) * 8, lane // 16 + 2 * ks
                    )
                )
                for m in rows
            ]
            key = 16 * ks + (lane % 8) + ((lane // 8) % 2) * 8
            # `dQ += dS K`, K via transposing b16 `ldmatrix`
            bk = [
                ldsm_x4_trans(kb + self._swz_off(key, 2 * (warp + c) + lane // 16))
                for c in cols
            ]
            for i in cutlass.range_constexpr(self.nb):
                ai = a[rows.index(mts[i])]
                bi = bk[cols.index(chs[i])]
                for h in cutlass.range_constexpr(2):
                    o = 8 * i + 4 * h
                    c = mma_16816(
                        ai,
                        [bi[2 * h], bi[2 * h + 1]],
                        [dq_acc[o + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dq_acc[o + x] = c[x]

    @cute.jit
    def _dq_mma_e4m3(self, kb, dsb, gq_base, warp, lane, mts, chs, rows, cols, dq_acc):
        sfa = []
        # this lane scales A row `lane // 4 + 8 * (lane % 2)` of each block
        q_row = lane // 4 + 8 * (lane % 2)
        for m in rows:
            g2 = ld_shared_v2_f32(gq_base + (m * 16 + q_row - q_row % 2) * 4)
            sfa.append(ue8m0_inv(g2[1] if q_row % 2 == 1 else g2[0]))
        for ks in cutlass.range_constexpr(self.BC // 32):
            a = [
                ldsm_x4(
                    dsb
                    + self._ds_off(
                        m * 16 + (lane % 8) + ((lane // 8) % 2) * 8, lane // 16 + 2 * ks
                    )
                )
                for m in rows
            ]
            bk = [
                ldsm_x2_trans_b8(kb + self._swz_off(lane + 32 * ks, warp + c))
                for c in cols
            ]
            for i in cutlass.range_constexpr(self.nb):
                ai = a[rows.index(mts[i])]
                bi = bk[cols.index(chs[i])]
                for h in cutlass.range_constexpr(2):
                    o = 8 * i + 4 * h
                    c = mma_mxf8(
                        ai,
                        [bi[h], bi[2 + h]],
                        [dq_acc[o], dq_acc[o + 1], dq_acc[o + 2], dq_acc[o + 3]],
                        sfa=sfa[rows.index(mts[i])],
                    )
                    for x in cutlass.range_constexpr(4):
                        dq_acc[o + x] = c[x]

    @cute.jit
    def _produce(
        self,
        j: Int32,
        total_it: Int32,
        n_q: Int32,
        q_start: Int32,
        b: Int32,
        hkv: Int32,
        h_q: Int32,
        h_kv: Int32,
        nq_pad: Int32,
        sbase: Int32,
        gq: Int64,
        gdo: Int64,
        gl: Int64,
        gd: Int64,
        dtid: Int32,
    ):
        if j < total_it:
            self._load_q(
                j,
                total_it,
                n_q,
                q_start,
                b,
                hkv,
                h_q,
                h_kv,
                nq_pad,
                sbase,
                gq,
                gdo,
                gl,
                gd,
                dtid,
                128,
            )
            # counts once this thread's copies land
            cute.arch.cp_async_mbarrier_arrive_noinc(self._full + j % self.stages)

    @cute.jit
    def _dq_loop(
        self,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        total_it: Int32,
        n_q: Int32,
        q_start: Int32,
        b: Int32,
        hkv: Int32,
        h_q: Int32,
        h_kv: Int32,
        nq_pad: Int32,
        sbase: Int32,
        gq: Int64,
        gdo: Int64,
        gdq: Int64,
        gl: Int64,
        gd: Int64,
    ):
        d = self.d
        dwarp = warp - 8
        dtid = tidx - self.NUM_THREADS
        dq_acc = cute.make_rmem_tensor(8 * self.nb, Float32)
        load = (
            total_it,
            n_q,
            q_start,
            b,
            hkv,
            h_q,
            h_kv,
            nq_pad,
            sbase,
            gq,
            gdo,
            gl,
            gd,
            dtid,
        )
        for s in cutlass.range_constexpr(self.stages - 1):
            self._produce(Int32(s), *load)
        for it in cutlass.range(total_it, unroll=1):
            if const_expr(self.dq_bulk):
                # every dQ warp has staged the previous tile
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
                if dtid == 0:
                    if it > 0:
                        prow0, _ = self._q_row0(
                            it - 1, n_q, q_start, b, hkv, h_q, h_kv, nq_pad
                        )
                        self._bulk_add(
                            gdq + Int64(prow0) * (d * 4),
                            sbase + self.dq_off,
                            self.dq_stage_bytes,
                        )
                        cute.arch.cp_async_bulk_commit_group()
                    # wait until the staging buffer is read
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.barrier(barrier_id=2, number_of_threads=128)
            # refill the stage tile it - 1 used
            j = it + (self.stages - 1)
            if j < total_it:
                if it > 0:
                    cute.arch.mbarrier_wait(
                        self._empty + (it - 1) % self.stages,
                        ((it - 1) // self.stages) & 1,
                    )
                self._produce(j, *load)
            if const_expr(not self.skip_dq):
                cute.arch.mbarrier_wait(self._ds_full + it % 2, (it // 2) & 1)
                row0, _ = self._q_row0(it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
                dsb = sbase + self.ds_off + (it % 2) * self.ds_bytes
                self._dq_wg(it, row0, sbase, dsb, gdq, dwarp, lane, dq_acc)
                cute.arch.mbarrier_arrive(self._ds_free + it % 2)
        if const_expr(self.dq_bulk):
            cute.arch.barrier(barrier_id=2, number_of_threads=128)
            if dtid == 0:
                if total_it > 0:
                    lrow0, _ = self._q_row0(
                        total_it - 1, n_q, q_start, b, hkv, h_q, h_kv, nq_pad
                    )
                    self._bulk_add(
                        gdq + Int64(lrow0) * (d * 4),
                        sbase + self.dq_off,
                        self.dq_stage_bytes,
                    )
                    cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def _store_dkdv(
        self,
        warp,
        lane,
        kblock,
        seqlen_k,
        s_dk0,
        s_dv,
        k_row0,
        dk_acc,
        dv_acc,
        mdK,
        mdV,
    ):
        d = self.d
        g = lane // 4
        t = lane % 4
        # `dK = alpha*sigma_Q * A^K`, `dV = sigma_dO*2^-8 * A^V`, bf16, one row per key
        s_dk = s_dk0
        gdk = Int64(mdK.iterator.toint())
        gdv = Int64(mdV.iterator.toint())
        for r in cutlass.range_constexpr(2):
            key = kblock * self.BC + warp * 16 + g + 8 * r
            if key < seqlen_k:
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
    def _main_ws(
        self,
        warp: Int32,
        lane: Int32,
        kblock: Int32,
        hkv: Int32,
        b: Int32,
        total_it: Int32,
        n_q: Int32,
        q_start: Int32,
        h_q: Int32,
        h_kv: Int32,
        nq_pad: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        sbase: Int32,
        beta: Float32,
        kc: Float32,
        s_dk0: Float32,
        s_dv: Float32,
        k_row0: Int32,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
    ):
        d = self.d
        s_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dp_acc = cute.make_rmem_tensor(4 * self.nt, Float32)
        dk_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        dv_acc = cute.make_rmem_tensor(4 * (d // 8), Float32)
        pa_t = cute.make_rmem_tensor(4 * self.m_steps, Int32)
        da_t = cute.make_rmem_tensor(4 * self.m_steps, Int32)
        # ue8m0 dK scale of this lane's A row for the packed tile
        sk_t = cute.make_rmem_tensor(1, Int32)
        sk_t.fill(0x7F)
        dk_acc.fill(0.0)
        dv_acc.fill(0.0)
        li = lane % 16
        v_row = ((li % 4) // 2) * 8 + (li // 4) * 2 + (li % 2) + (lane // 16) * 16
        tile = (n_q, q_start, b, hkv, h_q, h_kv, nq_pad, kblock, seqlen_q, seqlen_k)
        vf = None
        if const_expr(self.v_regs):
            # V's A fragments are the same for every tile
            vf = cute.make_rmem_tensor(4 * (self.row_bytes // 32), Int32)
            for kk in cutlass.range_constexpr(self.row_bytes // 32):
                arow = warp * 16 + (lane % 8) + ((lane // 8) % 2) * 8
                a_v = ldsm_x4(
                    sbase + self.v_off + self._swz_off(arow, lane // 16 + 2 * kk)
                )
                for x in cutlass.range_constexpr(4):
                    vf[4 * kk + x] = a_v[x]
        state = (s_acc, dp_acc, pa_t, da_t, sk_t, dk_acc, dv_acc, vf)

        if warp >= 4:
            # the second group starts after the first group's first S/dP
            cute.arch.barrier(barrier_id=3, number_of_threads=self.NUM_THREADS)
        if total_it > 0:
            self._ws_step(
                Int32(0), True, sbase, warp, lane, beta, kc, v_row, *tile, *state
            )
        elif warp < 4:
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=self.NUM_THREADS)
        for it in cutlass.range(1, total_it, unroll=1):
            self._ws_step(it, False, sbase, warp, lane, beta, kc, v_row, *tile, *state)
        if total_it > 0:
            st = (total_it - 1) % self.stages
            self._mma_dvdk(
                sbase + self.q_off + st * self.q_tile_bytes,
                sbase + self.do_off + st * self.q_tile_bytes,
                v_row,
                pa_t,
                da_t,
                dv_acc,
                dk_acc,
                sk_t[0],
            )
        self._store_dkdv(
            warp,
            lane,
            kblock,
            seqlen_k,
            s_dk0,
            s_dv,
            k_row0,
            dk_acc,
            dv_acc,
            mdK,
            mdV,
        )

    @cute.jit
    def _ws_step(
        self,
        it: Int32,
        first: cutlass.Constexpr,
        sbase: Int32,
        warp: Int32,
        lane: Int32,
        beta: Float32,
        kc: Float32,
        v_row: Int32,
        n_q: Int32,
        q_start: Int32,
        b: Int32,
        hkv: Int32,
        h_q: Int32,
        h_kv: Int32,
        nq_pad: Int32,
        kblock: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        s_acc: cute.Tensor,
        dp_acc: cute.Tensor,
        pa_t: cute.Tensor,
        da_t: cute.Tensor,
        sk_t: cute.Tensor,
        dk_acc: cute.Tensor,
        dv_acc: cute.Tensor,
        vf,
    ):
        st = it % self.stages
        pst = (it + self.stages - 1) % self.stages
        qb = sbase + self.q_off + st * self.q_tile_bytes
        ob = sbase + self.do_off + st * self.q_tile_bytes
        ldb = sbase + self.ld_off + st * self.ld_stage_bytes
        _, qi = self._q_row0(it, n_q, q_start, b, hkv, h_q, h_kv, nq_pad)
        cute.arch.mbarrier_wait(self._full + st, (it // self.stages) & 1)

        self._mma_sdp(
            sbase,
            qb,
            ob,
            sbase + self.dolo_off + st * self.q_tile_bytes,
            warp,
            lane,
            s_acc,
            dp_acc,
            vf,
        )
        if const_expr(first):
            if warp < 4:
                cute.arch.barrier_arrive(
                    barrier_id=3, number_of_threads=self.NUM_THREADS
                )
        else:
            # the previous tile's dV/dK, overlapping this tile's softmax
            self._mma_dvdk(
                sbase + self.q_off + pst * self.q_tile_bytes,
                sbase + self.do_off + pst * self.q_tile_bytes,
                v_row,
                pa_t,
                da_t,
                dv_acc,
                dk_acc,
                sk_t[0],
            )
        # dS in true units; the dK and dQ copies each get their own scale
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
            Float32(1.0),
            s_acc,
            dp_acc,
        )
        gk = None if const_expr(self.bf16) else self._key_scales(dp_acc)
        if const_expr(not first):
            cute.arch.mbarrier_arrive(self._empty + pst)
        if const_expr(not self.skip_dq):
            self._store_ds(
                it,
                sbase + self.ds_off + (it % 2) * self.ds_bytes,
                warp,
                lane,
                dp_acc,
                ldb + 2 * self.BM * 4,
            )
        self._pack(s_acc, dp_acc, pa_t, da_t, gk)
        if const_expr(not self.bf16):
            # this lane scales dK's A row `lane // 4 + 8 * (lane % 2)`, key row `lane % 2` here
            sk_t[0] = ue8m0_inv(gk[1] if lane % 2 == 1 else gk[0])


class AttnBwdDqSm120:
    BM = 128
    BN = 64
    NUM_THREADS = 256

    def __init__(
        self, head_dim: int, causal: bool, num_stages: int = 2, bf16: bool = False
    ):
        assert head_dim in (64, 128)
        self.d = head_dim
        self.causal = causal
        self.stages = num_stages
        self.bf16 = bf16
        self.p_off = 0.0 if bf16 else P_OFFSET
        self.row_bytes = head_dim * (2 if bf16 else 1)
        self.nt = self.BN // 8
        self.q_bytes = self.BM * self.row_bytes
        self.kv_tile_bytes = self.BN * self.row_bytes
        self.q_off = 0
        self.do_off = self.q_bytes
        self.dolo_off = 2 * self.q_bytes
        kv_bytes = 2 * num_stages * self.kv_tile_bytes
        if bf16:
            # K/V stages reuse the Q and dO space once their fragments are in registers
            self.k_off = 0
            self.smem_bytes = max(2 * self.q_bytes, kv_bytes)
        else:
            self.k_off = 3 * self.q_bytes
            self.smem_bytes = 3 * self.q_bytes + kv_bytes
        self.v_off = self.k_off + num_stages * self.kv_tile_bytes
        assert self.smem_bytes <= smem_limit(120)
        nbytes = self.smem_bytes

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        self.shared_storage = SharedStorage

    def _swz_off(self, row, chunk):
        return swizzle_offset(self.row_bytes, row, chunk)

    def _mma(self, a, b, c):
        if self.bf16:
            return mma_16816(a, b, c, "bf16")
        return mma_mxf8(a, b, c)

    def _load_rows(self, s_addr, g_addr, nrows, tidx):
        chunks = self.row_bytes // 16
        for i in range(nrows * chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // chunks
            ch = idx % chunks
            cp_async16(
                s_addr + self._swz_off(row, ch),
                g_addr + Int64(row * self.row_bytes + ch * 16),
            )

    @cute.jit
    def _load_tile(
        self,
        jn: Int32,
        n_blocks: Int32,
        sbase: Int32,
        gk: Int64,
        gv: Int64,
        tidx: Int32,
    ):
        if jn < n_blocks:
            sn = jn % self.stages
            self._load_rows(
                sbase + self.k_off + sn * self.kv_tile_bytes,
                gk + Int64(jn) * (self.BN * self.row_bytes),
                self.BN,
                tidx,
            )
            self._load_rows(
                sbase + self.v_off + sn * self.kv_tile_bytes,
                gv + Int64(jn) * (self.BN * self.row_bytes),
                self.BN,
                tidx,
            )

    @cute.jit
    def _tile(
        self,
        j: Int32,
        n_blocks: Int32,
        sbase: Int32,
        gk: Int64,
        gv: Int64,
        tidx: Int32,
        lane: Int32,
        t: Int32,
        beta: Float32,
        kappa0: Float32,
        kappa1: Float32,
        l0: Float32,
        l1: Float32,
        dk0: Float32,
        dk1: Float32,
        lim0: Int32,
        lim1: Int32,
        v_key: Int32,
        sfa: Int32,
        qf: cute.Tensor,
        of: cute.Tensor,
        olf: cute.Tensor,
        s_acc: cute.Tensor,
        dp_acc: cute.Tensor,
        dq_acc: cute.Tensor,
    ):
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()
        self._load_tile(j + (self.stages - 1), n_blocks, sbase, gk, gv, tidx)
        cute.arch.cp_async_commit_group()
        kbase = sbase + self.k_off + (j % self.stages) * self.kv_tile_bytes
        vbase = sbase + self.v_off + (j % self.stages) * self.kv_tile_bytes

        # `S = Q K^T`, `dP = dO V^T`, rows are this warp's 16 queries
        s_acc.fill(0.0)
        dp_acc.fill(0.0)
        for kk in cutlass.range_constexpr(self.row_bytes // 32):
            for npair in cutlass.range_constexpr(self.nt // 2):
                key = (lane % 8) + (lane // 16) * 8 + 16 * npair
                ch = (lane // 8) % 2 + 2 * kk
                bk = ldsm_x4(kbase + self._swz_off(key, ch))
                bv = ldsm_x4(vbase + self._swz_off(key, ch))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = self._mma(
                        [qf[4 * kk], qf[4 * kk + 1], qf[4 * kk + 2], qf[4 * kk + 3]],
                        [bk[2 * h], bk[2 * h + 1]],
                        [
                            s_acc[4 * nt],
                            s_acc[4 * nt + 1],
                            s_acc[4 * nt + 2],
                            s_acc[4 * nt + 3],
                        ],
                    )
                    for x in cutlass.range_constexpr(4):
                        s_acc[4 * nt + x] = c[x]
                    c = self._mma(
                        [of[4 * kk], of[4 * kk + 1], of[4 * kk + 2], of[4 * kk + 3]],
                        [bv[2 * h], bv[2 * h + 1]],
                        [
                            dp_acc[4 * nt],
                            dp_acc[4 * nt + 1],
                            dp_acc[4 * nt + 2],
                            dp_acc[4 * nt + 3],
                        ],
                    )
                    if const_expr(not self.bf16):
                        # dO's residual at 2^-4 through the A scale
                        c = mma_mxf8(
                            [
                                olf[4 * kk],
                                olf[4 * kk + 1],
                                olf[4 * kk + 2],
                                olf[4 * kk + 3],
                            ],
                            [bv[2 * h], bv[2 * h + 1]],
                            c,
                            sfa=0x7B,
                        )
                    for x in cutlass.range_constexpr(4):
                        dp_acc[4 * nt + x] = c[x]

        # `dS*g = P8 * (dP*kappa - D*g*2^-8)` with the per-query `g`; masked keys get `P8 = 0`
        kb = j * self.BN + t * 2
        for r in cutlass.range_constexpr(2):
            lq = l0 if r == 0 else l1
            kappa = kappa0 if r == 0 else kappa1
            dq = dk0 if r == 0 else dk1
            lim = lim0 if r == 0 else lim1
            for nt in cutlass.range_constexpr(self.nt):
                for e in cutlass.range_constexpr(2):
                    idx = 4 * nt + 2 * r + e
                    p8 = cute.math.exp2(
                        s_acc[idx] * beta - lq + Float32(self.p_off), fastmath=True
                    )
                    p8 = p8 if kb + (8 * nt + e) < lim else Float32(0.0)
                    dp_acc[idx] = p8 * (dp_acc[idx] * kappa - dq)

        if const_expr(self.bf16):
            self._dq_bf16(kbase, lane, dp_acc, dq_acc)
        else:
            self._dq_e4m3(kbase, v_key, sfa, dp_acc, dq_acc)

    @cute.jit
    def _dq_bf16(self, kbase, lane, dp_acc, dq_acc):
        for kk in cutlass.range_constexpr(self.BN // 16):
            nt0 = 2 * kk
            da = [
                pack_half2(dp_acc[4 * nt0 + 2 * x], dp_acc[4 * nt0 + 2 * x + 1], "bf16")
                for x in range(4)
            ]
            key = 16 * kk + (lane % 8) + ((lane // 8) % 2) * 8
            for npair in cutlass.range_constexpr(self.d // 16):
                # `dQ += dS K`, K via transposing b16 `ldmatrix`
                bk = ldsm_x4_trans(kbase + self._swz_off(key, 2 * npair + lane // 16))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_16816(
                        da,
                        [bk[2 * h], bk[2 * h + 1]],
                        [dq_acc[4 * nt + x] for x in range(4)],
                        "bf16",
                    )
                    for x in cutlass.range_constexpr(4):
                        dq_acc[4 * nt + x] = c[x]

    @cute.jit
    def _dq_e4m3(self, kbase, v_key, sfa, dp_acc, dq_acc):
        d = self.d
        # `dQ += dS K`; dS packed from the accumulators, K via transposing 8-bit `ldmatrix`
        for kk2 in cutlass.range_constexpr(self.BN // 32):
            t0 = 4 * kk2
            da = [
                pack_e4m3x4(
                    dp_acc[4 * t0],
                    dp_acc[4 * t0 + 1],
                    dp_acc[4 * t0 + 4],
                    dp_acc[4 * t0 + 5],
                ),
                pack_e4m3x4(
                    dp_acc[4 * t0 + 2],
                    dp_acc[4 * t0 + 3],
                    dp_acc[4 * t0 + 6],
                    dp_acc[4 * t0 + 7],
                ),
                pack_e4m3x4(
                    dp_acc[4 * t0 + 8],
                    dp_acc[4 * t0 + 9],
                    dp_acc[4 * t0 + 12],
                    dp_acc[4 * t0 + 13],
                ),
                pack_e4m3x4(
                    dp_acc[4 * t0 + 10],
                    dp_acc[4 * t0 + 11],
                    dp_acc[4 * t0 + 14],
                    dp_acc[4 * t0 + 15],
                ),
            ]
            key = v_key + 32 * kk2
            for npair in cutlass.range_constexpr(d // 16):
                bk = ldsm_x2_trans_b8(kbase + self._swz_off(key, npair))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_mxf8(
                        da,
                        [bk[h], bk[2 + h]],
                        [
                            dq_acc[4 * nt],
                            dq_acc[4 * nt + 1],
                            dq_acc[4 * nt + 2],
                            dq_acc[4 * nt + 3],
                        ],
                        sfa=sfa,
                    )
                    for x in cutlass.range_constexpr(4):
                        dq_acc[4 * nt + x] = c[x]

    @cute.jit
    def _load_frags(self, sbase, warp, lane, qf, of, olf):
        for kq in cutlass.range_constexpr(self.row_bytes // 32):
            arow = (lane % 8) + ((lane // 8) % 2) * 8 + warp * 16
            ach = lane // 16 + 2 * kq
            qfrag = ldsm_x4(sbase + self.q_off + self._swz_off(arow, ach))
            ofrag = ldsm_x4(sbase + self.do_off + self._swz_off(arow, ach))
            for x in cutlass.range_constexpr(4):
                qf[4 * kq + x] = qfrag[x]
                of[4 * kq + x] = ofrag[x]
            if const_expr(not self.bf16):
                lfrag = ldsm_x4(sbase + self.dolo_off + self._swz_off(arow, ach))
                for x in cutlass.range_constexpr(4):
                    olf[4 * kq + x] = lfrag[x]

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
        mdQ: cute.Tensor,
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
            mdQ,
            softmax_scale,
            seqlen_q,
            seqlen_k,
            nq_pad,
            nk_pad,
            h_q,
            h_kv,
        ).launch(
            grid=[nq_pad // self.BM, h_q, batch],
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
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        nq_pad: Int32,
        nk_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
    ):
        d = self.d
        kq_steps = self.row_bytes // 32
        tidx, _, _ = cute.arch.thread_idx()
        m_block, hq, b = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        g = lane // 4
        t = lane % 4
        hkv = hq // (h_q // h_kv)
        beta, kc, _, _, s_dq0 = bwd_scales(
            mStats, mDesc, b, hkv, softmax_scale, self.p_off
        )

        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sbase = Int32(
            storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint()
        )

        q_row0 = (b * h_q + hq) * nq_pad + m_block * self.BM
        k_row0 = (b * h_kv + hkv) * nk_pad
        gq = Int64(mQ.iterator.toint()) + Int64(q_row0) * self.row_bytes
        gdo = Int64(mdO.iterator.toint()) + Int64(q_row0) * self.row_bytes
        gdolo = Int64(mdOlo.iterator.toint()) + Int64(q_row0) * d
        gk = Int64(mK.iterator.toint()) + Int64(k_row0) * self.row_bytes
        gv = Int64(mV.iterator.toint()) + Int64(k_row0) * self.row_bytes

        n_blocks = nk_pad // self.BN
        if const_expr(self.causal):
            kmax = m_block * self.BM + seqlen_k - seqlen_q + (self.BM - 1)
            nb = (kmax + self.BN) // self.BN
            nb = nb if nb > 0 else Int32(0)
            n_blocks = min(n_blocks, nb)

        self._load_rows(sbase + self.q_off, gq, self.BM, tidx)
        self._load_rows(sbase + self.do_off, gdo, self.BM, tidx)
        if const_expr(not self.bf16):
            self._load_rows(sbase + self.dolo_off, gdolo, self.BM, tidx)
        cute.arch.cp_async_commit_group()
        qf = cute.make_rmem_tensor(4 * kq_steps, Int32)
        of = cute.make_rmem_tensor(4 * kq_steps, Int32)
        olf = cute.make_rmem_tensor(4 * kq_steps, Int32)
        if const_expr(self.bf16):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            self._load_frags(sbase, warp, lane, qf, of, olf)
            # every warp has its fragments before K/V overwrite Q and dO
            cute.arch.sync_threads()
        for s in cutlass.range_constexpr(self.stages - 1):
            self._load_tile(Int32(s), n_blocks, sbase, gk, gv, tidx)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()
        if const_expr(not self.bf16):
            self._load_frags(sbase, warp, lane, qf, of, olf)

        lrow = q_row0 + warp * 16 + g
        l0 = mL[lrow]
        l1 = mL[lrow + 8]
        g0 = Float32(1.0)
        g1 = Float32(1.0)
        if const_expr(not self.bf16):
            # per-query dS scales; the dQ MMA takes their inverse as A row scales
            g0 = mG[lrow]
            g1 = mG[lrow + 8]
        dk0 = mD[lrow] * g0 * Float32(2.0**-self.p_off)
        dk1 = mD[lrow + 8] * g1 * Float32(2.0**-self.p_off)
        # this lane scales A row `lane // 4 + 8 * (lane % 2)`
        sfa = ue8m0_inv(g1 if lane % 2 == 1 else g0)
        s_dq = s_dq0
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
            self._tile(
                j,
                n_blocks,
                sbase,
                gk,
                gv,
                tidx,
                lane,
                t,
                beta,
                g0 * kc,
                g1 * kc,
                l0,
                l1,
                dk0,
                dk1,
                lim0,
                lim1,
                v_key,
                sfa,
                qf,
                of,
                olf,
                s_acc,
                dp_acc,
                dq_acc,
            )

        gdq = Int64(mdQ.iterator.toint())
        for r in cutlass.range_constexpr(2):
            if q_idx0 + 8 * r < seqlen_q:
                orow = Int64(lrow + 8 * r)
                for nt in cutlass.range_constexpr(d // 8):
                    addr = gdq + (orow * d + Int64(8 * nt + 2 * t)) * 2
                    st_global_b32(
                        addr,
                        pack_half2(
                            dq_acc[4 * nt + 2 * r] * s_dq,
                            dq_acc[4 * nt + 2 * r + 1] * s_dq,
                            "bf16",
                        ),
                    )
