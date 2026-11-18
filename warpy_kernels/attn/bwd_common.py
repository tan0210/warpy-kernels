# type: ignore

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, Int64, const_expr, cute

from ..ptx import (
    ds_scale,
    e4m3x4_to_f32,
    ld_global_bf16,
    pack_e4m3x4,
    pack_half2,
    red_max_pos_f32,
    st_global_b16,
    st_global_b32,
    st_global_zero,
)
from .constants import LOG2E, P_OFFSET


class BwdPrep:
    ROWS_PER_CTA = 8

    def __init__(self, head_dim: int, zero_dq: bool = True, bf16: bool = False):
        self.d = head_dim
        self.per_lane = head_dim // 32
        self.zero_dq = zero_dq
        self.bf16 = bf16

    def _row_coords(self, row, nq, h_q, h_kv):
        b = row // (h_q * nq)
        rem = row % (h_q * nq)
        hq = rem // nq
        return b, hq, rem % nq, hq // (h_q // h_kv)

    @cute.jit
    def __call__(
        self,
        mdO: cute.Tensor,
        mO: cute.Tensor,
        mPmax: cute.Tensor,
        mdO8: cute.Tensor,
        mdO8lo: cute.Tensor,
        mD: cute.Tensor,
        mG: cute.Tensor,
        mStats: cute.Tensor,
        mDqAcc: cute.Tensor,
        rows: Int32,
        nq: Int32,
        nq_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
        stream: cuda.CUstream,
    ):
        nblk = (rows + (self.ROWS_PER_CTA - 1)) // self.ROWS_PER_CTA
        rows_pad = (rows // nq) * nq_pad
        nblk_pad = (rows_pad + (self.ROWS_PER_CTA - 1)) // self.ROWS_PER_CTA
        self.k_amax(mdO, mStats, mDqAcc, rows_pad, nq, nq_pad, h_q, h_kv).launch(
            grid=[nblk_pad, 1, 1], block=[256, 1, 1], stream=stream
        )
        self.k_quant(
            mdO,
            mO,
            mPmax,
            mdO8,
            mdO8lo,
            mD,
            mG,
            mStats,
            mDqAcc,
            rows,
            nq,
            nq_pad,
            h_q,
            h_kv,
        ).launch(grid=[nblk, 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def k_amax(
        self,
        mdO: cute.Tensor,
        mStats: cute.Tensor,
        mDqAcc: cute.Tensor,
        rows_pad: Int32,
        nq: Int32,
        nq_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
    ):
        # real rows feed the dO amax; padded rows zero dQaccum
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        prow = bidx * self.ROWS_PER_CTA + tidx // 32
        b, hq, i, hkv = self._row_coords(prow, nq_pad, h_q, h_kv)
        row = (b * h_q + hq) * nq + i
        if const_expr(self.zero_dq):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
            if prow < rows_pad and i >= nq:
                gz = (
                    Int64(mDqAcc.iterator.toint())
                    + (Int64(prow) * self.d + Int64(lane * self.per_lane)) * 4
                )
                st_global_zero(gz, self.per_lane)
        # bf16 dO is read as is, so it needs no amax
        if const_expr(not self.bf16):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
            if prow < rows_pad and i < nq:
                m = Float32(0.0)
                xin = ld_global_bf16(
                    Int64(mdO.iterator.toint())
                    + (Int64(row) * self.d + Int64(lane * self.per_lane)) * 2,
                    self.per_lane,
                )
                for e in cutlass.range_constexpr(self.per_lane):
                    m = cute.arch.fmax(m, xin[e] if xin[e] > Float32(0.0) else -xin[e])
                m = cute.arch.warp_reduction_max(m, threads_in_group=32)
                if lane == 0:
                    red_max_pos_f32(
                        Int64(mStats.iterator.toint())
                        + Int64((b * h_kv + hkv) * 4) * 4,
                        m,
                    )

    @cute.kernel
    def k_quant(
        self,
        mdO: cute.Tensor,
        mO: cute.Tensor,
        mPmax: cute.Tensor,
        mdO8: cute.Tensor,
        mdO8lo: cute.Tensor,
        mD: cute.Tensor,
        mG: cute.Tensor,
        mStats: cute.Tensor,
        mDqAcc: cute.Tensor,
        rows: Int32,
        nq: Int32,
        nq_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        row = bidx * self.ROWS_PER_CTA + tidx // 32
        if const_expr(self.bf16):
            if row < rows:
                b, hq, i, hkv = self._row_coords(row, nq, h_q, h_kv)
                col0 = lane * self.per_lane
                xin = ld_global_bf16(
                    Int64(mdO.iterator.toint())
                    + (Int64(row) * self.d + Int64(col0)) * 2,
                    self.per_lane,
                )
                oin = ld_global_bf16(
                    Int64(mO.iterator.toint())
                    + (Int64(row) * self.d + Int64(col0)) * 2,
                    self.per_lane,
                )
                prow = (b * h_q + hq) * nq_pad + i
                if const_expr(self.zero_dq):
                    gz = (
                        Int64(mDqAcc.iterator.toint())
                        + (Int64(prow) * self.d + Int64(col0)) * 4
                    )
                    st_global_zero(gz, self.per_lane)
                dsum = Float32(0.0)
                for e in cutlass.range_constexpr(self.per_lane):
                    dsum = dsum + xin[e] * oin[e]
                dsum = cute.arch.warp_reduction_sum(dsum, threads_in_group=32)
                if lane == 0:
                    mD[prow] = dsum
        else:
            if row < rows:
                b, hq, i, hkv = self._row_coords(row, nq, h_q, h_kv)
                amax = mStats[b, hkv, 0]
                s_do = (
                    amax * Float32(1.0 / 256.0) if amax > Float32(0.0) else Float32(1.0)
                )
                inv = Float32(1.0) / s_do
                col0 = lane * self.per_lane
                xin = ld_global_bf16(
                    Int64(mdO.iterator.toint())
                    + (Int64(row) * self.d + Int64(col0)) * 2,
                    self.per_lane,
                )
                oin = ld_global_bf16(
                    Int64(mO.iterator.toint())
                    + (Int64(row) * self.d + Int64(col0)) * 2,
                    self.per_lane,
                )
                x = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for e in cutlass.range_constexpr(self.per_lane):
                    x[e] = xin[e] * inv
                bits = pack_e4m3x4(x[0], x[1], x[2], x[3])
                deq = e4m3x4_to_f32(bits)
                prow = (b * h_q + hq) * nq_pad + i
                off = Int64(prow) * self.d + Int64(col0)
                gdst = Int64(mdO8.iterator.toint()) + off
                if const_expr(self.per_lane == 4):
                    st_global_b32(gdst, bits)
                else:
                    st_global_b16(gdst, bits)
                # the residual at 2^-4, which the dP product adds with a ue8m0 scale of 2^-4
                lo = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                for e in cutlass.range_constexpr(self.per_lane):
                    lo[e] = (x[e] - deq[e]) * Float32(16.0)
                bits_lo = pack_e4m3x4(lo[0], lo[1], lo[2], lo[3])
                deq_lo = e4m3x4_to_f32(bits_lo)
                gdst_lo = Int64(mdO8lo.iterator.toint()) + off
                if const_expr(self.per_lane == 4):
                    st_global_b32(gdst_lo, bits_lo)
                else:
                    st_global_b16(gdst_lo, bits_lo)
                if const_expr(self.zero_dq):
                    gz = (
                        Int64(mDqAcc.iterator.toint())
                        + (Int64(prow) * self.d + Int64(col0)) * 4
                    )
                    st_global_zero(gz, self.per_lane)
                dsum = Float32(0.0)
                nsum = Float32(0.0)
                for e in cutlass.range_constexpr(self.per_lane):
                    v = deq[e] * s_do
                    # D matches the two-term dO that dP sees
                    dsum = (
                        dsum
                        + (deq[e] + deq_lo[e] * Float32(1.0 / 16.0)) * s_do * oin[e]
                    )
                    nsum = nsum + v * v
                dsum = cute.arch.warp_reduction_sum(dsum, threads_in_group=32)
                nsum = cute.arch.warp_reduction_sum(nsum, threads_in_group=32)
                if lane == 0:
                    mD[prow] = dsum
                    # bound `|dS| <= 2 pmax |dO| max|V|` for this query
                    rb = (
                        Float32(2.0)
                        * mPmax[row]
                        * cute.math.sqrt(nsum, fastmath=True)
                        * mStats[b, hkv, 2]
                    )
                    # per-query dS scale of the dQ product
                    mG[prow] = ds_scale(rb)


@cute.jit
def bwd_scales(
    mStats: cute.Tensor,
    mDesc: cute.Tensor,
    b: Int32,
    hkv: Int32,
    softmax_scale: Float32,
    p_off: cutlass.Constexpr = P_OFFSET,
):
    sig_q = mDesc[b, hkv, 0]
    sig_k = mDesc[b, hkv, 1]
    sig_v = mDesc[b, hkv, 2]
    amax = mStats[b, hkv, 0]
    s_do = amax * Float32(1.0 / 256.0) if amax > Float32(0.0) else Float32(1.0)
    # `(beta, kc, dK scale, dV scale, dQ scale)`; `kappa = kc * g_tile`
    return (
        softmax_scale * sig_q * sig_k * LOG2E,
        s_do * sig_v * Float32(2.0**-p_off),
        softmax_scale * sig_q,
        s_do * Float32(2.0**-p_off),
        softmax_scale * sig_k,
    )


class BwdDqPost:
    NUM_THREADS = 256

    def __init__(self, head_dim: int, block_m: int, staged: bool):
        self.d = head_dim
        self.BM = block_m
        self.staged = staged
        self.num_warps = 4  # the dQ warpgroup of `AttnBwdSm120`
        self.nb = (block_m // 16) * (head_dim // 16) // self.num_warps
        self.tile_floats = block_m * head_dim
        self.pairs_per_thread = self.tile_floats // 2 // self.NUM_THREADS
        nfl = self.tile_floats

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, nfl], 16]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        mAcc: cute.Tensor,
        mStats: cute.Tensor,
        mDesc: cute.Tensor,
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        nq_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
        batch: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mAcc, mStats, mDesc, mdQ, softmax_scale, seqlen_q, nq_pad, h_q, h_kv
        ).launch(
            grid=[nq_pad // self.BM, h_q, batch],
            block=[self.NUM_THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes() if const_expr(self.staged) else 0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mAcc: cute.Tensor,
        mStats: cute.Tensor,
        mDesc: cute.Tensor,
        mdQ: cute.Tensor,
        softmax_scale: Float32,
        seqlen_q: Int32,
        nq_pad: Int32,
        h_q: Int32,
        h_kv: Int32,
    ):
        d = self.d
        tidx, _, _ = cute.arch.thread_idx()
        tile, hq, b = cute.arch.block_idx()
        hkv = hq // (h_q // h_kv)
        _, _, _, _, s_dq0 = bwd_scales(mStats, mDesc, b, hkv, softmax_scale)
        s_dq = s_dq0
        acc0 = ((b * h_q + hq) * nq_pad + tile * self.BM) * d
        gout = Int64(mdQ.iterator.toint())
        dchunks = d // 16
        if const_expr(self.staged):
            # coalesced copy into shared memory, then coalesced row-major writes
            storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
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
                # inverse of `slot = ((((warp * nb + i) * 2 + h) * 2 + r) * 32 + lane)` in `_dq_wg`
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
                addr = (
                    gout + (Int64((b * h_q + hq) * seqlen_q + qg) * d + Int64(col)) * 2
                )
                st_global_b32(addr, pack_half2(x0 * s_dq, x1 * s_dq, "bf16"))
