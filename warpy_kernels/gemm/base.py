# type: ignore

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, Int64, cute
from cutlass.base_dsl.ast_helpers import const_expr

from ..arch import smem_limit
from ..ptx import (
    cp_async16,
    ld_global_b32,
    ld_global_nc_f32,
    ld_global_nc_v2_f32,
    pack_half2,
    st_global_b32,
    st_global_v4_b32,
    unpack_half2,
)
from ..utils.fragments import Accum, ldsm_a, ldsm_a_trans, ldsm_b, ldsm_b_trans
from ..utils.layout import GmemRows, PanelTile, SmemTile
from ..utils.math import dgelu, gelu
from ..utils.pipeline import CpAsyncPipeline
from ..utils.sched import grouped_tile
from ..utils.warp import quad_transpose


class GemmBase:
    """``out = op(A) @ op(B).T`` with a 128 x ``bn`` CTA tile, eight warps of 64 x ``bn / 4``, loaded by ``cp.async``.

    Subclasses provide ``_mma`` and may replace the operand pipeline and the kernel.
    """

    BM = 128
    NUM_THREADS = 256
    FP8 = False  # whether the class has an fp8 MMA
    OUT_MX = False  # whether the class can write MXFP8 outputs

    def __init__(
        self,
        ty: str = "bf16",
        bk: int = 64,
        stages: int = 3,
        group_m: int = 8,
        min_blocks: int = 1,
        bn: int = 128,
        frag_pipe: bool = False,
        fp8: bool = False,
        act: str = "",
        store_pre: bool = False,
        a_trans: bool = False,
        b_trans: bool = False,
        accum: bool = False,
        row_scale: bool = False,
        col_scale: bool = False,
        out_mx: bool = False,
        arch: int = 80,
    ):
        assert self.FP8 or not fp8, f"{type(self).__name__} has no fp8 MMA"
        assert self.OUT_MX or not out_mx, f"{type(self).__name__} cannot write MXFP8"
        assert not ((a_trans or b_trans) and fp8), (
            "transposed operands need 16-bit inputs"
        )
        assert act in ("", "gelu", "dgelu") and (act == "gelu" or not store_pre), (
            "store_pre needs act='gelu'"
        )
        self.arch = arch
        self.fp8 = fp8
        self.out_mx = out_mx
        self.accum = accum
        self.row_scale = row_scale
        self.col_scale = col_scale
        self.a_trans = a_trans
        self.b_trans = b_trans
        self.act = act
        self.store_pre = store_pre
        self.es = 1 if fp8 else 2  # bytes per input element
        assert (
            ty in ("bf16", "f16")
            and bk * self.es in (64, 128)
            and stages >= 2
            and bn in (128, 256)
        )
        assert bn == 128 or min_blocks == 1, (
            "a 64x64 warp tile needs 128 accumulator registers"
        )
        self.ty = ty
        self.frag_pipe = frag_pipe
        self.threads = self.NUM_THREADS
        self.BN = bn
        self.WN = bn // 4
        self.NT = self.WN // 8
        self.BK = bk
        self.row_bytes = bk * self.es
        self.KK = self.row_bytes // 32  # MMA k steps per tile
        self.sw_bits = 3 if self.row_bytes == 128 else 2  # 128B or 64B swizzle
        self.stages = stages
        self.group_m = group_m
        self.min_blocks = min_blocks
        self.a_bytes = self.BM * bk * self.es
        self.b_bytes = bn * bk * self.es
        self.smem_bytes = stages * (self.a_bytes + self.b_bytes)
        assert (self.smem_bytes + self._smem_overhead()) * min_blocks <= smem_limit(
            arch
        ), f"shared memory does not fit: {self.smem_bytes} B x {min_blocks} CTA(s)"
        self.shared_storage = self._make_storage()

    def _smem_overhead(self):
        return 0

    def _make_storage(self):
        nbytes = self.smem_bytes

        @cute.struct
        class SharedStorage:
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, nbytes], 1024]

        return SharedStorage

    def _mma(self, a, b, c, sfa):
        raise NotImplementedError

    def _sfa(self, sfp, sf16, kt, kk):
        return [Int32(0x7F)] * 4  # unit A-row scales

    def _a_tile(self, sbase, st):
        base = sbase + st * self.a_bytes
        if const_expr(self.a_trans):
            return PanelTile(base, self.BK, self.BM)
        return SmemTile(base, self.row_bytes, self.BM)

    def _b_tile(self, sbase, st):
        base = sbase + self.stages * self.a_bytes + st * self.b_bytes
        if const_expr(self.b_trans):
            return PanelTile(base, self.BK, self.BN)
        return SmemTile(base, self.row_bytes, self.BN)

    def _a_frag(self, sa, warp_m, mi, kk, lane):
        if const_expr(self.a_trans):
            return ldsm_a_trans(sa, warp_m * 64 + mi * 16, kk, lane)
        return ldsm_a(sa, warp_m * 64 + mi * 16, kk, lane)

    def _b_frag(self, sb, warp_n, nj, kk, lane):
        if const_expr(self.b_trans):
            return ldsm_b_trans(sb, warp_n * self.WN + nj * 16, kk, lane)
        return ldsm_b(sb, warp_n * self.WN + nj * 16, kk, lane)

    def _load_rows(self, dst, src, kt, tidx):
        per = 16 // self.es  # elements per 16-byte chunk
        for i in range(dst.rows * dst.chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // dst.chunks
            ch = idx % dst.chunks
            cp_async16(dst.addr(row, ch), src.addr(row, kt * self.BK + ch * per))

    def _load_panels(self, dst, src, kt, tidx):
        chunks = dst.width // 8
        for i in range(self.BK * chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // chunks
            ch = idx % chunks
            cp_async16(dst.addr(row, ch), src.addr(kt * self.BK + row, ch * 8))

    def _load(self, kt, st, sbase, ga, gb, tidx):
        sa = self._a_tile(sbase, st)
        sb = self._b_tile(sbase, st)
        if const_expr(self.a_trans):
            self._load_panels(sa, ga, kt, tidx)
        else:
            self._load_rows(sa, ga, kt, tidx)
        if const_expr(self.b_trans):
            self._load_panels(sb, gb, kt, tidx)
        else:
            self._load_rows(sb, gb, kt, tidx)

    def _operands(self, mA, mB, m, n, k, m0, n0):
        if const_expr(self.a_trans):
            ga = GmemRows(Int64(mA.iterator.toint()) + Int64(m0) * self.es, m, self.es)
        else:
            ga = GmemRows(
                Int64(mA.iterator.toint()) + Int64(m0) * Int64(k) * self.es, k, self.es
            )
        if const_expr(self.b_trans):
            gb = GmemRows(Int64(mB.iterator.toint()) + Int64(n0) * self.es, n, self.es)
        else:
            gb = GmemRows(
                Int64(mB.iterator.toint()) + Int64(n0) * Int64(k) * self.es, k, self.es
            )
        return ga, gb

    @cute.jit
    def _frag_load(self, st, kk, buf, sbase, warp_m, warp_n, lane, fa, fb):
        sa = self._a_tile(sbase, st)
        sb = self._b_tile(sbase, st)
        for mi in cutlass.range_constexpr(4):
            r = self._a_frag(sa, warp_m, mi, kk, lane)
            for x in cutlass.range_constexpr(4):
                fa[buf * 16 + mi * 4 + x] = r[x]
        for nj in cutlass.range_constexpr(self.NT // 2):
            r = self._b_frag(sb, warp_n, nj, kk, lane)
            for x in cutlass.range_constexpr(4):
                fb[buf * 2 * self.NT + nj * 4 + x] = r[x]

    @cute.jit
    def _mma_step(self, buf, fa, fb, acc, sfp, sf16, kt, kk):
        sfa = self._sfa(sfp, sf16, kt, kk)
        for nj in cutlass.range_constexpr(self.NT // 2):
            for h in cutlass.range_constexpr(2):
                b = [
                    fb[buf * 2 * self.NT + nj * 4 + 2 * h],
                    fb[buf * 2 * self.NT + nj * 4 + 2 * h + 1],
                ]
                for mi in cutlass.range_constexpr(4):
                    ao = buf * 16 + mi * 4
                    a = [fa[ao], fa[ao + 1], fa[ao + 2], fa[ao + 3]]
                    acc.mma(
                        mi,
                        2 * nj + h,
                        lambda a, b, c, s=sfa[mi]: self._mma(a, b, c, s),
                        a,
                        b,
                    )

    @cute.jit
    def _compute(self, st, sbase, warp_m, warp_n, lane, acc, sfp, sf16, kt):
        sa = self._a_tile(sbase, st)
        sb = self._b_tile(sbase, st)
        for kk in cutlass.range_constexpr(self.KK):
            sfa = self._sfa(sfp, sf16, kt, kk)
            a = []
            for mi in cutlass.range_constexpr(4):
                a.append(self._a_frag(sa, warp_m, mi, kk, lane))
            for nj in cutlass.range_constexpr(self.NT // 2):
                b = self._b_frag(sb, warp_n, nj, kk, lane)
                for h in cutlass.range_constexpr(2):
                    for mi in cutlass.range_constexpr(4):
                        acc.mma(
                            mi,
                            2 * nj + h,
                            lambda a, b, c, s=sfa[mi]: self._mma(a, b, c, s),
                            a[mi],
                            [b[2 * h], b[2 * h + 1]],
                        )

    @cute.jit
    def _run(self, pipe, nk, sbase, warp_m, warp_n, lane, acc, sfp, sf16):
        pipe.prologue()
        if const_expr(self.frag_pipe):
            fa = cute.make_rmem_tensor(32, Int32)
            fb = cute.make_rmem_tensor(4 * self.NT, Int32)
            pipe.wait_first()
            self._frag_load(Int32(0), 0, 0, sbase, warp_m, warp_n, lane, fa, fb)
            kk_n = self.KK
            issue_kk = pipe.issue_kk(kk_n)
            for it in cutlass.range(nk, unroll=1):
                st = it % self.stages
                for kk in cutlass.range_constexpr(kk_n):
                    if const_expr(kk == issue_kk):
                        pipe.issue(it + (self.stages - 1))
                    if const_expr(kk == kk_n - 1):
                        pipe.advance(it, lane)
                        # the last tile reads a stale, unused buffer
                        self._frag_load(
                            (it + 1) % self.stages,
                            0,
                            0,
                            sbase,
                            warp_m,
                            warp_n,
                            lane,
                            fa,
                            fb,
                        )
                    else:
                        self._frag_load(
                            st,
                            kk + 1,
                            (kk + 1) % 2,
                            sbase,
                            warp_m,
                            warp_n,
                            lane,
                            fa,
                            fb,
                        )
                    self._mma_step(kk % 2, fa, fb, acc, sfp, sf16, it, kk)
        else:
            for it in cutlass.range(nk, unroll=1):
                pipe.step_begin(it)
                self._compute(
                    it % self.stages, sbase, warp_m, warp_n, lane, acc, sfp, sf16, it
                )
                pipe.step_end(it, lane)

    def _sfp(self, mASF, k, m0, warp_m, g, t):
        return Int64(0)

    def _pipeline(
        self, storage, mA, mB, m, n, k, m0, n0, pid_m, pid_n, nk, tidx, tma_args
    ):
        sbase = Int32(
            storage.buf.get_tensor(cute.make_layout(self.smem_bytes)).iterator.toint()
        )
        ga, gb = self._operands(mA, mB, m, n, k, m0, n0)
        pipe = CpAsyncPipeline(
            self.stages,
            nk,
            lambda i: self._load(i, i % self.stages, sbase, ga, gb, tidx),
        )
        return sbase, pipe

    def _tma_args(self, mA, mB):
        return None

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPre: cute.Tensor,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
        mWs: cute.Tensor,
        mFlags: cute.Tensor,
        mQ: cute.Tensor,
        mQS: cute.Tensor,
        mASF: cute.Tensor,
        m: Int32,
        n: Int32,
        k: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        self._launch(
            mA,
            mB,
            mC,
            mPre,
            mSA,
            mSB,
            mWs,
            mFlags,
            mQ,
            mQS,
            mASF,
            m,
            n,
            k,
            ctas,
            stream,
        )

    def _launch(
        self,
        mA,
        mB,
        mC,
        mPre,
        mSA,
        mSB,
        mWs,
        mFlags,
        mQ,
        mQS,
        mASF,
        m,
        n,
        k,
        ctas,
        stream,
    ):
        self.kernel(
            mA, mB, mC, mPre, mSA, mSB, mQ, mQS, mASF, m, n, k, self._tma_args(mA, mB)
        ).launch(
            grid=[ctas, 1, 1],
            block=[self.NUM_THREADS, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPre: cute.Tensor,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
        mQ: cute.Tensor,
        mQS: cute.Tensor,
        mASF: cute.Tensor,
        m: Int32,
        n: Int32,
        k: Int32,
        tma_args,
    ):
        alpha = self._alpha(mSA, mSB)
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        warp_m = warp // 4
        warp_n = warp % 4
        g = lane // 4
        t = lane % 4
        pid_m, pid_n = grouped_tile(pid, m // self.BM, n // self.BN, self.group_m)
        m0 = pid_m * self.BM
        n0 = pid_n * self.BN
        sf16 = Int64(16) * Int64(k // 32)  # bytes between the A scales of rows 16 apart
        sfp = self._sfp(mASF, k, m0, warp_m, g, t)
        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        nk = k // self.BK
        sbase, pipe = self._pipeline(
            storage, mA, mB, m, n, k, m0, n0, pid_m, pid_n, nk, tidx, tma_args
        )
        acc = Accum(4, self.NT)
        acc.fill(0.0)
        self._run(pipe, nk, sbase, warp_m, warp_n, lane, acc, sfp, sf16)
        self._epilogue(
            mC, mPre, mSA, mSB, mQ, mQS, n, m0, n0, warp_m, warp_n, g, t, acc, alpha
        )

    def _alpha(self, mSA, mSB):
        alpha = Float32(1.0)
        if const_expr(self.fp8 and not self.row_scale):
            alpha = alpha * Float32(mSA[0])
        if const_expr(self.fp8 and not self.col_scale):
            alpha = alpha * Float32(mSB[0])
        return alpha

    def _store_mx(self, gq, gqs, n, row, c0, t, vals):
        raise NotImplementedError

    @cute.jit
    def _epilogue(
        self, mC, mPre, mSA, mSB, mQ, mQS, n, m0, n0, warp_m, warp_n, g, t, acc, alpha
    ):
        gc = Int64(mC.iterator.toint())
        gp = Int64(mPre.iterator.toint())
        gsa = Int64(mSA.iterator.toint())
        gsb = Int64(mSB.iterator.toint())
        gq = Int64(mQ.iterator.toint())
        gqs = Int64(mQS.iterator.toint())
        upw = self.NT // 4  # 16-byte stores per lane
        for mi in cutlass.range_constexpr(4):
            for r in cutlass.range_constexpr(2):
                row = m0 + warp_m * 64 + mi * 16 + g + 8 * r
                ra = alpha
                if const_expr(self.row_scale):
                    ra = alpha * ld_global_nc_f32(gsa + Int64(row) * 4)
                words = []
                vals = []
                for nt in cutlass.range_constexpr(self.NT):
                    col = n0 + warp_n * self.WN + nt * 8 + 2 * t
                    o = (mi * self.NT + nt) * 4 + 2 * r
                    off = (Int64(row) * Int64(n) + Int64(col)) * 2
                    v0 = acc[o] * ra
                    v1 = acc[o + 1] * ra
                    if const_expr(self.col_scale):
                        # `col` is even, so the pair is one 8-byte load
                        s0, s1 = ld_global_nc_v2_f32(gsb + Int64(col) * 4)
                        v0 = v0 * s0
                        v1 = v1 * s1
                    if const_expr(self.accum):
                        # add the output buffer before the activation
                        c0, c1 = unpack_half2(ld_global_b32(gc + off), self.ty)
                        v0 = v0 + c0
                        v1 = v1 + c1
                    if const_expr(self.act == "gelu"):
                        if const_expr(self.store_pre):
                            st_global_b32(gp + off, pack_half2(v0, v1, self.ty))
                        v0 = gelu(v0)
                        v1 = gelu(v1)
                    if const_expr(self.act == "dgelu"):
                        p0, p1 = unpack_half2(ld_global_b32(gp + off), self.ty)
                        v0 = v0 * dgelu(p0)
                        v1 = v1 * dgelu(p1)
                    words.append(pack_half2(v0, v1, self.ty))
                    vals.append(v0)
                    vals.append(v1)
                if const_expr(self.out_mx):
                    self._store_mx(gq, gqs, n, row, n0 + warp_n * self.WN, t, vals)
                # transpose within each quad so every lane holds 16 contiguous bytes
                units = quad_transpose(
                    [words[q * upw : (q + 1) * upw] for q in range(4)], t
                )
                for w in cutlass.range_constexpr(upw):
                    col = n0 + warp_n * self.WN + 8 * (t * upw + w)
                    st_global_v4_b32(
                        gc + (Int64(row) * Int64(n) + Int64(col)) * 2,
                        units[0][w],
                        units[1][w],
                        units[2][w],
                        units[3][w],
                    )
