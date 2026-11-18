# type: ignore

import cutlass
from cutlass import Int32, Int64, cute
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.cute.nvgpu import cpasync

from ..ptx import (
    fence_gpu,
    ld_global_cg_v4_f32,
    ld_global_nc_u8,
    mma_16816,
    mma_mxf8,
    st_global_v4_f32,
    st_release_gpu_b32,
    wait_flag_acquire,
)
from ..utils.fragments import Accum
from ..utils.pipeline import TmaPipeline
from ..utils.sched import grouped_tile
from .sm80 import GemmSm80
from .sm89 import GemmSm89


class GemmSm120(GemmSm89):
    """Adds block-scaled e4m3 MMAs, TMA loads, a producer warp, persistent stream-K tiles and register
    reallocation, for SM120.
    """

    def __init__(
        self,
        ty: str = "bf16",
        bk: int = 64,
        stages: int = 3,
        group_m: int = 8,
        min_blocks: int = 1,
        bn: int = 128,
        frag_pipe: bool = False,
        tma: bool = False,
        warp_spec: bool = False,
        persistent: bool = False,
        setmaxnreg: bool = False,
        fp8: bool = False,
        act: str = "",
        store_pre: bool = False,
        a_trans: bool = False,
        b_trans: bool = False,
        accum: bool = False,
        row_scale: bool = False,
        col_scale: bool = False,
        a_mx: bool = False,
        out_mx: bool = False,
        arch: int = 120,
    ):
        assert fp8 or not a_mx, "MX scales on A need fp8 inputs"
        assert tma or not warp_spec, "warp_spec needs tma"
        assert warp_spec or not persistent, "persistent needs warp_spec"
        assert warp_spec or not setmaxnreg, "setmaxnreg needs warp_spec"
        self.a_mx = a_mx
        self.tma = tma
        self.warp_spec = warp_spec
        self.persistent = persistent
        self.setmaxnreg = setmaxnreg
        super().__init__(
            ty,
            bk,
            stages,
            group_m,
            min_blocks,
            bn,
            frag_pipe,
            fp8,
            act,
            store_pre,
            a_trans,
            b_trans,
            accum,
            row_scale,
            col_scale,
            out_mx,
            arch,
        )
        # a producer warpgroup can hand its registers to the MMA warps
        self.threads = self.NUM_THREADS + (
            128 if setmaxnreg else 32 if warp_spec else 0
        )
        self.tx_bytes = self.a_bytes + self.b_bytes

    # the block-scaled MMA accumulates in full fp32, so SM120 keeps the plain k loop of `GemmSm80`
    _run = GemmSm80._run
    _compute = GemmSm80._compute
    _mma_step = GemmSm80._mma_step

    def _smem_overhead(self):
        return 1024 if self.tma else 0

    def _make_storage(self):
        if not self.tma:
            return super()._make_storage()
        edt = (
            cutlass.Uint8
            if self.fp8
            else cutlass.BFloat16
            if self.ty == "bf16"
            else cutlass.Float16
        )
        n_stage = self.stages
        a_elems = self.stages * self.BM * self.BK
        b_elems = self.stages * self.BN * self.BK

        @cute.struct
        class SharedStorage:
            # full barriers, then empty barriers
            mbar: cute.struct.MemRange[cutlass.Int64, 2 * n_stage]
            # A then B stages, 1024-byte aligned for the swizzle
            a: cute.struct.Align[cute.struct.MemRange[edt, a_elems], 1024]
            b: cute.struct.Align[cute.struct.MemRange[edt, b_elems], 1024]

        return SharedStorage

    def _mma(self, a, b, c, sfa):
        if self.a_mx:
            return mma_mxf8(a, b, c, sfa=sfa)
        if self.fp8:
            return mma_mxf8(
                a, b, c
            )  # twice as fast as `mma_fp8` here, with unit scales
        return mma_16816(a, b, c, self.ty)

    @cute.jit
    def _sfa(self, sfp, sf16, kt, kk):
        # this lane's A-row scale in each m16 tile, for k32 step `kk` of k-tile `kt`
        out = []
        for mi in cutlass.range_constexpr(4):
            if const_expr(self.a_mx):
                out.append(
                    ld_global_nc_u8(sfp + Int64(mi) * sf16 + Int64(kt * self.KK + kk))
                )
            else:
                out.append(Int32(0x7F))
        return out

    def _sfp(self, mASF, k, m0, warp_m, g, t):
        return Int64(mASF.iterator.toint()) + Int64(
            m0 + warp_m * 64 + g + 8 * (t % 2)
        ) * Int64(k // 32)

    @cute.jit
    def _stage_layout(self, rows: cutlass.Constexpr):
        # TMA reads the swizzle in bytes, so 128B is <3,4,3> and 64B is <2,4,3>
        sw = cute.make_swizzle(self.sw_bits, 4, 3)
        return cute.make_composed_layout(
            sw,
            0,
            cute.make_layout(
                (rows, self.BK, self.stages), stride=(self.BK, 1, rows * self.BK)
            ),
        )

    @cute.jit
    def _stage_layout_t(self, width: cutlass.Constexpr):
        # 64-column panels of 128B-swizzled rows, one `[BK, 64]` box per panel
        sw = cute.make_swizzle(3, 4, 3)
        panels = width // 64
        return cute.make_composed_layout(
            sw,
            0,
            cute.make_layout(
                (self.BK, 64, panels, self.stages),
                stride=(64, 1, self.BK * 64, panels * self.BK * 64),
            ),
        )

    def _tma_args(self, mA, mB):
        if not self.tma:
            return None
        if self.a_trans:
            one_a, box_a = (
                cute.slice_(self._stage_layout_t(self.BM), (None, None, 0, 0)),
                (self.BK, 64),
            )
        else:
            one_a, box_a = (
                cute.slice_(self._stage_layout(self.BM), (None, None, 0)),
                (self.BM, self.BK),
            )
        if self.b_trans:
            one_b, box_b = (
                cute.slice_(self._stage_layout_t(self.BN), (None, None, 0, 0)),
                (self.BK, 64),
            )
        else:
            one_b, box_b = (
                cute.slice_(self._stage_layout(self.BN), (None, None, 0)),
                (self.BN, self.BK),
            )
        atom_a, tA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mA, one_a, box_a
        )
        atom_b, tB = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mB, one_b, box_b
        )
        return atom_a, tA, atom_b, tB

    def _smem_tiles(self, storage):
        sbase = Int32(
            storage.a.get_tensor(
                cute.make_layout(self.stages * self.BM * self.BK)
            ).iterator.toint()
        )  # B follows A, as with `cp.async`
        la = (
            self._stage_layout_t(self.BM)
            if self.a_trans
            else self._stage_layout(self.BM)
        )
        lb = (
            self._stage_layout_t(self.BN)
            if self.b_trans
            else self._stage_layout(self.BN)
        )
        sA = storage.a.get_tensor(la.outer, swizzle=la.inner)
        sB = storage.b.get_tensor(lb.outer, swizzle=lb.inner)
        return sbase, sA, sB

    def _tma_parts(self, atom, sT, gT, trans, pid, rows):
        if const_expr(trans):
            g = cute.local_tile(gT, (self.BK, 64), (None, None))
        else:
            g = cute.local_tile(gT, (rows, self.BK), (pid, None))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sT, 0, 2),
            cute.group_modes(g, 0, 2),
        )

    def _tma_issue(self, pipe, atom, tsS, tgG, trans, pid, it, s, panels):
        if const_expr(trans):
            for p in range(panels):
                cute.copy(
                    atom,
                    tgG[(None, it, pid * panels + p)],
                    tsS[(None, p, s)],
                    tma_bar_ptr=pipe.full + s,
                )
        else:
            cute.copy(atom, tgG[(None, it)], tsS[(None, s)], tma_bar_ptr=pipe.full + s)

    def _tma_loader(self, pipe, parts, pid_m, pid_n):
        atom_a, tAsA, tAgA, atom_b, tBsB, tBgB = parts

        def load(c, it):
            pipe.arm(c, self.tx_bytes)
            s = c % self.stages
            self._tma_issue(
                pipe, atom_a, tAsA, tAgA, self.a_trans, pid_m, it, s, self.BM // 64
            )
            self._tma_issue(
                pipe, atom_b, tBsB, tBgB, self.b_trans, pid_n, it, s, self.BN // 64
            )

        return load

    def _pipeline(
        self, storage, mA, mB, m, n, k, m0, n0, pid_m, pid_n, nk, tidx, tma_args
    ):
        if const_expr(not self.tma):
            return super()._pipeline(
                storage, mA, mB, m, n, k, m0, n0, pid_m, pid_n, nk, tidx, tma_args
            )
        atom_a, tA, atom_b, tB = tma_args
        sbase, sA, sB = self._smem_tiles(storage)
        pipe = TmaPipeline(storage.mbar.data_ptr(), self.stages, nk, None, tidx < 32)
        pipe.init(self.NUM_THREADS // 32)
        tAsA, tAgA = self._tma_parts(atom_a, sA, tA, self.a_trans, pid_m, self.BM)
        tBsB, tBgB = self._tma_parts(atom_b, sB, tB, self.b_trans, pid_n, self.BN)
        load = self._tma_loader(
            pipe, (atom_a, tAsA, tAgA, atom_b, tBsB, tBgB), pid_m, pid_n
        )
        pipe.arm_and_load = lambda i: load(i, i)
        return sbase, pipe

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
        if const_expr(not self.warp_spec):
            return super()._launch(
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
        self.kernel_ws(
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
            self._tma_args(mA, mB),
        ).launch(
            grid=[ctas, 1, 1],
            block=[self.threads, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.kernel
    def kernel_ws(
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
        tma_args,
    ):
        alpha = self._alpha(mSA, mSB)
        sf16 = Int64(16) * Int64(k // 32)  # bytes between the A scales of rows 16 apart
        # warps 0-7 run the MMAs; warp 8 issues TMA and runs ahead across tiles
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        ncta, _, _ = cute.arch.grid_dim()
        warp = tidx // 32
        lane = tidx % 32
        npm = m // self.BM
        npn = n // self.BN
        total = npm * npn
        nk = k // self.BK

        storage = cutlass.memory.SmemAllocator().allocate(self.shared_storage)
        sbase, sA, sB = self._smem_tiles(storage)
        pipe = TmaPipeline(storage.mbar.data_ptr(), self.stages, nk, None, True)
        pipe.init(self.NUM_THREADS // 32)

        gws = Int64(mWs.iterator.toint())
        gflags = Int64(mFlags.iterator.toint())
        dp_end, per, s0, s1 = self._sk_split(total, nk, bid, ncta)
        n_dp = (dp_end - bid + ncta - 1) // ncta  # whole tiles of this CTA
        sk_last = (s1 - 1) // nk  # stream-K tiles run last to first
        n_seg = sk_last - s0 // nk + 1 if s1 > s0 else Int32(0)

        if warp >= self.NUM_THREADS // 32:
            if const_expr(self.setmaxnreg):
                cute.arch.setmaxregister_decrease(24)
            if warp == self.NUM_THREADS // 32:
                for tile in cutlass.range(bid, dp_end, ncta, unroll=1):
                    pid_m, pid_n = grouped_tile(tile, npm, npn, self.group_m)
                    self._produce(
                        pipe,
                        ((tile - bid) // ncta) * nk,
                        Int32(0),
                        nk,
                        pid_m,
                        pid_n,
                        sA,
                        sB,
                        tma_args,
                    )
                cbase = n_dp * nk
                for i in cutlass.range(n_seg, unroll=1):
                    sk = sk_last - i
                    lo = max(s0, sk * nk) - sk * nk
                    hi = min(s1, sk * nk + nk) - sk * nk
                    pid_m, pid_n = grouped_tile(dp_end + sk, npm, npn, self.group_m)
                    self._produce(
                        pipe, cbase, lo, hi - lo, pid_m, pid_n, sA, sB, tma_args
                    )
                    cbase = cbase + (hi - lo)
        else:
            if const_expr(self.setmaxnreg):
                cute.arch.setmaxregister_increase(240)
            warp_m = warp // 4
            warp_n = warp % 4
            g = lane // 4
            t = lane % 4
            acc = Accum(4, self.NT)
            fa = None
            fb = None
            if const_expr(self.frag_pipe):
                fa = cute.make_rmem_tensor(32, Int32)
                fb = cute.make_rmem_tensor(4 * self.NT, Int32)
            for tile in cutlass.range(bid, dp_end, ncta, unroll=1):
                pid_m, pid_n = grouped_tile(tile, npm, npn, self.group_m)
                j = (tile - bid) // ncta
                acc.fill(0.0)
                sfp = self._sfp(mASF, k, pid_m * self.BM, warp_m, g, t)
                self._mainloop(
                    pipe,
                    j * nk,
                    nk,
                    sbase,
                    warp_m,
                    warp_n,
                    lane,
                    acc,
                    fa,
                    fb,
                    sfp,
                    sf16,
                    Int32(0),
                )
                self._epilogue(
                    mC,
                    mPre,
                    mSA,
                    mSB,
                    mQ,
                    mQS,
                    n,
                    pid_m * self.BM,
                    pid_n * self.BN,
                    warp_m,
                    warp_n,
                    g,
                    t,
                    acc,
                    alpha,
                )
            cbase = n_dp * nk
            for i in cutlass.range(n_seg, unroll=1):
                sk = sk_last - i
                lo = max(s0, sk * nk) - sk * nk
                hi = min(s1, sk * nk + nk) - sk * nk
                pid_m, pid_n = grouped_tile(dp_end + sk, npm, npn, self.group_m)
                acc.fill(0.0)
                sfp = self._sfp(mASF, k, pid_m * self.BM, warp_m, g, t)
                self._mainloop(
                    pipe,
                    cbase,
                    hi - lo,
                    sbase,
                    warp_m,
                    warp_n,
                    lane,
                    acc,
                    fa,
                    fb,
                    sfp,
                    sf16,
                    lo,
                )
                cbase = cbase + (hi - lo)
                if hi < nk:
                    self._sk_publish(gws, gflags, bid, tidx, acc)
                else:
                    if lo > 0:  # earlier CTAs hold the first k-blocks
                        self._sk_gather(gws, gflags, (sk * nk) // per, bid, tidx, acc)
                    self._epilogue(
                        mC,
                        mPre,
                        mSA,
                        mSB,
                        mQ,
                        mQS,
                        n,
                        pid_m * self.BM,
                        pid_n * self.BN,
                        warp_m,
                        warp_n,
                        g,
                        t,
                        acc,
                        alpha,
                    )

    @cute.jit
    def _sk_split(self, total: Int32, nk: Int32, bid: Int32, ncta: Int32):
        rem = Int32(0)
        if const_expr(self.persistent):
            rem = total % ncta  # tiles after the last full round
        units = rem * nk
        per = max((units + ncta - 1) // ncta, (nk + 3) // 4)  # at least a quarter tile
        s0 = min(bid * per, units)
        return total - rem, per, s0, min(s0 + per, units)

    @cute.jit
    def _produce(
        self,
        pipe,
        cbase: Int32,
        k0: Int32,
        count: Int32,
        pid_m,
        pid_n,
        sA,
        sB,
        tma_args,
    ):
        atom_a, tA, atom_b, tB = tma_args
        tAsA, tAgA = self._tma_parts(atom_a, sA, tA, self.a_trans, pid_m, self.BM)
        tBsB, tBgB = self._tma_parts(atom_b, sB, tB, self.b_trans, pid_n, self.BN)
        load = self._tma_loader(
            pipe, (atom_a, tAsA, tAgA, atom_b, tBsB, tBgB), pid_m, pid_n
        )
        for it in cutlass.range(count, unroll=1):
            load(cbase + it, k0 + it)

    @cute.jit
    def _mainloop(
        self,
        pipe,
        cbase: Int32,
        count: Int32,
        sbase,
        warp_m,
        warp_n,
        lane,
        acc,
        fa,
        fb,
        sfp,
        sf16,
        k0: Int32,
    ):
        kk_n = self.KK
        if const_expr(self.frag_pipe):
            pipe.wait_full(cbase)
            self._frag_load(
                cbase % self.stages, 0, 0, sbase, warp_m, warp_n, lane, fa, fb
            )
            for it in cutlass.range(count, unroll=1):
                c = cbase + it
                for kk in cutlass.range_constexpr(kk_n):
                    if const_expr(kk == kk_n - 1):
                        pipe.release(c, lane)
                        if it + 1 < count:
                            pipe.wait_full(c + 1)
                            self._frag_load(
                                (c + 1) % self.stages,
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
                            c % self.stages,
                            kk + 1,
                            (kk + 1) % 2,
                            sbase,
                            warp_m,
                            warp_n,
                            lane,
                            fa,
                            fb,
                        )
                    self._mma_step(kk % 2, fa, fb, acc, sfp, sf16, k0 + it, kk)
        else:
            for it in cutlass.range(count, unroll=1):
                c = cbase + it
                pipe.wait_full(c)
                self._compute(
                    c % self.stages,
                    sbase,
                    warp_m,
                    warp_n,
                    lane,
                    acc,
                    sfp,
                    sf16,
                    k0 + it,
                )
                pipe.release(c, lane)

    @cute.jit
    def _sk_publish(self, gws: Int64, gflags: Int64, bid: Int32, tidx: Int32, acc):
        slot = gws + Int64(bid) * (self.NUM_THREADS * 16 * self.NT * 4)
        for e4 in cutlass.range_constexpr(4 * self.NT):
            st_global_v4_f32(
                slot + Int64(e4 * self.NUM_THREADS + tidx) * 16,
                acc[4 * e4],
                acc[4 * e4 + 1],
                acc[4 * e4 + 2],
                acc[4 * e4 + 3],
            )
        fence_gpu()
        cute.arch.barrier(barrier_id=1, number_of_threads=self.NUM_THREADS)
        if tidx == 0:
            st_release_gpu_b32(gflags + Int64(bid) * 4, Int32(1))

    @cute.jit
    def _sk_gather(
        self, gws: Int64, gflags: Int64, first: Int32, bid: Int32, tidx: Int32, acc
    ):
        for src in cutlass.range(first, bid, 1, unroll=1):
            if tidx == 0:
                wait_flag_acquire(gflags + Int64(src) * 4)
            cute.arch.barrier(barrier_id=1, number_of_threads=self.NUM_THREADS)
            slot = gws + Int64(src) * (self.NUM_THREADS * 16 * self.NT * 4)
            for e4 in cutlass.range_constexpr(4 * self.NT):
                v = ld_global_cg_v4_f32(slot + Int64(e4 * self.NUM_THREADS + tidx) * 16)
                for x in cutlass.range_constexpr(4):
                    acc[4 * e4 + x] = acc[4 * e4 + x] + v[x]
