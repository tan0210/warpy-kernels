# type: ignore

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Int32, Int64, cute
from cutlass.base_dsl.ast_helpers import const_expr
from cutlass.cute.nvgpu import cpasync

from ..sm120.ptx import (
    SMEM_LIMIT,
    cp_async16,
    ld_global_b32,
    ldsm_x4,
    ldsm_x4_trans,
    mma_16816,
    mma_fp8,
    pack_half2,
    st_global_b32,
    swizzle_offset,
    unpack_half2,
    warp_sync,
)


class GemmSm120:
    BM = 128
    NUM_THREADS = 256

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
    ):
        self.accum = accum
        assert not ((a_trans or b_trans) and fp8), (
            "transposed operands need 16-bit inputs"
        )
        self.a_trans = a_trans
        self.b_trans = b_trans
        assert act in ("", "gelu", "dgelu") and (act == "gelu" or not store_pre), (
            "store_pre needs act='gelu'"
        )
        self.act = act
        self.store_pre = store_pre
        self.fp8 = fp8
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
        assert tma or not warp_spec, "warp_spec needs tma"
        assert warp_spec or not persistent, "persistent needs warp_spec"
        self.tma = tma
        self.warp_spec = warp_spec
        self.persistent = persistent
        assert warp_spec or not setmaxnreg, "setmaxnreg needs warp_spec"
        self.setmaxnreg = setmaxnreg
        # one producer warp, or a whole producer warpgroup so its registers can be handed to the MMA warps
        self.threads = self.NUM_THREADS + (
            128 if setmaxnreg else 32 if warp_spec else 0
        )
        self.BN = bn
        self.WN = bn // 4
        self.NT = self.WN // 8
        self.BK = bk
        self.row_bytes = bk * self.es
        self.KK = (
            self.row_bytes // 32
        )  # MMA k steps per tile: k16 for 16-bit inputs, k32 for fp8
        self.sw_bits = 3 if self.row_bytes == 128 else 2  # 128B or 64B swizzle
        self.stages = stages
        self.group_m = group_m
        self.min_blocks = min_blocks
        self.a_bytes = self.BM * bk * self.es
        self.b_bytes = bn * bk * self.es
        self.tx_bytes = self.a_bytes + self.b_bytes
        self.smem_bytes = stages * (self.a_bytes + self.b_bytes)
        assert (self.smem_bytes + (1024 if tma else 0)) * min_blocks <= SMEM_LIMIT, (
            f"shared memory does not fit: {self.smem_bytes} B x {min_blocks} CTA(s)"
        )
        nbytes = self.smem_bytes
        if tma:
            edt = (
                cutlass.Uint8
                if fp8
                else cutlass.BFloat16
                if ty == "bf16"
                else cutlass.Float16
            )
            n_stage, a_elems, b_elems = stages, stages * self.BM * bk, stages * bn * bk

            @cute.struct
            class SharedStorage:
                # full barriers, then empty barriers
                mbar: cute.struct.MemRange[cutlass.Int64, 2 * n_stage]
                # A then B stages, 1024-byte aligned for the swizzle
                a: cute.struct.Align[cute.struct.MemRange[edt, a_elems], 1024]
                b: cute.struct.Align[cute.struct.MemRange[edt, b_elems], 1024]
        else:

            @cute.struct
            class SharedStorage:
                buf: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Uint8, nbytes], 1024
                ]

        self.shared_storage = SharedStorage

    def _swz(self, row, chunk):
        return swizzle_offset(self.row_bytes, row, chunk)

    @cute.jit
    def _stage_layout(self, rows: cutlass.Constexpr):
        sw = cute.make_swizzle(
            self.sw_bits, 4, 3
        )  # TMA reads the swizzle in byte units: 128B is <3,4,3>, 64B is <2,4,3>
        return cute.make_composed_layout(
            sw,
            0,
            cute.make_layout(
                (rows, self.BK, self.stages), stride=(self.BK, 1, rows * self.BK)
            ),
        )

    def _load_trans(self, s_tile, g, kt, stride, width, tidx):
        # a `[BK, width]` tile with `width` contiguous, stored as 64-column panels of 128B-swizzled rows
        chunks = width // 8
        for i in range(self.BK * chunks // self.NUM_THREADS):
            idx = tidx + i * self.NUM_THREADS
            row = idx // chunks
            ch = idx % chunks
            cp_async16(
                s_tile + (ch // 8) * (self.BK * 128) + swizzle_offset(128, row, ch % 8),
                g + (Int64(kt * self.BK + row) * Int64(stride) + Int64(ch * 8)) * 2,
            )

    def _a_frag(self, sa, warp_m, mi, kk, lane):
        if const_expr(self.a_trans):
            j = lane // 8
            c = (warp_m * 64 + mi * 16 + (j % 2) * 8) // 8
            return ldsm_x4_trans(
                sa
                + (c // 8) * (self.BK * 128)
                + swizzle_offset(128, kk * 16 + (j // 2) * 8 + lane % 8, c % 8)
            )
        arow = warp_m * 64 + mi * 16 + (lane % 8) + ((lane // 8) % 2) * 8
        return ldsm_x4(sa + self._swz(arow, lane // 16 + 2 * kk))

    def _b_frag(self, sb, warp_n, nj, kk, lane):
        if const_expr(self.b_trans):
            j = lane // 8
            c = (warp_n * self.WN + nj * 16 + (j // 2) * 8) // 8
            return ldsm_x4_trans(
                sb
                + (c // 8) * (self.BK * 128)
                + swizzle_offset(128, kk * 16 + (j % 2) * 8 + lane % 8, c % 8)
            )
        brow = warp_n * self.WN + nj * 16 + (lane % 8) + (lane // 16) * 8
        return ldsm_x4(sb + self._swz(brow, (lane // 8) % 2 + 2 * kk))

    def _load(self, kt, st, sbase, ga, gb, k, m, n, tidx):
        chunks = self.row_bytes // 16
        per = 16 // self.es  # elements per 16-byte chunk
        sa = sbase + st * self.a_bytes
        sb = sbase + self.stages * self.a_bytes + st * self.b_bytes
        if const_expr(self.a_trans):
            self._load_trans(sa, ga, kt, m, self.BM, tidx)
        else:
            for i in range(self.BM * chunks // self.NUM_THREADS):
                idx = tidx + i * self.NUM_THREADS
                row = idx // chunks
                ch = idx % chunks
                cp_async16(
                    sa + self._swz(row, ch),
                    ga
                    + (Int64(row) * Int64(k) + Int64(kt * self.BK + ch * per))
                    * self.es,
                )
        if const_expr(self.b_trans):
            self._load_trans(sb, gb, kt, n, self.BN, tidx)
        else:
            for i in range(self.BN * chunks // self.NUM_THREADS):
                idx = tidx + i * self.NUM_THREADS
                row = idx // chunks
                ch = idx % chunks
                cp_async16(
                    sb + self._swz(row, ch),
                    gb
                    + (Int64(row) * Int64(k) + Int64(kt * self.BK + ch * per))
                    * self.es,
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

    def _tma_issue(self, atom, tsS, tgG, trans, pid, it, s, panels):
        if const_expr(trans):
            for p in range(panels):
                cute.copy(
                    atom,
                    tgG[(None, it, pid * panels + p)],
                    tsS[(None, p, s)],
                    tma_bar_ptr=self._full + s,
                )
        else:
            cute.copy(atom, tgG[(None, it)], tsS[(None, s)], tma_bar_ptr=self._full + s)

    @cute.jit
    def _tma_load(self, jn: Int32, nk: Int32):
        if jn < nk:
            s = jn % self.stages
            lane0 = (cute.arch.thread_idx()[0] % 32) == 0
            if lane0:  # lane 0 waits for the stage and arms the barrier
                cute.arch.mbarrier_wait(self._empty + s, ((jn // self.stages) & 1) ^ 1)
                cute.arch.mbarrier_arrive_and_expect_tx(self._full + s, self.tx_bytes)
            warp_sync()  # the TMA copy runs `elect.sync` on the full warp, so reconverge first
            self._tma_issue(
                self._atom_a,
                self._tAsA,
                self._tAgA,
                self.a_trans,
                self._pid_m,
                jn,
                s,
                self.BM // 64,
            )
            self._tma_issue(
                self._atom_b,
                self._tBsB,
                self._tBgB,
                self.b_trans,
                self._pid_n,
                jn,
                s,
                self.BN // 64,
            )

    @cute.jit
    def _tile_coords(self, tile: Int32, npm: Int32, npn: Int32):
        gm = Int32(self.group_m)
        per_group = gm * npn
        first_m = (tile // per_group) * gm
        rem = npm - first_m
        size = min(gm, rem)
        return first_m + (tile % per_group) % size, (tile % per_group) // size

    @cute.jit
    def _tma_load_ws(
        self, c: Int32, it: Int32, pid_m: Int32, pid_n: Int32, tAsA, tAgA, tBsB, tBgB
    ):
        s = c % self.stages
        if (
            cute.arch.thread_idx()[0] % 32
        ) == 0:  # lane 0 waits for the stage and arms the barrier
            cute.arch.mbarrier_wait(self._empty + s, ((c // self.stages) & 1) ^ 1)
            cute.arch.mbarrier_arrive_and_expect_tx(self._full + s, self.tx_bytes)
        warp_sync()  # the TMA copy runs `elect.sync` on the full warp, so reconverge first
        self._tma_issue(
            self._atom_a, tAsA, tAgA, self.a_trans, pid_m, it, s, self.BM // 64
        )
        self._tma_issue(
            self._atom_b, tBsB, tBgB, self.b_trans, pid_n, it, s, self.BN // 64
        )

    @cute.jit
    def _tma_release(self, it: Int32, lane: Int32):
        warp_sync()
        if lane == 0:
            cute.arch.mbarrier_arrive(
                self._empty + it % self.stages
            )  # this warp is done reading the stage

    @cute.jit
    def _mma(self, a, b, c):
        if const_expr(self.fp8):
            return mma_fp8(a, b, c)
        else:
            return mma_16816(a, b, c, self.ty)

    @cute.jit
    def _frag_load(self, st, kk, buf, sbase, warp_m, warp_n, lane, fa, fb):
        sa = sbase + st * self.a_bytes
        sb = sbase + self.stages * self.a_bytes + st * self.b_bytes
        for mi in cutlass.range_constexpr(4):
            r = self._a_frag(sa, warp_m, mi, kk, lane)
            for x in cutlass.range_constexpr(4):
                fa[buf * 16 + mi * 4 + x] = r[x]
        for nj in cutlass.range_constexpr(self.NT // 2):
            r = self._b_frag(sb, warp_n, nj, kk, lane)
            for x in cutlass.range_constexpr(4):
                fb[buf * 2 * self.NT + nj * 4 + x] = r[x]

    @cute.jit
    def _mma_step(self, buf, fa, fb, acc):
        for nj in cutlass.range_constexpr(self.NT // 2):
            for h in cutlass.range_constexpr(2):
                nt = 2 * nj + h
                b = [
                    fb[buf * 2 * self.NT + nj * 4 + 2 * h],
                    fb[buf * 2 * self.NT + nj * 4 + 2 * h + 1],
                ]
                for mi in cutlass.range_constexpr(4):
                    o = (mi * self.NT + nt) * 4
                    ao = buf * 16 + mi * 4
                    a = [fa[ao], fa[ao + 1], fa[ao + 2], fa[ao + 3]]
                    c = self._mma(a, b, [acc[o], acc[o + 1], acc[o + 2], acc[o + 3]])
                    for x in cutlass.range_constexpr(4):
                        acc[o + x] = c[x]

    @cute.jit
    def _compute(self, st, sbase, warp_m, warp_n, lane, acc):
        sa = sbase + st * self.a_bytes
        sb = sbase + self.stages * self.a_bytes + st * self.b_bytes
        for kk in cutlass.range_constexpr(self.KK):
            a = []
            for mi in cutlass.range_constexpr(4):
                a.append(self._a_frag(sa, warp_m, mi, kk, lane))
            for nj in cutlass.range_constexpr(self.NT // 2):
                b = self._b_frag(sb, warp_n, nj, kk, lane)
                for h in cutlass.range_constexpr(2):
                    nt = 2 * nj + h
                    for mi in cutlass.range_constexpr(4):
                        o = (mi * self.NT + nt) * 4
                        c = self._mma(
                            a[mi],
                            [b[2 * h], b[2 * h + 1]],
                            [acc[o], acc[o + 1], acc[o + 2], acc[o + 3]],
                        )
                        for x in cutlass.range_constexpr(4):
                            acc[o + x] = c[x]

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPre: cute.Tensor,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
        m: Int32,
        n: Int32,
        k: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        if const_expr(self.tma):
            if const_expr(self.a_trans):
                one_a = cute.slice_(self._stage_layout_t(self.BM), (None, None, 0, 0))
                box_a = (self.BK, 64)
            else:
                one_a = cute.slice_(self._stage_layout(self.BM), (None, None, 0))
                box_a = (self.BM, self.BK)
            if const_expr(self.b_trans):
                one_b = cute.slice_(self._stage_layout_t(self.BN), (None, None, 0, 0))
                box_b = (self.BK, 64)
            else:
                one_b = cute.slice_(self._stage_layout(self.BN), (None, None, 0))
                box_b = (self.BN, self.BK)
            atom_a, tA = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mA, one_a, box_a
            )
            atom_b, tB = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), mB, one_b, box_b
            )
        else:
            atom_a = None
            tA = None
            atom_b = None
            tB = None
        if const_expr(self.warp_spec):
            self.kernel_ws(
                mA, mB, mC, mPre, mSA, mSB, m, n, k, atom_a, tA, atom_b, tB
            ).launch(
                grid=[ctas, 1, 1],
                block=[self.threads, 1, 1],
                smem=self.shared_storage.size_in_bytes(),
                min_blocks_per_mp=self.min_blocks,
                stream=stream,
            )
        else:
            self.kernel(
                mA, mB, mC, mPre, mSA, mSB, m, n, k, atom_a, tA, atom_b, tB
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
        m: Int32,
        n: Int32,
        k: Int32,
        atom_a,
        tA,
        atom_b,
        tB,
    ):
        alpha = Float32(1.0)
        if const_expr(self.fp8):
            alpha = Float32(mSA[0]) * Float32(mSB[0])
        tidx, _, _ = cute.arch.thread_idx()
        pid, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        warp_m = warp // 4
        warp_n = warp % 4
        g = lane // 4
        t = lane % 4

        # grouped raster: `group_m` row blocks sweep all column blocks before the next group
        gm = Int32(self.group_m)
        npm = m // self.BM
        npn = n // self.BN
        per_group = gm * npn
        first_m = (pid // per_group) * gm
        rem = npm - first_m
        size = min(gm, rem)
        pid_m = first_m + (pid % per_group) % size
        pid_n = (pid % per_group) // size
        m0 = pid_m * self.BM
        n0 = pid_n * self.BN

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        nk = k // self.BK
        gc = Int64(mC.iterator.toint())
        gp = Int64(mPre.iterator.toint())
        if const_expr(self.tma):
            # A and B stages are consecutive, matching the `cp.async` offsets
            sbase = Int32(
                storage.a.get_tensor(
                    cute.make_layout(self.stages * self.BM * self.BK)
                ).iterator.toint()
            )
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
            self._full = storage.mbar.data_ptr()
            self._empty = self._full + self.stages
            self._atom_a = atom_a
            self._atom_b = atom_b
            self._pid_m = pid_m
            self._pid_n = pid_n
            if tidx == 0:
                for s in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(self._full + s, 1)
                    cute.arch.mbarrier_init(self._empty + s, self.NUM_THREADS // 32)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            self._tAsA, self._tAgA = self._tma_parts(
                atom_a, sA, tA, self.a_trans, pid_m, self.BM
            )
            self._tBsB, self._tBgB = self._tma_parts(
                atom_b, sB, tB, self.b_trans, pid_n, self.BN
            )
            ga = Int64(0)
            gb = Int64(0)
            if tidx < 32:
                for s in cutlass.range_constexpr(self.stages - 1):
                    self._tma_load(Int32(s), nk)
        else:
            sbase = Int32(
                storage.buf.get_tensor(
                    cute.make_layout(self.smem_bytes)
                ).iterator.toint()
            )
            ga = Int64(mA.iterator.toint()) + Int64(m0) * Int64(k) * self.es
            gb = Int64(mB.iterator.toint()) + Int64(n0) * Int64(k) * self.es
            if const_expr(self.a_trans):
                ga = Int64(mA.iterator.toint()) + Int64(m0) * self.es
            if const_expr(self.b_trans):
                gb = Int64(mB.iterator.toint()) + Int64(n0) * self.es
            for s in cutlass.range_constexpr(self.stages - 1):
                if s < nk:
                    self._load(Int32(s), s, sbase, ga, gb, k, m, n, tidx)
                cute.arch.cp_async_commit_group()

        acc = cute.make_rmem_tensor(16 * self.NT, Float32)
        acc.fill(0.0)
        if const_expr(self.frag_pipe):
            fa = cute.make_rmem_tensor(32, Int32)
            fb = cute.make_rmem_tensor(4 * self.NT, Int32)
            if const_expr(self.tma):
                cute.arch.mbarrier_wait(self._full, 0)
            else:
                cute.arch.cp_async_wait_group(self.stages - 2)
                cute.arch.sync_threads()
            self._frag_load(Int32(0), 0, 0, sbase, warp_m, warp_n, lane, fa, fb)
            kk_n = self.KK
            # TMA issues late in the tile so the slowest warp has released the stage being refilled
            issue_kk = kk_n // 2 if self.tma else 0
            for it in cutlass.range(nk, unroll=1):
                st = it % self.stages
                for kk in cutlass.range_constexpr(kk_n):
                    if const_expr(kk == issue_kk):
                        nxt = it + (self.stages - 1)
                        if const_expr(self.tma):
                            if tidx < 32:
                                self._tma_load(nxt, nk)
                        else:
                            if nxt < nk:
                                self._load(
                                    nxt, nxt % self.stages, sbase, ga, gb, k, m, n, tidx
                                )
                            cute.arch.cp_async_commit_group()
                    if const_expr(kk == kk_n - 1):
                        # the next stage must have landed and every warp be done with the stage being refilled
                        if const_expr(self.tma):
                            self._tma_release(it, lane)
                            if it + 1 < nk:
                                cute.arch.mbarrier_wait(
                                    self._full + (it + 1) % self.stages,
                                    ((it + 1) // self.stages) & 1,
                                )
                        else:
                            cute.arch.cp_async_wait_group(self.stages - 2)
                            cute.arch.sync_threads()
                        # the last tile reads a stale, unused stage
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
                    self._mma_step(kk % 2, fa, fb, acc)
        else:
            if const_expr(self.tma):
                for it in cutlass.range(nk, unroll=1):
                    if tidx < 32:
                        self._tma_load(it + (self.stages - 1), nk)
                    cute.arch.mbarrier_wait(
                        self._full + it % self.stages, (it // self.stages) & 1
                    )
                    self._compute(it % self.stages, sbase, warp_m, warp_n, lane, acc)
                    self._tma_release(it, lane)
            else:
                for it in cutlass.range(nk, unroll=1):
                    cute.arch.cp_async_wait_group(self.stages - 2)
                    cute.arch.sync_threads()
                    nxt = it + (self.stages - 1)
                    if nxt < nk:
                        self._load(nxt, nxt % self.stages, sbase, ga, gb, k, m, n, tidx)
                    cute.arch.cp_async_commit_group()
                    self._compute(it % self.stages, sbase, warp_m, warp_n, lane, acc)

        self._epilogue(gc, gp, n, m0, n0, warp_m, warp_n, g, t, acc, alpha)

    @cute.jit
    def _gelu(self, x):
        return (
            Float32(0.5)
            * x
            * (Float32(1.0) + cute.math.erf(x * Float32(0.7071067811865476)))
        )

    @cute.jit
    def _dgelu(self, x):
        cdf = Float32(0.5) * (
            Float32(1.0) + cute.math.erf(x * Float32(0.7071067811865476))
        )
        pdf = Float32(0.3989422804014327) * cute.math.exp2(
            x * x * Float32(-0.7213475204444817), fastmath=True
        )
        return cdf + x * pdf

    @cute.jit
    def _epilogue(self, gc, gp, n, m0, n0, warp_m, warp_n, g, t, acc, alpha):
        for mi in cutlass.range_constexpr(4):
            for nt in cutlass.range_constexpr(self.NT):
                col = n0 + warp_n * self.WN + nt * 8 + 2 * t
                for r in cutlass.range_constexpr(2):
                    row = m0 + warp_m * 64 + mi * 16 + g + 8 * r
                    o = (mi * self.NT + nt) * 4 + 2 * r
                    off = (Int64(row) * Int64(n) + Int64(col)) * 2
                    v0 = acc[o] * alpha
                    v1 = acc[o + 1] * alpha
                    if const_expr(self.accum):
                        # the output buffer holds an addend, which joins before the activation
                        c0, c1 = unpack_half2(ld_global_b32(gc + off), self.ty)
                        v0 = v0 + c0
                        v1 = v1 + c1
                    if const_expr(self.act == "gelu"):
                        if const_expr(self.store_pre):
                            st_global_b32(gp + off, pack_half2(v0, v1, self.ty))
                        v0 = self._gelu(v0)
                        v1 = self._gelu(v1)
                    if const_expr(self.act == "dgelu"):
                        p0, p1 = unpack_half2(ld_global_b32(gp + off), self.ty)
                        v0 = v0 * self._dgelu(p0)
                        v1 = v1 * self._dgelu(p1)
                    st_global_b32(gc + off, pack_half2(v0, v1, self.ty))

    @cute.kernel
    def kernel_ws(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mPre: cute.Tensor,
        mSA: cute.Tensor,
        mSB: cute.Tensor,
        m: Int32,
        n: Int32,
        k: Int32,
        atom_a,
        tA,
        atom_b,
        tB,
    ):
        alpha = Float32(1.0)
        if const_expr(self.fp8):
            alpha = Float32(mSA[0]) * Float32(mSB[0])
        # warps 0-7 run the MMAs; warp 8 only issues TMA, so it runs ahead across tile boundaries
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        ncta, _, _ = cute.arch.grid_dim()
        warp = tidx // 32
        lane = tidx % 32
        npm = m // self.BM
        npn = n // self.BN
        total = npm * npn
        nk = k // self.BK

        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        sbase = Int32(
            storage.a.get_tensor(
                cute.make_layout(self.stages * self.BM * self.BK)
            ).iterator.toint()
        )
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
        self._full = storage.mbar.data_ptr()
        self._empty = self._full + self.stages
        self._atom_a = atom_a
        self._atom_b = atom_b
        if tidx == 0:
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(self._full + s, 1)
                cute.arch.mbarrier_init(self._empty + s, self.NUM_THREADS // 32)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        if warp >= self.NUM_THREADS // 32:
            if const_expr(self.setmaxnreg):
                cute.arch.setmaxregister_decrease(24)
            if warp == self.NUM_THREADS // 32:
                for tile in cutlass.range(bid, total, ncta, unroll=1):
                    pid_m, pid_n = self._tile_coords(tile, npm, npn)
                    base = ((tile - bid) // ncta) * nk
                    tAsA, tAgA = self._tma_parts(
                        atom_a, sA, tA, self.a_trans, pid_m, self.BM
                    )
                    tBsB, tBgB = self._tma_parts(
                        atom_b, sB, tB, self.b_trans, pid_n, self.BN
                    )
                    for it in cutlass.range(nk, unroll=1):
                        self._tma_load_ws(
                            base + it, it, pid_m, pid_n, tAsA, tAgA, tBsB, tBgB
                        )
        else:
            if const_expr(self.setmaxnreg):
                cute.arch.setmaxregister_increase(240)
            warp_m = warp // 4
            warp_n = warp % 4
            g = lane // 4
            t = lane % 4
            gc = Int64(mC.iterator.toint())
            gp = Int64(mPre.iterator.toint())
            acc = cute.make_rmem_tensor(16 * self.NT, Float32)
            if const_expr(self.frag_pipe):
                fa = cute.make_rmem_tensor(32, Int32)
                fb = cute.make_rmem_tensor(4 * self.NT, Int32)
            kk_n = self.KK
            for tile in cutlass.range(bid, total, ncta, unroll=1):
                pid_m, pid_n = self._tile_coords(tile, npm, npn)
                base = ((tile - bid) // ncta) * nk
                acc.fill(0.0)
                if const_expr(self.frag_pipe):
                    cute.arch.mbarrier_wait(
                        self._full + base % self.stages, (base // self.stages) & 1
                    )
                    self._frag_load(
                        base % self.stages, 0, 0, sbase, warp_m, warp_n, lane, fa, fb
                    )
                    for it in cutlass.range(nk, unroll=1):
                        c = base + it
                        for kk in cutlass.range_constexpr(kk_n):
                            if const_expr(kk == kk_n - 1):
                                self._tma_release(c, lane)
                                if it + 1 < nk:
                                    cute.arch.mbarrier_wait(
                                        self._full + (c + 1) % self.stages,
                                        ((c + 1) // self.stages) & 1,
                                    )
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
                            self._mma_step(kk % 2, fa, fb, acc)
                else:
                    for it in cutlass.range(nk, unroll=1):
                        c = base + it
                        cute.arch.mbarrier_wait(
                            self._full + c % self.stages, (c // self.stages) & 1
                        )
                        self._compute(c % self.stages, sbase, warp_m, warp_n, lane, acc)
                        self._tma_release(c, lane)
                self._epilogue(
                    gc,
                    gp,
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
