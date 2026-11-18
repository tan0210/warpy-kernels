# type: ignore

import cutlass
from cutlass import Int32, const_expr, cute
from cutlass.cute.nvgpu import cpasync

from ..ptx import ldsm_x2_trans_b8, mma_16816, mma_mxf8, pack_e4m3x4
from ..utils.pipeline import TmaPipeline
from .constants import E4M3_ONE_X4
from .fwd_sm80 import AttnFwdSm80


class AttnFwdSm120(AttnFwdSm80):
    """Adds e4m3 inputs (block-scaled ``mma.sync``), K/V loads by TMA and a producer warp group, for SM120."""

    FP8 = True
    BF16_ROWSUM_MMA = True
    PRODUCER_REGS = 24  # 8 * 240 + 4 * 24 fills 12 * 168
    CONSUMER_REGS = 240

    def __init__(
        self,
        head_dim: int,
        causal: bool,
        num_stages: int = 4,
        pipelined: bool = True,
        block_n: int = 64,
        tile_local: bool = True,
        lpt: bool = False,
        tma: bool = False,
        return_lse: bool = True,
        has_seqused: bool = False,
        lse_natural: bool = False,
        warp_spec: bool = False,
        bf16: bool = False,
        arch: int = 120,
    ):
        assert not (bf16 and tma), "bf16 inputs load with cp.async"
        assert tma or not warp_spec
        self.tma = tma
        self.warp_spec = warp_spec
        super().__init__(
            head_dim,
            causal,
            num_stages,
            pipelined,
            block_n,
            tile_local,
            lpt,
            return_lse,
            has_seqused,
            lse_natural,
            bf16,
            arch,
        )
        self.threads = self.NUM_THREADS + (128 if warp_spec else 0)

    def _smem_overhead(self):
        return 1024 if self.tma else 0

    def _make_storage(self):
        if not self.tma:
            return super()._make_storage()
        n_stage, tile_b, q_b = self.stages, self.kv_tile_bytes, self.q_bytes

        @cute.struct
        class SharedStorage:
            # full barriers, then empty barriers
            mbar: cute.struct.MemRange[cutlass.Int64, 2 * n_stage]
            # Q, K, V stages, 1024-byte aligned for the swizzle
            q: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, q_b], 1024]
            k: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, n_stage * tile_b], 1024
            ]
            v: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, n_stage * tile_b], 1024
            ]

        return SharedStorage

    def _mma(self, a, b, c):
        if self.bf16:
            return mma_16816(a, b, c, "bf16")
        return mma_mxf8(a, b, c)

    @cute.jit
    def _pv(self, jt, sbase, v_key, acc, o_acc, l_acc):
        if const_expr(self.bf16):
            self._pv_bf16(self._v_tile(sbase, jt), acc, o_acc, l_acc)
        else:
            self._pv_e4m3(self._v_tile(sbase, jt), v_key, acc, o_acc, l_acc)

    @cute.jit
    def _pv_e4m3(self, vtile, v_key, acc, o_acc, l_acc):
        ones = Int32(E4M3_ONE_X4)
        for kk2 in cutlass.range_constexpr(self.BN // 32):
            t0 = 4 * kk2
            # P packed from the S accumulators
            pa = [
                pack_e4m3x4(
                    acc[4 * t0], acc[4 * t0 + 1], acc[4 * t0 + 4], acc[4 * t0 + 5]
                ),
                pack_e4m3x4(
                    acc[4 * t0 + 2], acc[4 * t0 + 3], acc[4 * t0 + 6], acc[4 * t0 + 7]
                ),
                pack_e4m3x4(
                    acc[4 * t0 + 8], acc[4 * t0 + 9], acc[4 * t0 + 12], acc[4 * t0 + 13]
                ),
                pack_e4m3x4(
                    acc[4 * t0 + 10],
                    acc[4 * t0 + 11],
                    acc[4 * t0 + 14],
                    acc[4 * t0 + 15],
                ),
            ]
            c = mma_mxf8(pa, [ones, ones], [l_acc[0], l_acc[1], l_acc[2], l_acc[3]])
            for x in cutlass.range_constexpr(4):
                l_acc[x] = c[x]
            key = v_key + 32 * kk2
            for npair in cutlass.range_constexpr(self.d // 16):
                # V via transposing 8-bit `ldmatrix`
                vb = ldsm_x2_trans_b8(vtile.addr(key, npair))
                for h in cutlass.range_constexpr(2):
                    nt = 2 * npair + h
                    c = mma_mxf8(
                        pa,
                        [vb[h], vb[2 + h]],
                        [
                            o_acc[4 * nt],
                            o_acc[4 * nt + 1],
                            o_acc[4 * nt + 2],
                            o_acc[4 * nt + 3],
                        ],
                    )
                    for x in cutlass.range_constexpr(4):
                        o_acc[4 * nt + x] = c[x]

    @cute.jit
    def _stage_layout(self):
        sw = cute.make_swizzle(self.sw_bits, 4, 3)  # 128B for d=128, 64B for d=64
        return cute.make_composed_layout(
            sw,
            0,
            cute.make_layout(
                (self.BN, self.d, self.stages), stride=(self.d, 1, self.kv_tile_bytes)
            ),
        )

    def _tma_args(self, mK, mV):
        if not self.tma:
            return None
        one_stage = cute.slice_(self._stage_layout(), (None, None, 0))
        # view [B, H, N, d] as (N, d, H, B)
        mK_t = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[2, 3, 1, 0]))
        mV_t = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[2, 3, 1, 0]))
        atom_k, tk = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mK_t, one_stage, (self.BN, self.d)
        )
        atom_v, tv = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mV_t, one_stage, (self.BN, self.d)
        )
        return atom_k, tk, atom_v, tv

    def _setup(self, storage, tidx, hkv, b, tma_args):
        if const_expr(not self.tma):
            return super()._setup(storage, tidx, hkv, b, tma_args)
        atom_k, tk, atom_v, tv = tma_args
        # Q, K, V stages are consecutive, matching the `cp.async` offsets
        sbase = Int32(
            storage.q.get_tensor(cute.make_layout(self.q_bytes)).iterator.toint()
        )
        stage_layout = self._stage_layout()
        sK = storage.k.get_tensor(stage_layout.outer, swizzle=stage_layout.inner)
        sV = storage.v.get_tensor(stage_layout.outer, swizzle=stage_layout.inner)
        self._pipe = TmaPipeline(storage.mbar.data_ptr(), self.stages, None, None, True)
        self._pipe.init(self.NUM_THREADS // 32)
        self._atom_k = atom_k
        self._atom_v = atom_v
        gKt = cute.local_tile(tk[None, None, hkv, b], (self.BN, self.d), (None, 0))
        gVt = cute.local_tile(tv[None, None, hkv, b], (self.BN, self.d), (None, 0))
        self._tKsK, self._tKgK = cpasync.tma_partition(
            atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(gKt, 0, 2),
        )
        self._tVsV, self._tVgV = cpasync.tma_partition(
            atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(gVt, 0, 2),
        )
        return sbase

    @cute.jit
    def _tma_load(self, jn: Int32, n_blocks: Int32):
        if jn < n_blocks:
            s = jn % self.stages
            self._pipe.arm(jn, 2 * self.kv_tile_bytes)
            cute.copy(
                self._atom_k,
                self._tKgK[(None, jn)],
                self._tKsK[(None, s)],
                tma_bar_ptr=self._pipe.full + s,
            )
            cute.copy(
                self._atom_v,
                self._tVgV[(None, jn)],
                self._tVsV[(None, s)],
                tma_bar_ptr=self._pipe.full + s,
            )

    @cute.jit
    def _prologue(self, n_blocks, sbase, gk, gv, tidx):
        if const_expr(self.tma):
            if const_expr(not self.warp_spec):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                if tidx < 32:
                    for s in cutlass.range_constexpr(self.stages - 1):
                        self._tma_load(Int32(s), n_blocks)
            cute.arch.cp_async_wait_group(0)
        else:
            AttnFwdSm80._prologue(self, n_blocks, sbase, gk, gv, tidx)

    @cute.jit
    def _first_wait(self, n_blocks):
        if const_expr(self.tma):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
            if n_blocks > 0:
                self._pipe.wait_full(Int32(0))

    @cute.jit
    def _tile_wait(self, j, n_blocks, sbase, gk, gv, tidx):
        if const_expr(self.tma):
            if const_expr(not self.warp_spec):  # noqa: SIM102  # the DSL needs the const_expr and runtime ifs kept separate
                if tidx < 32:
                    self._tma_load(j + (self.stages - 1), n_blocks)
            if const_expr(self.pipelined):
                if j + 1 < n_blocks:  # tile j+1 must be loaded
                    self._pipe.wait_full(j + 1)
            else:
                self._pipe.wait_full(j)
        else:
            AttnFwdSm80._tile_wait(self, j, n_blocks, sbase, gk, gv, tidx)

    @cute.jit
    def _tile_release(self, j, lane):
        if const_expr(self.tma):
            self._pipe.release(j, lane)

    def _min_blocks(self):
        return 1 if self.warp_spec else 0

    @cute.jit
    def _producer(self, warp, n_blocks):
        if const_expr(self.warp_spec):
            cute.arch.setmaxregister_decrease(self.PRODUCER_REGS)
            if warp == self.NUM_THREADS // 32:
                for j in cutlass.range(n_blocks, unroll=1):
                    self._tma_load(j, n_blocks)

    def _consumer_start(self):
        if self.warp_spec:
            cute.arch.setmaxregister_increase(self.CONSUMER_REGS)
