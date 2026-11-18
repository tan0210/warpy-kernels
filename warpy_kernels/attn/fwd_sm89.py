# type: ignore

import cutlass
from cutlass import Float32, Int32, Int64, const_expr, cute

from ..ptx import (
    ldsm_x4_trans,
    mma_16816,
    mma_fp8,
    pack_e4m3x4,
    pack_half2,
    prmt,
    st_global_v2_b32,
)
from .constants import E4M3_ONE_X4
from .fwd_sm80 import AttnFwdSm80


class AttnFwdSm89(AttnFwdSm80):
    """Adds e4m3 inputs for SM89 with ``mma.sync`` m16n8k32, adding each tile's P V into fp32."""

    FP8 = True

    def _mma(self, a, b, c):
        if self.bf16:
            return mma_16816(a, b, c, "bf16")
        return mma_fp8(a, b, c)

    @cute.jit
    def _pv(self, jt, sbase, v_key, acc, o_acc, l_acc):
        if const_expr(self.bf16):
            self._pv_bf16(self._v_tile(sbase, jt), acc, o_acc, l_acc)
        else:
            self._pv_e4m3(self._v_tile(sbase, jt), acc, o_acc, l_acc)

    @cute.jit
    def _pv_e4m3(self, vtile, acc, o_acc, l_acc):
        lane = cute.arch.thread_idx()[0] % 32
        ones = Int32(E4M3_ONE_X4)
        nk = self.BN // 32
        # P of 32 keys as the A operand, keys in the order the V fragments below use
        pa = [
            [
                pack_e4m3x4(
                    acc[16 * kk2],
                    acc[16 * kk2 + 1],
                    acc[16 * kk2 + 4],
                    acc[16 * kk2 + 5],
                ),
                pack_e4m3x4(
                    acc[16 * kk2 + 2],
                    acc[16 * kk2 + 3],
                    acc[16 * kk2 + 6],
                    acc[16 * kk2 + 7],
                ),
                pack_e4m3x4(
                    acc[16 * kk2 + 8],
                    acc[16 * kk2 + 9],
                    acc[16 * kk2 + 12],
                    acc[16 * kk2 + 13],
                ),
                pack_e4m3x4(
                    acc[16 * kk2 + 10],
                    acc[16 * kk2 + 11],
                    acc[16 * kk2 + 14],
                    acc[16 * kk2 + 15],
                ),
            ]
            for kk2 in range(nk)
        ]
        # the fp8 MMA accumulates with reduced precision, so each tile's sums start from zero
        c = [Float32(0.0)] * 4
        for kk2 in cutlass.range_constexpr(nk):
            c = mma_fp8(pa[kk2], [ones, ones], c)
        for x in cutlass.range_constexpr(4):
            l_acc[x] = l_acc[x] + c[x]
        for npair in cutlass.range_constexpr(self.d // 16):
            part = [[Float32(0.0)] * 4 for _ in range(2)]
            for kk2 in cutlass.range_constexpr(nk):
                # lane holds keys 2q, 2q + 1, + 8, + 16 of dims 2g and 2g + 1; split the bytes by dim
                r = ldsm_x4_trans(vtile.addr(32 * kk2 + lane, npair))
                bs = [
                    [prmt(r[0], r[1], 0x6420), prmt(r[2], r[3], 0x6420)],
                    [prmt(r[0], r[1], 0x7531), prmt(r[2], r[3], 0x7531)],
                ]
                for h in cutlass.range_constexpr(2):
                    part[h] = mma_fp8(pa[kk2], bs[h], part[h])
            for h in cutlass.range_constexpr(2):
                for x in cutlass.range_constexpr(4):
                    i = 4 * (2 * npair + h) + x
                    o_acc[i] = o_acc[i] + part[h][x]

    @cute.jit
    def _store_o(self, go, row, t, oss, o_acc, r, inv, valid):
        if const_expr(self.bf16):
            AttnFwdSm80._store_o(self, go, row, t, oss, o_acc, r, inv, valid)
        else:
            if valid:
                # O tiles 2p and 2p + 1 hold the even and odd dims of 16p + 4t .. 16p + 4t + 3
                for p in cutlass.range_constexpr(self.d // 16):
                    e0 = 4 * (2 * p) + 2 * r
                    e1 = 4 * (2 * p + 1) + 2 * r
                    st_global_v2_b32(
                        go + (Int64(row) * oss + Int64(16 * p + 4 * t)) * 2,
                        pack_half2(o_acc[e0] * inv, o_acc[e1] * inv, "bf16"),
                        pack_half2(o_acc[e0 + 1] * inv, o_acc[e1 + 1] * inv, "bf16"),
                    )
