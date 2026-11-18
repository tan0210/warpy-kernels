# type: ignore

import cutlass
from cutlass import Int32, Int64, const_expr, cute

from ..ptx import (
    abs_bits,
    join_b16,
    mma_fp8,
    mx_scale,
    pack_e4m3x2,
    st_global_u8,
    st_global_v2_b32,
)
from ..utils.fragments import Accum
from ..utils.warp import quad_transpose
from .sm80 import GemmSm80


class GemmSm89(GemmSm80):
    """Adds e4m3 inputs (``mma.sync`` m16n8k32) and MXFP8 outputs, for SM89 and newer."""

    FP8 = True
    OUT_MX = True

    def _mma(self, a, b, c, sfa):
        if self.fp8:
            return mma_fp8(a, b, c)
        return super()._mma(a, b, c, sfa)

    def _promote(self, acc, part):
        for i in range(part.mt * part.nt * 4):
            acc[i] = acc[i] + part[i]

    @cute.jit
    def _run(self, pipe, nk, sbase, warp_m, warp_n, lane, acc, sfp, sf16):
        if const_expr(self.fp8):
            # e4m3 MMAs accumulate one k-tile at a time, then add in fp32, which keeps the accuracy of cuBLAS
            self._part = Accum(4, self.NT)
        GemmSm80._run(self, pipe, nk, sbase, warp_m, warp_n, lane, acc, sfp, sf16)

    @cute.jit
    def _compute(self, st, sbase, warp_m, warp_n, lane, acc, sfp, sf16, kt):
        if const_expr(self.fp8):
            self._part.fill(0.0)
            GemmSm80._compute(
                self, st, sbase, warp_m, warp_n, lane, self._part, sfp, sf16, kt
            )
            self._promote(acc, self._part)
        else:
            GemmSm80._compute(self, st, sbase, warp_m, warp_n, lane, acc, sfp, sf16, kt)

    @cute.jit
    def _mma_step(self, buf, fa, fb, acc, sfp, sf16, kt, kk):
        if const_expr(self.fp8):
            if const_expr(kk == 0):
                self._part.fill(0.0)
            GemmSm80._mma_step(self, buf, fa, fb, self._part, sfp, sf16, kt, kk)
            if const_expr(kk == self.KK - 1):
                self._promote(acc, self._part)
        else:
            GemmSm80._mma_step(self, buf, fa, fb, acc, sfp, sf16, kt, kk)

    @cute.jit
    def _store_mx(self, gq, gqs, n, row, c0, t, vals):
        # each 32-column block of a row lives in one quad: 4 column tiles x 2 values per lane
        for blk in cutlass.range_constexpr(self.NT // 4):
            amax = Int32(0)
            for x in cutlass.range_constexpr(8):
                amax = max(amax, abs_bits(vals[blk * 8 + x]))
            amax = max(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
            amax = max(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
            byte, inv = mx_scale(amax)
            units = [
                [
                    pack_e4m3x2(
                        vals[blk * 8 + 2 * j] * inv, vals[blk * 8 + 2 * j + 1] * inv
                    )
                ]
                for j in range(4)
            ]
            # afterwards lane t holds all 8 bytes of column tile `4 * blk + t`
            units = quad_transpose(units, t)
            st_global_v2_b32(
                gq + Int64(row) * Int64(n) + Int64(c0 + (blk * 4 + t) * 8),
                join_b16(units[0][0], units[1][0]),
                join_b16(units[2][0], units[3][0]),
            )
            if t == 0:
                st_global_u8(
                    gqs + Int64(row) * Int64(n // 32) + Int64((c0 + blk * 32) // 32),
                    byte,
                )
