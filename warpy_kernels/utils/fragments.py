# type: ignore

from cutlass import Float32, cute

from ..ptx import ldsm_x4, ldsm_x4_trans
from .staged import Staged


def ldsm_a(tile, row0, kk, lane):
    row = (
        row0 + (lane % 8) + ((lane // 8) % 2) * 8
    )  # an m16 A tile, 16 bytes of k per row
    return ldsm_x4(tile.addr(row, lane // 16 + 2 * kk))


def ldsm_b(tile, row0, kk, lane):
    row = row0 + (lane % 8) + (lane // 16) * 8  # two n8 B tiles, 16 bytes of k per row
    return ldsm_x4(tile.addr(row, (lane // 8) % 2 + 2 * kk))


def ldsm_a_trans(panels, col0, kk, lane):
    j = lane // 8  # k-major A in a `PanelTile`
    return ldsm_x4_trans(
        panels.addr(kk * 16 + (j // 2) * 8 + lane % 8, (col0 + (j % 2) * 8) // 8)
    )


def ldsm_b_trans(panels, col0, kk, lane):
    j = lane // 8  # k-major B in a `PanelTile`
    return ldsm_x4_trans(
        panels.addr(kk * 16 + (j % 2) * 8 + lane % 8, (col0 + (j // 2) * 8) // 8)
    )


class Accum(Staged):
    """Float32 accumulators of an ``mt`` x ``nt`` grid of m16n8 MMA tiles, four values per lane each.

    Value ``x`` of tile ``(mi, ni)`` sits at row ``lane // 4 + 8 * (x // 2)`` and column ``2 * (lane % 4) + x % 2``.
    """

    RUNTIME = ("regs",)

    def __init__(self, mt, nt):
        self.mt = mt
        self.nt = nt
        self.regs = cute.make_rmem_tensor(mt * nt * 4, Float32)

    def __getitem__(self, i):
        return self.regs[i]

    def __setitem__(self, i, v):
        self.regs[i] = v

    def fill(self, v):
        self.regs.fill(v)

    def tile(self, mi, ni):
        o = (mi * self.nt + ni) * 4
        return [self.regs[o], self.regs[o + 1], self.regs[o + 2], self.regs[o + 3]]

    def mma(self, mi, ni, op, a, b):
        o = (mi * self.nt + ni) * 4
        c = op(a, b, self.tile(mi, ni))
        for x in range(4):
            self.regs[o + x] = c[x]
