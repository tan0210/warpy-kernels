# type: ignore

import cutlass
from cutlass import Int32, cute


@cute.jit
def grouped_tile(tile: Int32, npm: Int32, npn: Int32, group_m: cutlass.Constexpr):
    gm = Int32(
        group_m
    )  # row blocks that sweep all column blocks together, for L2 reuse
    per_group = gm * npn
    first_m = (tile // per_group) * gm
    rem = npm - first_m
    size = min(gm, rem)
    return first_m + (tile % per_group) % size, (tile % per_group) // size
