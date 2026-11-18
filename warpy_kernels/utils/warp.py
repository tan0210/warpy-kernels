# type: ignore

import cutlass
from cutlass import const_expr, cute


@cute.jit
def quad_transpose(units, t):
    for b in cutlass.range_constexpr(1, 3):
        hi = (t & b) != 0
        for p in cutlass.range_constexpr(4):
            if const_expr(not p & b):
                q = p | b
                for w in cutlass.range_constexpr(len(units[p])):
                    x = cute.arch.shuffle_sync_bfly(
                        units[p][w] if hi else units[q][w], offset=b
                    )
                    keep_p, keep_q = (
                        units[p][w],
                        units[q][w],
                    )  # lane `t` ends with lane `p`'s `units[t]`
                    units[p][w] = x if hi else keep_p
                    units[q][w] = keep_q if hi else x
    return units
