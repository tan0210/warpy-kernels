# pyright: reportAttributeAccessIssue=false

from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op
from cutlass.cutlass_dsl import T

from .common import _ldsm, asm, unpack_struct


@dsl_user_op
def ldsm_x2_trans_b8(addr, *, loc=None, ip=None):
    return _ldsm(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {$0, $1, $2, $3}, [$4];",
        addr,
        loc,
        ip,
    )


@dsl_user_op
def mma_mxf8(a, b, c, sfa=0x7F, sfb=0x7F, *, loc=None, ip=None):
    # `mma_fp8` with ue8m0 scales in byte 0, twice as fast on SM120
    # lane L scales A row `L // 4 + 8 * (L % 2)`, the lanes of quad n scale B column n
    code = (
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, $14, {0, 0}, $15, {0, 0};"
    )
    args = (
        [Int32(x) for x in a]
        + [Int32(x) for x in b]
        + [Float32(x) for x in c]
        + [Int32(sfa), Int32(sfb)]
    )
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        args,
        code,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r,r",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 4, loc, ip)
