# pyright: reportAttributeAccessIssue=false

from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op
from cutlass.cutlass_dsl import T

from .common import _mma, asm, unpack_struct


@dsl_user_op
def mma_fp8(a, b, c, ta="e4m3", tb="e4m3", *, loc=None, ip=None):
    code = (
        f"mma.sync.aligned.m16n8k32.row.col.f32.{ta}.{tb}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    return _mma(code, a, b, c, loc, ip)


@dsl_user_op
def pack_e4m3x4(x0, x1, x2, x3, *, loc=None, ip=None):
    code = (
        "{\n\t.reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t}"
    )
    return Int32(
        asm(
            T.i32(),
            [Float32(x0), Float32(x1), Float32(x2), Float32(x3)],
            code,
            "=r,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def e4m3x4_to_f32(bits, *, loc=None, ip=None):
    code = (
        "{\n\t.reg .b16 lo, hi;\n\t.reg .b32 h01, h23;\n\t.reg .f16 a, b, c, d;\n\t"
        "mov.b32 {lo, hi}, $4;\n\t"
        "cvt.rn.f16x2.e4m3x2 h01, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 h23, hi;\n\t"
        "mov.b32 {a, b}, h01;\n\t"
        "mov.b32 {c, d}, h23;\n\t"
        "cvt.f32.f16 $0, a;\n\tcvt.f32.f16 $1, b;\n\tcvt.f32.f16 $2, c;\n\tcvt.f32.f16 $3, d;\n\t}"
    )
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(bits)],
        code,
        "=f,=f,=f,=f,r",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 4, loc, ip)


@dsl_user_op
def st_shared_e4m3_pair(addr_lo, addr_hi, x_lo, x_hi, *, loc=None, ip=None):
    code = (
        "{\n\t.reg .b16 h, h2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h, $3, $2;\n\t"
        "st.shared.u8 [$0], h;\n\t"
        "shr.b16 h2, h, 8;\n\t"
        "st.shared.u8 [$1], h2;\n\t}"
    )
    asm(
        None,
        [Int32(addr_lo), Int32(addr_hi), Float32(x_lo), Float32(x_hi)],
        code,
        "r,r,f,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def pack_e4m3x2(x0, x1, *, loc=None, ip=None):
    # two e4m3 values in the low 16 bits, ``x0`` first
    code = (
        "{\n\t.reg .b16 p;\n\tcvt.rn.satfinite.e4m3x2.f32 p, $2, $1;\n\t"
        "cvt.u32.u16 $0, p;\n\t}"
    )
    return Int32(
        asm(T.i32(), [Float32(x0), Float32(x1)], code, "=r,f,f", False, loc, ip)
    )
