# pyright: reportAttributeAccessIssue=false

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op
from cutlass.cutlass_dsl import T


def asm(ret, args, code, constraints, side_effects, loc, ip):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        code,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def unpack_struct(res, ty, cls, n, loc, ip):
    return [cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n)]


def _ldsm(code, addr, loc, ip):
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        code,
        "=r,=r,=r,=r,r,~{memory}",
        True,
        loc,
        ip,
    )
    return unpack_struct(res, T.i32(), Int32, 4, loc, ip)


def _mma(code, a, b, c, loc, ip):
    args = [Int32(x) for x in a] + [Int32(x) for x in b] + [Float32(x) for x in c]
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        args,
        code,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 4, loc, ip)


@dsl_user_op
def ldsm_x4(addr, *, loc=None, ip=None):
    return _ldsm(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        addr,
        loc,
        ip,
    )


@dsl_user_op
def ldsm_x4_trans(addr, *, loc=None, ip=None):
    return _ldsm(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
        addr,
        loc,
        ip,
    )


@dsl_user_op
def mma_16816(a, b, c, ty, *, loc=None, ip=None):
    code = (
        f"mma.sync.aligned.m16n8k16.row.col.f32.{ty}.{ty}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    return _mma(code, a, b, c, loc, ip)


@dsl_user_op
def ue8m0_inv(g, *, loc=None, ip=None):
    # ue8m0 byte of `1 / g` for a power of two `g`, from its exponent bits
    code = "{\n\t.reg .b32 r;\n\tmov.b32 r, $1;\n\tshr.u32 r, r, 23;\n\tsub.s32 $0, 254, r;\n\t}"
    return Int32(asm(T.i32(), [Float32(g)], code, "=r,f", False, loc, ip))


@dsl_user_op
def pow2_inv(g, *, loc=None, ip=None):
    # `1 / g` for a normal power of two `g`, from its exponent bits
    code = "{\n\t.reg .b32 r;\n\tmov.b32 r, $1;\n\tsub.s32 r, 0x7F000000, r;\n\tmov.b32 $0, r;\n\t}"
    return Float32(asm(T.f32(), [Float32(g)], code, "=f,f", False, loc, ip))


@dsl_user_op
def prmt(a, b, sel, *, loc=None, ip=None):
    # bytes of `{b, a}` picked by the four nibbles of `sel`
    return Int32(
        asm(
            T.i32(),
            [Int32(a), Int32(b)],
            f"prmt.b32 $0, $1, $2, {sel:#x};",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_shared_f32(saddr, *, loc=None, ip=None):
    return Float32(
        asm(
            T.f32(),
            [Int32(saddr)],
            "ld.shared.f32 $0, [$1];",
            "=f,r,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def pack_half2(lo, hi, ty, *, loc=None, ip=None):
    return Int32(
        asm(
            T.i32(),
            [Float32(hi), Float32(lo)],
            f"cvt.rn.{ty}x2.f32 $0, $1, $2;",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def unpack_half2(x, ty, *, loc=None, ip=None):
    if ty == "bf16":
        code = "{ .reg .b32 t; shl.b32 t, $2, 16; mov.b32 $0, t; and.b32 t, $2, 0xffff0000; mov.b32 $1, t; }"
    else:
        code = "{ .reg .b16 l, h; mov.b32 {l, h}, $2; cvt.f32.f16 $0, l; cvt.f32.f16 $1, h; }"
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int32(x)],
        code,
        "=f,=f,r",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 2, loc, ip)


@dsl_user_op
def ds_scale(bound, *, loc=None, ip=None):
    # `2^(7 - floor(log2 bound))` clamped to `[2^-60, 2^60]`, from exponent bits only
    code = (
        "{\n\t.reg .b32 r;\n\t"
        "mov.b32 r, $1;\n\tshr.u32 r, r, 23;\n\tsub.s32 r, 261, r;\n\t"
        "max.s32 r, r, 67;\n\tmin.s32 r, r, 187;\n\tshl.b32 r, r, 23;\n\tmov.b32 $0, r;\n\t}"
    )
    return Float32(asm(T.f32(), [Float32(bound)], code, "=f,f", False, loc, ip))


@dsl_user_op
def div_rn(x, y, *, loc=None, ip=None):
    return Float32(
        asm(
            T.f32(),
            [Float32(x), Float32(y)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def cp_async16(saddr, gaddr, *, loc=None, ip=None):
    asm(
        None,
        [Int32(saddr), Int64(gaddr)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_global_b32(gaddr, *, loc=None, ip=None):
    return Int32(
        asm(
            T.i32(),
            [Int64(gaddr)],
            "ld.global.b32 $0, [$1];",
            "=r,l,~{memory}",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_nc_f32(gaddr, *, loc=None, ip=None):
    return Float32(
        asm(
            T.f32(),
            [Int64(gaddr)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def ld_global_nc_v2_f32(gaddr, *, loc=None, ip=None):
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(gaddr)],
        "ld.global.nc.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 2, loc, ip)


@dsl_user_op
def ld_global_bf16(gaddr, n, *, loc=None, ip=None):
    # `n` (2 or 4) bf16 at `gaddr` as fp32, one vector load
    if n == 4:
        code = (
            "{\n\t.reg .b32 a, b, t0, t1, t2, t3;\n\tld.global.nc.v2.b32 {a, b}, [$4];\n\t"
            "shl.b32 t0, a, 16;\n\tand.b32 t1, a, 0xffff0000;\n\tshl.b32 t2, b, 16;\n\tand.b32 t3, b, 0xffff0000;\n\t"
            "mov.b32 $0, t0;\n\tmov.b32 $1, t1;\n\tmov.b32 $2, t2;\n\tmov.b32 $3, t3;\n\t}"
        )
        res = asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Int64(gaddr)],
            code,
            "=f,=f,=f,=f,l",
            False,
            loc,
            ip,
        )
        return unpack_struct(res, T.f32(), Float32, 4, loc, ip)
    code = (
        "{\n\t.reg .b32 a, t0, t1;\n\tld.global.nc.b32 a, [$2];\n\t"
        "shl.b32 t0, a, 16;\n\tand.b32 t1, a, 0xffff0000;\n\tmov.b32 $0, t0;\n\tmov.b32 $1, t1;\n\t}"
    )
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(gaddr)],
        code,
        "=f,=f,l",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 2, loc, ip)


@dsl_user_op
def ld_shared_v2_f32(saddr, *, loc=None, ip=None):
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int32(saddr)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r,~{memory}",
        True,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 2, loc, ip)


@dsl_user_op
def st_global_b32(gaddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(val)],
        "st.global.b32 [$0], $1;",
        "l,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_global_v4_b32(gaddr, w0, w1, w2, w3, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(w0), Int32(w1), Int32(w2), Int32(w3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_global_v4_f32(gaddr, x0, x1, x2, x3, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Float32(x0), Float32(x1), Float32(x2), Float32(x3)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_global_cg_v4_f32(gaddr, *, loc=None, ip=None):
    res = asm(  # bypasses L1
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(gaddr)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l,~{memory}",
        True,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 4, loc, ip)


@dsl_user_op
def fence_gpu(*, loc=None, ip=None):
    asm(None, [], "fence.acq_rel.gpu;", "~{memory}", True, loc, ip)


@dsl_user_op
def st_release_gpu_b32(gaddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(val)],
        "st.release.gpu.global.b32 [$0], $1;",
        "l,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def wait_flag_eq(gaddr, value, *, loc=None, ip=None):
    # spin until the flag equals `value`, with acquire ordering
    code = (
        "{\n\t.reg .pred p;\n\t.reg .b32 v;\n\t"
        "SPIN${:uid}:\n\tld.acquire.gpu.global.b32 v, [$0];\n\tsetp.ne.b32 p, v, $1;\n\t@p bra SPIN${:uid};\n\t}"
    )
    asm(None, [Int64(gaddr), Int32(value)], code, "l,r,~{memory}", True, loc, ip)


@dsl_user_op
def ld_global_cg_v2_f32(gaddr, *, loc=None, ip=None):
    res = asm(  # bypasses L1
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(gaddr)],
        "ld.global.cg.v2.f32 {$0, $1}, [$2];",
        "=f,=f,l,~{memory}",
        True,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 2, loc, ip)


@dsl_user_op
def st_global_v2_f32(gaddr, x0, x1, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Float32(x0), Float32(x1)],
        "st.global.v2.f32 [$0], {$1, $2};",
        "l,f,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def wait_flag_acquire(gaddr, *, loc=None, ip=None):
    # spin until the flag is set, then reset it
    code = (
        "{\n\t.reg .pred p;\n\t.reg .b32 v;\n\t"
        "SPIN${:uid}:\n\tld.acquire.gpu.global.b32 v, [$0];\n\tsetp.eq.b32 p, v, 0;\n\t@p bra SPIN${:uid};\n\t"
        "st.relaxed.gpu.global.b32 [$0], 0;\n\t}"
    )
    asm(None, [Int64(gaddr)], code, "l,~{memory}", True, loc, ip)


@dsl_user_op
def st_global_b16(gaddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(val)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $1;\n\tst.global.b16 [$0], lo;\n\t}",
        "l,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_global_zero(gaddr, nwords, *, loc=None, ip=None):
    # `nwords` (2 or 4) zero words, one vector store
    vec = "{z, z, z, z}" if nwords == 4 else "{z, z}"
    v = "v4" if nwords == 4 else "v2"
    asm(
        None,
        [Int64(gaddr)],
        f"{{\n\t.reg .b32 z;\n\tmov.b32 z, 0;\n\tst.global.{v}.b32 [$0], {vec};\n\t}}",
        "l,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_b32(saddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int32(saddr), Int32(val)],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_shared_v2_f32(saddr, v0, v1, *, loc=None, ip=None):
    asm(
        None,
        [Int32(saddr), Float32(v0), Float32(v1)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def red_add_f32(gaddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Float32(val)],
        "red.global.add.f32 [$0], $1;",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def red_max_pos_f32(gaddr, val, *, loc=None, ip=None):
    # valid for `val >= 0`, where floats order like their bits
    asm(
        None,
        [Int64(gaddr), Float32(val)],
        "{\n\t.reg .b32 r;\n\tmov.b32 r, $1;\n\tred.global.max.u32 [$0], r;\n\t}",
        "l,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def warp_sync(*, loc=None, ip=None):
    asm(None, [], "bar.warp.sync 0xffffffff;", "~{memory}", True, loc, ip)


@dsl_user_op
def ld_global_nc_u8(gaddr, *, loc=None, ip=None):
    return Int32(
        asm(
            T.i32(),
            [Int64(gaddr)],
            "{\n\t.reg .b16 t;\n\tld.global.nc.u8 t, [$1];\n\tcvt.u32.u16 $0, t;\n\t}",
            "=r,l",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def st_global_u8(gaddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(val)],
        "{\n\t.reg .b16 t;\n\tcvt.u16.u32 t, $1;\n\tst.global.u8 [$0], t;\n\t}",
        "l,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def st_global_v2_b32(gaddr, w0, w1, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(w0), Int32(w1)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def abs_bits(x, *, loc=None, ip=None):
    # bits of `|x|`, which order like the values themselves
    code = "{\n\t.reg .b32 r;\n\tmov.b32 r, $1;\n\tand.b32 $0, r, 0x7fffffff;\n\t}"
    return Int32(asm(T.i32(), [Float32(x)], code, "=r,f", False, loc, ip))


@dsl_user_op
def mx_scale(amax_bits, *, loc=None, ip=None):
    # ue8m0 byte of the smallest power of two that maps `amax` to at most 448, and its inverse
    code = (
        "{\n\t.reg .b32 e;\n\t"
        "add.u32 e, $2, 0x1fffff;\n\tshr.u32 e, e, 23;\n\tsub.s32 e, e, 8;\n\t"
        "max.s32 e, e, 0;\n\tmin.s32 e, e, 254;\n\tmov.b32 $0, e;\n\t"
        "sub.s32 e, 254, e;\n\tmax.s32 e, e, 1;\n\tshl.b32 e, e, 23;\n\tmov.b32 $1, e;\n\t}"
    )
    res = asm(
        llvm.StructType.get_literal([T.i32(), T.f32()]),
        [Int32(amax_bits)],
        code,
        "=r,=f,r",
        False,
        loc,
        ip,
    )
    return Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)), Float32(
        llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    )


@dsl_user_op
def join_b16(lo, hi, *, loc=None, ip=None):
    # low halves of ``lo`` and ``hi`` as one word
    return Int32(
        asm(
            T.i32(),
            [Int32(lo), Int32(hi)],
            "prmt.b32 $0, $1, $2, 0x5410;",
            "=r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def pack_fixed2(x0, x1, *, loc=None, ip=None):
    # `x0` and `x1` rounded to int32 lanes of one int64 that adds lane-wise, `x0` low
    code = (
        "{\n\t.reg .b32 a, b, z;\n\t.reg .s64 la, hb;\n\t"
        "cvt.rni.s32.f32 a, $1;\n\tcvt.rni.s32.f32 b, $2;\n\t"
        "cvt.s64.s32 la, a;\n\tmov.b32 z, 0;\n\tmov.b64 hb, {z, b};\n\t"
        "add.s64 $0, hb, la;\n\t}"
    )
    return Int64(
        asm(T.i64(), [Float32(x0), Float32(x1)], code, "=l,f,f", False, loc, ip)
    )


@dsl_user_op
def red_add_u64(gaddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int64(val)],
        "red.global.add.u64 [$0], $1;",
        "l,l,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def ld_fixed2_f32(gaddr, *, loc=None, ip=None):
    # the two int32 lane sums of a `pack_fixed2` accumulator at `gaddr`, as fp32
    code = (
        "{\n\t.reg .b32 l, h, s;\n\t"
        "ld.global.v2.b32 {l, h}, [$2];\n\t"
        "shr.s32 s, l, 31;\n\tsub.s32 h, h, s;\n\t"
        "cvt.rn.f32.s32 $0, l;\n\tcvt.rn.f32.s32 $1, h;\n\t}"
    )
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int64(gaddr)],
        code,
        "=f,=f,l,~{memory}",
        True,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 2, loc, ip)


@dsl_user_op
def st_shared_b64(saddr, val, *, loc=None, ip=None):
    asm(
        None,
        [Int32(saddr), Int64(val)],
        "st.shared.b64 [$0], $1;",
        "r,l,~{memory}",
        True,
        loc,
        ip,
    )
