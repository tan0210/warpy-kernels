# pyright: reportAttributeAccessIssue=false

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.op import dsl_user_op
from cutlass.cutlass_dsl import T

SMEM_LIMIT = 101376  # usable shared memory bytes per CTA


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
def ldsm_x2_trans_b8(addr, *, loc=None, ip=None):
    return _ldsm(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {$0, $1, $2, $3}, [$4];",
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
def mma_fp8(a, b, c, ta="e4m3", tb="e4m3", *, loc=None, ip=None):
    code = (
        f"mma.sync.aligned.m16n8k32.row.col.f32.{ta}.{tb}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    return _mma(code, a, b, c, loc, ip)


@dsl_user_op
def mma_mxf8(a, b, c, *, loc=None, ip=None):
    # `mma_fp8` with unit scales, twice as fast on SM120
    code = (
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, $14, {0, 0}, $14, {0, 0};"
    )
    args = (
        [Int32(x) for x in a]
        + [Int32(x) for x in b]
        + [Float32(x) for x in c]
        + [Int32(0x7F7F7F7F)]  # ue8m0 1.0 in every byte
    )
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        args,
        code,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f,r",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 4, loc, ip)


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
def red_add_v2_f32(gaddr, v0, v1, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Float32(v0), Float32(v1)],
        "red.global.add.v2.f32 [$0], {$1, $2};",
        "l,f,f,~{memory}",
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
def bulk_reduce_add_f32(gaddr, saddr, nbytes, *, loc=None, ip=None):
    asm(
        None,
        [Int64(gaddr), Int32(saddr), Int32(nbytes)],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,r,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def warp_sync(*, loc=None, ip=None):
    asm(None, [], "bar.warp.sync 0xffffffff;", "~{memory}", True, loc, ip)


def swizzle_offset(row_bytes, row, chunk):
    if row_bytes == 128:  # 128B swizzle
        return row * 128 + (((row & 7) ^ chunk) << 4)
    return row * 64 + ((((row >> 1) & 3) ^ chunk) << 4)  # 64B swizzle
