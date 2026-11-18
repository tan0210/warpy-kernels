# pyright: reportAttributeAccessIssue=false

from cutlass import Float32, Int32, Int64
from cutlass._mlir_helpers.op import dsl_user_op

from .common import asm


@dsl_user_op
def stsm_x4_trans(addr, r0, r1, r2, r3, *, loc=None, ip=None):
    asm(
        None,
        [Int32(addr), Int32(r0), Int32(r1), Int32(r2), Int32(r3)],
        "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r,~{memory}",
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
