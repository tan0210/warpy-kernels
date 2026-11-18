import torch

from tests.fixtures import (
    attention,
    compare,
    dequantize_qkv,
    quantize_qkv,
    randn_qkv,
    reference,
    rel_l2,
    sink_qkv,
)
from tests.registry import case
from warpy_kernels import attn_fwd, device_arch
from warpy_kernels.attn.fwd_api import (
    FWD_CANDIDATES_BF16,
    FWD_CANDIDATES_BF16_SM89,
    FWD_CANDIDATES_SM89,
)

TOL = {"out": 6e-2, "lse": 5e-2}

SHAPES = [
    (1, 1, 1, 128, 64, 128),
    (2, 4, 2, 1000, 1000, 128),
    (1, 2, 1, 300, 777, 64),
    (1, 2, 2, 512, 1024, 128),
    (1, 4, 4, 2048, 2048, 64),
    (1, 2, 1, 512, 128, 128),
    (1, 2, 1, 512, 128, 64),
]

FWD_CONFIGS = [
    {"pipelined": False, "num_stages": 3, "block_n": 64},
    {"pipelined": True, "num_stages": 3, "block_n": 64},
    {"pipelined": False, "num_stages": 2, "block_n": 128},
    {"pipelined": True, "num_stages": 3, "block_n": 128},
    {"pipelined": True, "num_stages": 4, "block_n": 64},
    {"tile_local": False},
    {"lpt": True},
]


def _unique(cands):
    flat = cands[64] + cands[128]
    return [c for i, c in enumerate(flat) if c not in flat[:i]]


@case("fwd", shape=SHAPES, causal=[False, True])
def fwd(ctx, shape, causal):
    q, k, v = randn_qkv(*shape)
    compare(ctx, q, k, v, None, causal, TOL)


def _fwd_fits(shape, fwd_cfg, **_):
    return not (
        shape[5] == 128
        and fwd_cfg.get("block_n") == 128
        and fwd_cfg.get("num_stages") == 3
    )


@case(
    "fwd_cfg",
    where=_fwd_fits,
    shape=[SHAPES[1], SHAPES[4], SHAPES[5]],
    causal=[False, True],
    fwd_cfg=FWD_CONFIGS,
    tma=[False, True],
)
def fwd_cfg(ctx, shape, causal, fwd_cfg, tma):
    q, k, v = randn_qkv(*shape)
    compare(ctx, q, k, v, None, causal, TOL, fwd_cfg=dict(fwd_cfg, tma=tma))


@case(
    "fwd_ws",
    where=_fwd_fits,
    shape=[SHAPES[1], SHAPES[4], SHAPES[5]],
    causal=[False, True],
    fwd_cfg=FWD_CONFIGS,
)
def fwd_ws(ctx, shape, causal, fwd_cfg):
    q, k, v = randn_qkv(*shape)
    compare(
        ctx, q, k, v, None, causal, TOL, fwd_cfg=dict(fwd_cfg, tma=True, warp_spec=True)
    )


@case(
    "fwd_sm89_cfg",
    where=lambda shape, fwd_cfg, **_: (
        fwd_cfg in FWD_CANDIDATES_SM89[shape[5]] and device_arch() != 120
    ),
    shape=[SHAPES[1], SHAPES[4], SHAPES[5], SHAPES[6]],
    causal=[False, True],
    fwd_cfg=_unique(FWD_CANDIDATES_SM89),
)
def fwd_sm89_cfg(ctx, shape, causal, fwd_cfg):
    q, k, v = randn_qkv(*shape)
    compare(ctx, q, k, v, None, causal, TOL, fwd_cfg=fwd_cfg)


@case("sink", gap=[8, 13, 16])
def sink(ctx, gap):
    q, k, v = sink_qkv(256, 32768, 128, gap)
    ref, _ = reference(*dequantize_qkv(*quantize_qkv(q, k, v)), False)
    for tile_local in (True, False):
        out, _ = attention(q, k, v, fwd_cfg={"tile_local": tile_local})
        err = rel_l2(out, ref)
        if tile_local:
            ctx.metric("out", err, 2e-2)
        else:
            ctx.note(f"running-max reference: out {err:.2e}")


BF16_TOL = {"out": 1e-2, "lse": 1e-2}


def _bf16_compare(ctx, shape, causal, fwd_cfg=None):
    Nq, Nk = shape[3], shape[4]
    q, k, v = [x.to(torch.bfloat16) for x in randn_qkv(*shape)]
    out, lse, _ = attn_fwd(q, k, v, causal=causal, **(fwd_cfg or {}))
    out_ref, lse_ref = reference(q.float(), k.float(), v.float(), causal)
    visible = torch.ones(Nq, dtype=torch.bool, device=q.device)
    if causal:
        visible = (torch.arange(Nq, device=q.device) + Nk - Nq) >= 0
    ctx.metric(
        "out", rel_l2(out[:, :, visible], out_ref[:, :, visible]), BF16_TOL["out"]
    )
    ctx.metric(
        "lse",
        (lse[:, :, visible] - lse_ref[:, :, visible]).abs().max().item(),
        BF16_TOL["lse"],
    )
    ctx.expect(
        bool((out[:, :, ~visible] == 0).all()),
        "rows without a visible key must give O = 0",
    )


@case("fwd_bf16", shape=SHAPES, causal=[False, True])
def fwd_bf16(ctx, shape, causal):
    _bf16_compare(ctx, shape, causal)


BF16_CANDIDATES = (
    FWD_CANDIDATES_BF16 if device_arch() == 120 else FWD_CANDIDATES_BF16_SM89
)


@case(
    "fwd_bf16_cfg",
    where=lambda shape, fwd_cfg, **_: fwd_cfg in BF16_CANDIDATES[shape[5]],
    shape=[SHAPES[1], SHAPES[4], SHAPES[5], SHAPES[6]],
    causal=[False, True],
    fwd_cfg=_unique(BF16_CANDIDATES),
)
def fwd_bf16_cfg(ctx, shape, causal, fwd_cfg):
    _bf16_compare(ctx, shape, causal, fwd_cfg)
