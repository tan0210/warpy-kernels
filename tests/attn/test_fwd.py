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
