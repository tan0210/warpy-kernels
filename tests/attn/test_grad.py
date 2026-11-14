import torch

from tests.fixtures import (
    attention,
    compare,
    compare_seeds,
    hard_inputs,
    randn_qkv,
    rel_l2,
)
from tests.registry import case

TOL = {"out": 6e-2, "lse": 5e-2, "dq": 8e-2, "dk": 8e-2, "dv": 8e-2}

SHAPES = [
    (1, 1, 1, 128, 128, 128),
    (2, 4, 2, 1000, 1000, 128),
    (1, 2, 1, 300, 777, 64),
    (1, 2, 2, 512, 1024, 128),
    (1, 4, 4, 2048, 2048, 64),
]

BWD_CONFIGS = [
    {"dq_bulk": False},
    {"num_stages": 2},
    {"num_stages": 3},
    {"num_stages": 2, "dq_bulk": False},
    {"kb_major": True},
    {"block_m": 32},
    {"block_m": 64},
]


def _run(ctx, shape, causal, bwd_cfg=None):
    B, Hq, Hkv, Nq, Nk, d = shape
    q, k, v = randn_qkv(B, Hq, Hkv, Nq, Nk, d)
    g = torch.Generator(device="cuda").manual_seed(7)
    dout = torch.randn(B, Hq, Nq, d, device="cuda", generator=g)
    compare(ctx, q, k, v, dout, causal, TOL, bwd_cfg=bwd_cfg)


@case("grad", shape=SHAPES, causal=[False, True])
def grad(ctx, shape, causal):
    _run(ctx, shape, causal)


def _bwd_fits(shape, bwd_cfg, **_):
    return not (shape[5] == 128 and bwd_cfg.get("block_m") == 64)


@case("grad_cfg", where=_bwd_fits, shape=[(2, 4, 2, 1000, 1000, 128), (1, 4, 4, 2048, 2048, 64)], causal=[False, True],
      bwd_cfg=BWD_CONFIGS)
def grad_cfg(ctx, shape, causal, bwd_cfg):
    _run(ctx, shape, causal, bwd_cfg)


@case("deterministic", shape=SHAPES, causal=[False, True])
def deterministic(ctx, shape, causal):
    B, Hq, Hkv, Nq, Nk, d = shape
    q, k, v = randn_qkv(B, Hq, Hkv, Nq, Nk, d)
    dout = torch.randn(B, Hq, Nq, d, device="cuda")
    grads = []
    for _ in range(2):
        qq, kk, vv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        out, _ = attention(qq, kk, vv, causal, bwd_cfg={"deterministic": True})
        out.backward(dout)
        grads.append((qq.grad, kk.grad, vv.grad))
    ctx.expect(all(torch.equal(a, b) for a, b in zip(*grads)), "gradients differ between identical runs")
    _run(ctx, shape, causal, {"deterministic": True})


HARD_SHAPES = [(1, 4, 2, 1024, 1024, 128), (1, 4, 4, 1024, 1024, 64)]
REPEAT_CONFIGS = [{}, {"num_stages": 2}, {"dq_bulk": False}]


@case("grad_hard", shape=HARD_SHAPES, causal=[False, True], bwd_cfg=[{}, {"num_stages": 2}, {"deterministic": True}])
def grad_hard(ctx, shape, causal, bwd_cfg):
    compare_seeds(ctx, lambda seed: hard_inputs(*shape, seed=seed), (0, 1, 2), causal, TOL, bwd_cfg=bwd_cfg)


@case("repeat", shape=[(2, 4, 2, 1000, 1000, 128), (1, 4, 4, 2048, 2048, 64)], causal=[False, True],
      bwd_cfg=REPEAT_CONFIGS)
def repeat(ctx, shape, causal, bwd_cfg):
    B, Hq, Hkv, Nq, Nk, d = shape
    q, k, v = randn_qkv(B, Hq, Hkv, Nq, Nk, d)
    dout = torch.randn(B, Hq, Nq, d, device="cuda")
    grads = []
    for _ in range(4):
        qq, kk, vv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
        out, _ = attention(qq, kk, vv, causal, bwd_cfg=bwd_cfg)
        out.backward(dout)
        grads.append((qq.grad, kk.grad, vv.grad))
    first = grads[0]
    # dK and dV are deterministic, so a difference between identical runs means a race
    ctx.expect(all(torch.equal(g[1], first[1]) and torch.equal(g[2], first[2]) for g in grads[1:]),
               "dK or dV differ between identical runs")
    # dQ also varies with the order of its fp32 reduction
    ctx.metric("dq_spread", max(rel_l2(g[0], first[0]) for g in grads[1:]), 1e-2, gate=False)
