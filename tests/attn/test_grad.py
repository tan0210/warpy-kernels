import torch

from tests.fixtures import attention, compare, randn_qkv
from tests.registry import case

TOL = dict(out=6e-2, lse=5e-2, dq=8e-2, dk=8e-2, dv=8e-2)

SHAPES = [
    (1, 1, 1, 128, 128, 128),
    (2, 4, 2, 1000, 1000, 128),
    (1, 2, 1, 300, 777, 64),
    (1, 2, 2, 512, 1024, 128),
    (1, 4, 4, 2048, 2048, 64),
]

BWD_CONFIGS = [
    dict(dq_bulk=False),
    dict(num_warps=4),
    dict(num_stages=3),
    dict(block_m=32, dq_bulk=True),
    dict(block_m=32, dq_bulk=False, min_blocks=2),
    dict(kb_major=True),
    dict(dq_wg=True),
    dict(dq_wg=True, dq_bulk=False),
    dict(dq_wg=True, num_stages=3, kb_major=True),
    dict(block_m=32, dq_bulk=False, min_blocks=2, kb_major=True),
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
    return not (shape[5] == 128 and bwd_cfg.get("min_blocks") == 2)


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
        out, _ = attention(qq, kk, vv, causal, bwd_cfg=dict(deterministic=True))
        out.backward(dout)
        grads.append((qq.grad, kk.grad, vv.grad))
    ctx.expect(all(torch.equal(a, b) for a, b in zip(*grads)), "gradients differ between identical runs")
    _run(ctx, shape, causal, dict(deterministic=True))
    