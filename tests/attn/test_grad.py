import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.fixtures import (
    attention,
    compare,
    compare_seeds,
    hard_inputs,
    randn_qkv,
    reference,
    rel_l2,
    row_p99,
)
from tests.registry import case
from warpy_kernels import attn_bwd, attn_fwd, device_arch
from warpy_kernels.attn.bwd_api import candidates_for

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


@case(
    "grad_cfg",
    where=_bwd_fits,
    shape=[(2, 4, 2, 1000, 1000, 128), (1, 4, 4, 2048, 2048, 64)],
    causal=[False, True],
    bwd_cfg=BWD_CONFIGS,
)
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
    ctx.expect(
        all(torch.equal(a, b) for a, b in zip(*grads)),
        "gradients differ between identical runs",
    )
    _run(ctx, shape, causal, {"deterministic": True})


HARD_SHAPES = [(1, 4, 2, 1024, 1024, 128), (1, 4, 4, 1024, 1024, 64)]
REPEAT_CONFIGS = [{}, {"num_stages": 2}, {"dq_bulk": False}]


@case(
    "grad_hard",
    shape=HARD_SHAPES,
    causal=[False, True],
    bwd_cfg=[{}, {"num_stages": 2}, {"deterministic": True}],
)
def grad_hard(ctx, shape, causal, bwd_cfg):
    compare_seeds(
        ctx,
        lambda seed: hard_inputs(*shape, seed=seed),
        (0, 1, 2),
        causal,
        TOL,
        bwd_cfg=bwd_cfg,
    )


@case(
    "repeat",
    shape=[(2, 4, 2, 1000, 1000, 128), (1, 4, 4, 2048, 2048, 64)],
    causal=[False, True],
    bwd_cfg=REPEAT_CONFIGS,
)
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
    ctx.expect(
        all(
            torch.equal(g[1], first[1]) and torch.equal(g[2], first[2])
            for g in grads[1:]
        ),
        "dK or dV differ between identical runs",
    )
    # dQ also varies with the order of its fp32 reduction
    ctx.metric(
        "dq_spread", max(rel_l2(g[0], first[0]) for g in grads[1:]), 1e-2, gate=False
    )


BF16_TOL = {"out": 1e-2, "lse": 1e-2, "dq": 2e-2, "dk": 2e-2, "dv": 2e-2}


def _fa2_grads(q, k, v, dout, causal):
    rep = q.shape[1] // k.shape[1]
    q, k, v = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(rep, 1),
            v.repeat_interleave(rep, 1),
            is_causal=causal,
        )
    out.backward(dout)
    return q.grad, k.grad, v.grad


def _bf16_grad(ctx, q, k, v, dout, causal, bwd_cfg=None):
    Nq, Nk = q.shape[2], k.shape[2]
    q, k, v, dout = [x.to(torch.bfloat16) for x in (q, k, v, dout)]
    out, lse, pmax = attn_fwd(q, k, v, causal=causal)
    dq, dk, dv = attn_bwd(
        q, k, v, out, lse, pmax, dout, causal=causal, **(bwd_cfg or {})
    )
    qf, kf, vf = [x.float().requires_grad_() for x in (q, k, v)]
    out_ref, lse_ref = reference(qf, kf, vf, causal)
    out_ref.backward(dout.float())
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
    for name, got, want in (
        ("dq", dq, qf.grad),
        ("dk", dk, kf.grad),
        ("dv", dv, vf.grad),
    ):
        ctx.metric(name, rel_l2(got, want), BF16_TOL[name])
        ctx.expect(not torch.isnan(got).any().item(), f"{name} contains NaN")
    ctx.metric("dq_p99", row_p99(dq, qf.grad), gate=True)
    ctx.metric("dk_p99", row_p99(dk, kf.grad), gate=True)
    # SDPA's causal mask is top-left aligned, so it only matches when Nq == Nk
    if not causal or Nq == Nk:
        try:
            fa2 = _fa2_grads(q, k, v, dout, causal)
        except (
            RuntimeError
        ):  # PyTorch builds without FlashAttention, such as on Windows
            ctx.note("FA2 unavailable")
            return
        errs = [rel_l2(g, w) for g, w in zip(fa2, (qf.grad, kf.grad, vf.grad))]
        ctx.note("FA2 dq {:.2e} dk {:.2e} dv {:.2e}".format(*errs))


def _bf16_run(ctx, shape, causal, bwd_cfg=None):
    B, Hq, Hkv, Nq, Nk, d = shape
    q, k, v = randn_qkv(B, Hq, Hkv, Nq, Nk, d)
    g = torch.Generator(device="cuda").manual_seed(7)
    dout = torch.randn(B, Hq, Nq, d, device="cuda", generator=g)
    _bf16_grad(ctx, q, k, v, dout, causal, bwd_cfg)


@case("grad_bf16", shape=SHAPES, causal=[False, True])
def grad_bf16(ctx, shape, causal):
    _bf16_run(ctx, shape, causal)


BF16_CANDIDATES = candidates_for(device_arch(), True)
# the deterministic backward is SM120 only
DETERMINISTIC = [{"deterministic": True}] if device_arch() == 120 else []


@case(
    "grad_bf16_cfg",
    where=lambda shape, bwd_cfg, **_: (
        bwd_cfg in BF16_CANDIDATES[shape[5]] + DETERMINISTIC
    ),
    shape=[(2, 4, 2, 1000, 1000, 128), (1, 4, 4, 2048, 2048, 64)],
    causal=[False, True],
    bwd_cfg=BF16_CANDIDATES[128] + BF16_CANDIDATES[64] + DETERMINISTIC,
)
def grad_bf16_cfg(ctx, shape, causal, bwd_cfg):
    _bf16_run(ctx, shape, causal, bwd_cfg)


@case(
    "grad_bf16_hard",
    shape=HARD_SHAPES,
    causal=[False, True],
    bwd_cfg=[{}] + DETERMINISTIC,
)
def grad_bf16_hard(ctx, shape, causal, bwd_cfg):
    _bf16_grad(ctx, *hard_inputs(*shape), causal, bwd_cfg)
