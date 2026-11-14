import math

import torch

from tests.registry import Skip
from warpy_kernels import attn_bwd, attn_fwd


class Ctx:
    def __init__(self):
        self.metrics = []
        self.notes = []
        self.data = {}
        self.failures = []
        self.gated = set()

    def metric(self, name, value, tol=None, gate=None):
        """Record ``value``; the test fails if ``tol`` is given and ``value`` is NaN or above it.

        With ``--baseline``, a gated metric also fails when it is worse than the baseline run by more than
        ``--slack``. Metrics with a ``tol`` are gated unless ``gate`` is ``False``.
        """
        self.metrics.append((name, value, tol))
        if gate if gate is not None else tol is not None:
            self.gated.add(name)
        if tol is not None and not value <= tol:
            self.failures.append(f"{name} = {value:.3e} exceeds {tol:.1e}")

    def expect(self, cond, message):
        """Fail the test with ``message`` unless ``cond`` holds."""
        if not cond:
            self.failures.append(message)

    def note(self, text):
        """Print ``text`` under the test's result line."""
        self.notes.append(text)

    def skip(self, reason):
        raise Skip(reason)


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def row_p99(a, b):
    """99th percentile over rows of the per-row relative L2 error of ``a`` against ``b``.

    Rows far below the typical row norm are measured against a tenth of the median row norm, so near-zero
    reference rows don't dominate.
    """
    a, b = a.float().flatten(0, -2), b.float().flatten(0, -2)
    ref = b.norm(dim=1)
    floor = 0.1 * ref.median().clamp_min(1e-30)
    return ((a - b).norm(dim=1) / torch.maximum(ref, floor)).quantile(0.99).item()


def hard_inputs(B, Hq, Hkv, Nq, Nk, d, seed=0, logit_std=4.0, dout_spread=1.5):
    """Peaky-softmax inputs with logits of standard deviation ``logit_std`` and dO row norms spread as ``exp(N(0, s))``.

    Returns:
        ``(q, k, v, dout)`` in float32.
    """
    q, k, v = randn_qkv(B, Hq, Hkv, Nq, Nk, d, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1000)
    dout = torch.randn(B, Hq, Nq, d, device="cuda", generator=g)
    dout = dout * torch.exp(dout_spread * torch.randn(B, Hq, Nq, 1, device="cuda", generator=g))
    s = math.sqrt(logit_std)
    return q * s, k * s, v, dout


def randn_qkv(B, Hq, Hkv, Nq, Nk, d, seed=0):
    """Random float32 ``q, k, v`` on the GPU, with ``k`` and ``v`` centered over the sequence."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, Hq, Nq, d, device="cuda", generator=g)
    k = torch.randn(B, Hkv, Nk, d, device="cuda", generator=g)
    v = torch.randn(B, Hkv, Nk, d, device="cuda", generator=g)
    return q, k - k.mean(dim=2, keepdim=True), v - v.mean(dim=2, keepdim=True)


def sink_qkv(Nq, Nk, d, gap_nats, seed=1):
    """Float32 inputs with one key ``gap_nats`` above a long near-uniform tail."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = 1.0 + 0.1 * torch.randn(1, 1, Nq, d, device="cuda", generator=g)
    k = 0.01 * torch.randn(1, 1, Nk, d, device="cuda", generator=g)
    k[:, :, 0, :] = gap_nats / math.sqrt(d)
    v = torch.randn(1, 1, Nk, d, device="cuda", generator=g)
    return q, k, v


def _quantize(x, group_dims):
    scale = x.abs().amax(dim=group_dims, keepdim=True).clamp_min(1e-12) / 448.0
    return (x / scale).to(torch.float8_e4m3fn), scale


def quantize_qkv(q, k, v):
    """E4M3 ``q, k, v`` with one scale per (batch, kv head); queries share the scale of their kv group.

    Returns:
        ``(q8, k8, v8, q_descale, k_descale, v_descale)``, the descales shaped ``[B, Hkv]``.
    """
    B, Hq, Nq, d = q.shape
    Hkv = k.shape[1]
    q8, sq = _quantize(q.view(B, Hkv, Hq // Hkv, Nq, d), (2, 3, 4))
    k8, sk = _quantize(k, (2, 3))
    v8, sv = _quantize(v, (2, 3))
    return (
        q8.view(B, Hq, Nq, d),
        k8,
        v8,
        sq.view(B, Hkv),
        sk.view(B, Hkv),
        sv.view(B, Hkv),
    )


def dequantize_qkv(q8, k8, v8, sq, sk, sv):
    rep = q8.shape[1] // k8.shape[1]
    q = q8.float() * sq.repeat_interleave(rep, 1)[:, :, None, None]
    return q, k8.float() * sk[:, :, None, None], v8.float() * sv[:, :, None, None]


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, fwd_cfg, bwd_cfg):
        q8, k8, v8, sq, sk, sv = quantize_qkv(q, k, v)
        out, lse, pmax = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, **fwd_cfg)
        ctx.save_for_backward(q8, k8, v8, sq, sk, sv, out, lse, pmax)
        ctx.causal, ctx.bwd_cfg, ctx.dtype = causal, bwd_cfg, q.dtype
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _):  # type: ignore
        q8, k8, v8, sq, sk, sv, out, lse, pmax = ctx.saved_tensors
        out, lse = (
            out.detach(),
            lse.detach(),
        )  # saved outputs carry autograd history, which dlpack rejects
        dq, dk, dv = attn_bwd(
            q8,
            k8,
            v8,
            out,
            lse,
            pmax,
            dout.contiguous().to(torch.bfloat16),
            sq,
            sk,
            sv,
            causal=ctx.causal,
            **ctx.bwd_cfg,
        )
        return dq.to(ctx.dtype), dk.to(ctx.dtype), dv.to(ctx.dtype), None, None, None


def attention(q, k, v, causal=False, fwd_cfg=None, bwd_cfg=None):
    """Differentiable FP8 attention on float tensors: quantize, run the kernels, pass gradients straight through.

    Returns:
        ``(out, lse)``, with ``out`` in bf16 and ``lse`` in base 2.
    """
    return _Attention.apply(q, k, v, causal, fwd_cfg or {}, bwd_cfg or {})


def reference(q, k, v, causal):
    """fp32 attention on ``q, k, v`` that is differentiable; rows that see no key give ``O = 0`` and ``lse = -inf``.

    Returns:
        ``(out, lse)`` with ``lse`` in base 2.
    """
    _, Hq, Nq, d = q.shape
    Nk = k.shape[2]
    rep = Hq // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    visible = torch.ones(Nq, dtype=torch.bool, device=q.device)
    if causal:
        row = torch.arange(Nq, device=q.device)[:, None]
        col = torch.arange(Nk, device=q.device)[None, :]
        s = s.masked_fill(col > row + Nk - Nq, -1e30)
        visible = (torch.arange(Nq, device=q.device) + Nk - Nq) >= 0
    out = (torch.softmax(s, dim=-1) @ v) * visible[:, None]
    lse = torch.where(visible, torch.logsumexp(s, dim=-1) / math.log(2.0), -math.inf)
    return out, lse


def _errors(q, k, v, dout, causal, fwd_cfg, bwd_cfg):
    Nq, Nk = q.shape[2], k.shape[2]
    q, k, v = [x.detach().clone().requires_grad_(dout is not None) for x in (q, k, v)]
    out, lse = attention(q, k, v, causal, fwd_cfg, bwd_cfg)
    qd, kd, vd = [
        x.detach().requires_grad_(dout is not None)
        for x in dequantize_qkv(*quantize_qkv(q, k, v))
    ]
    out_ref, lse_ref = reference(qd, kd, vd, causal)
    if dout is not None:
        dout = dout.to(torch.bfloat16)
        out.backward(dout)
        out_ref.backward(dout.float())
    visible = torch.ones(Nq, dtype=torch.bool, device=q.device)
    if causal:
        visible = (torch.arange(Nq, device=q.device) + Nk - Nq) >= 0
    errs = {
        "out": rel_l2(out[:, :, visible], out_ref[:, :, visible]),
        "lse": (lse[:, :, visible] - lse_ref[:, :, visible]).abs().max().item(),
    }
    checks = [
        (bool((out[:, :, ~visible] == 0).all()), "rows without a visible key must give O = 0"),
        (not torch.isfinite(lse[:, :, ~visible]).any().item(), "rows without a visible key must give a non-finite lse"),
    ]
    if dout is not None:
        for name, got, want in (("dq", q.grad, qd.grad), ("dk", k.grad, kd.grad), ("dv", v.grad, vd.grad)):
            errs[name] = rel_l2(got, want)
            checks.append((not torch.isnan(got).any().item(), f"{name} contains NaN"))
        errs["dq_p99"] = row_p99(q.grad, qd.grad)
        errs["dk_p99"] = row_p99(k.grad, kd.grad)
    return errs, checks


def _record(ctx, errs, checks, tol):
    for name, value in errs.items():
        # worst-row errors have no fixed tolerance but are still checked against a baseline run
        ctx.metric(name, value, tol.get(name), gate=True if name.endswith("_p99") else None)
    for cond, message in checks:
        ctx.expect(cond, message)


def compare(ctx, q, k, v, dout, causal, tol, fwd_cfg=None, bwd_cfg=None):
    """Run the kernels and the reference on ``q, k, v`` and record the errors of ``out``, ``lse`` and the gradients.

    Rows that see no key must give ``O = 0`` and a non-finite ``lse``; the other rows are compared to the reference.
    With ``dout``, the worst-row errors ``dq_p99`` and ``dk_p99`` are recorded too. See ``row_p99``.

    Args:
        ctx: Test context.
        q: Float32 queries ``[B, Hq, Nq, d]``.
        k: Float32 keys ``[B, Hkv, Nk, d]``.
        v: Float32 values ``[B, Hkv, Nk, d]``.
        dout: Gradient of the output, or ``None`` for a forward-only comparison.
        causal: Bottom-right aligned causal mask.
        tol: Tolerances by metric name (``out``, ``lse``, ``dq``, ``dk``, ``dv``, ``dq_p99``, ``dk_p99``); missing
            names are recorded without a tolerance.
        fwd_cfg: Forward kernel options.
        bwd_cfg: Backward kernel options.
    """
    _record(ctx, *_errors(q, k, v, dout, causal, fwd_cfg, bwd_cfg), tol)


def compare_seeds(ctx, make_inputs, seeds, causal, tol, fwd_cfg=None, bwd_cfg=None):
    """Like ``compare``, but record each error as its mean over ``seeds``.

    Args:
        ctx: Test context.
        make_inputs: Called with a seed; returns ``(q, k, v, dout)``.
        seeds: Seeds to average over.
        causal: Bottom-right aligned causal mask.
        tol: Tolerances by metric name, applied to the means.
        fwd_cfg: Forward kernel options.
        bwd_cfg: Backward kernel options.
    """
    runs = [_errors(*make_inputs(seed), causal, fwd_cfg, bwd_cfg) for seed in seeds] # type: ignore
    errs = {name: sum(r[0][name] for r in runs) / len(runs) for name in runs[0][0]}
    _record(ctx, errs, [c for r in runs for c in r[1]], tol)
