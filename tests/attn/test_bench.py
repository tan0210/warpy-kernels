import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.fixtures import quantize_qkv, randn_qkv
from tests.registry import bench
from warpy_kernels import attn_fwd
from warpy_kernels.attn import autotune_attn_bwd, autotune_attn_fwd, make_attn_fwd
from warpy_kernels.utils import time_launch_ms

SEQS = [1024, 2048, 4096, 8192, 16384]
BASELINE = "FA2 (bf16)"
OURS = "WarpAttention (FP8)"


def _fa2(q, k, v, causal):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _record(ctx, flops, ms, ms_ref, best):
    ctx.metric(f"tf:{BASELINE}", flops / ms_ref / 1e9)
    ctx.metric(f"tf:{OURS}", flops / ms / 1e9)
    ctx.note(f"{ms_ref / ms:.2f}x {BASELINE}")
    ctx.data["config"] = best


@bench("fwd", d=[64, 128], causal=[False, True], n=SEQS)
def bench_fwd(ctx, d, causal, n, B=1, H=32):
    flops = 4 * B * H * n * n * d * (0.5 if causal else 1.0)
    args = quantize_qkv(*randn_qkv(B, H, H, n, n, d))
    best, results = autotune_attn_fwd(*args, causal=causal, iters=20, verbose=False)
    ms = min(t for _, t in results if not isinstance(t, str))
    qb, kb, vb = [torch.randn(B, H, n, d, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    ms_ref = time_launch_ms(lambda: _fa2(qb, kb, vb, causal), 20)
    _record(ctx, flops, ms, ms_ref, best)


@bench("bwd", d=[64, 128], causal=[False, True], n=SEQS)
def bench_bwd(ctx, d, causal, n, B=1, H=32):
    flops = 10 * B * H * n * n * d * (0.5 if causal else 1.0)
    q8, k8, v8, sq, sk, sv = quantize_qkv(*randn_qkv(B, H, H, n, n, d))
    out, lse, pmax = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal)
    dout = torch.randn(B, H, n, d, device="cuda", dtype=torch.bfloat16)
    best, results = autotune_attn_bwd(q8, k8, v8, out, lse, pmax, dout, sq, sk, sv, causal=causal, iters=10,
                                          verbose=False)
    ms = min(t for _, t in results if not isinstance(t, str))
    q, k, v = [torch.randn(B, H, n, d, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    o = _fa2(q, k, v, causal)
    ms_ref = time_launch_ms(lambda: o.backward(dout, retain_graph=True), 10)
    _record(ctx, flops, ms, ms_ref, best)


@bench("cross", nq=[1024, 4096, 16384], nk=[128, 256, 512, 1024])
def bench_cross(ctx, nq, nk, B=2, H=16, d=128):
    flops = 4 * B * H * nq * nk * d
    args = quantize_qkv(*randn_qkv(B, H, H, nq, nk, d))
    best, results = autotune_attn_fwd(*args, iters=20, verbose=False)
    ms = min(t for _, t in results if not isinstance(t, str))
    q, k, v = [torch.randn(B, H, n, d, device="cuda", dtype=torch.bfloat16) for n in (nq, nk, nk)]
    ms_ref = time_launch_ms(lambda: _fa2(q, k, v, False), 20)
    _record(ctx, flops, ms, ms_ref, best)


@bench("layout", n=[2048, 4096, 8192])
def bench_layout(ctx, n, B=2, H=16, d=128):
    flops = 4 * B * H * n * n * d
    q8, k8, v8, sq, sk, sv = quantize_qkv(*randn_qkv(B, H, H, n, n, d))
    autotune_attn_fwd(q8, k8, v8, sq, sk, sv, iters=20, verbose=False)
    launch, *_ = make_attn_fwd(q8, k8, v8, sq, sk, sv)
    ms_flat = time_launch_ms(launch, 50)
    qv, kv = [x.transpose(1, 2).contiguous().transpose(1, 2) for x in (q8, k8)]
    fused = torch.zeros(B, n, 3, H, d, dtype=torch.uint8, device="cuda")
    fused[:, :, 2] = v8.transpose(1, 2).view(torch.uint8)
    vv = fused[:, :, 2].view(torch.float8_e4m3fn).transpose(1, 2)
    launch_view, *_ = make_attn_fwd(qv, kv, vv, sq, sk, sv)
    ms_view = time_launch_ms(launch_view, 50)
    bufs = [torch.empty(x.shape, dtype=x.dtype, device="cuda") for x in (q8, k8, v8)]
    launch_copy, out, *_ = make_attn_fwd(*bufs, sq, sk, sv) # type: ignore

    def with_copies():
        for buf, x in zip(bufs, (qv, kv, vv)):
            buf.copy_(x)
        launch_copy()
        out.transpose(1, 2).contiguous()

    ms_copy = time_launch_ms(with_copies, 50)
    for label, ms in (("contiguous [B, H, N, d]", ms_flat), ("[B, N, H, d] view", ms_view),
                      ("[B, N, H, d] with copies", ms_copy)):
        ctx.metric(f"tf:{label}", flops / ms / 1e9)


@bench("lse", n=[2048, 4096, 8192])
def bench_lse(ctx, n, B=2, H=16, d=128):
    flops = 4 * B * H * n * n * d
    args = quantize_qkv(*randn_qkv(B, H, H, n, n, d))
    autotune_attn_fwd(*args, iters=20, verbose=False)
    for label, flag in (("return_lse=True", True), ("return_lse=False", False)):
        launch, *_ = make_attn_fwd(*args, return_lse=flag)
        ctx.metric(f"tf:{label}", flops / time_launch_ms(launch, 50) / 1e9)
