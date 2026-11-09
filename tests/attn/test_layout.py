import torch

from tests.fixtures import quantize_qkv, randn_qkv
from tests.registry import case
from warpy_kernels import attn_fwd

SHAPES = [(2, 4, 4, 1024, 1024, 128), (1, 2, 2, 1000, 777, 64)]
SELF_SHAPES = [(2, 4, 4, 1024, 1024, 128), (1, 2, 2, 1000, 1000, 64)]


def _bshd(x):
    return x.transpose(1, 2).contiguous().transpose(1, 2)


def _inputs(shape):
    B, Hq, Hkv, Nq, Nk, d = shape
    return quantize_qkv(*randn_qkv(B, Hq, Hkv, Nq, Nk, d))


def _same(ctx, got, want, what):
    ctx.expect(torch.equal(got, want), what)


@case("layout_bshd", shape=SHAPES, causal=[False, True], tma=[False, True])
def layout_bshd(ctx, shape, causal, tma):
    q8, k8, v8, sq, sk, sv = _inputs(shape)
    ref, lse_ref, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, tma=tma)
    out, lse, _ = attn_fwd(
        _bshd(q8), _bshd(k8), _bshd(v8), sq, sk, sv, causal=causal, tma=tma
    )
    ctx.expect(
        out.transpose(1, 2).is_contiguous(),
        "output should keep the [B, N, H, d] memory layout",
    )
    _same(ctx, out, ref, "out differs from the contiguous layout")
    ctx.expect(torch.equal(lse, lse_ref), "lse differs from the contiguous layout")


@case("layout_fused_qkv", shape=SELF_SHAPES, causal=[False, True], tma=[False, True])
def layout_fused_qkv(ctx, shape, causal, tma):
    q8, k8, v8, sq, sk, sv = _inputs(shape)
    ref, _, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, tma=tma)
    parts = [x.transpose(1, 2).contiguous().view(torch.uint8) for x in (q8, k8, v8)]
    qkv = torch.stack(parts, dim=2).view(torch.float8_e4m3fn)
    q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
    out, _, _ = attn_fwd(q, k, v, sq, sk, sv, causal=causal, tma=tma)
    _same(ctx, out, ref, "out differs from the contiguous layout")


@case("no_lse", shape=SHAPES, causal=[False, True])
def no_lse(ctx, shape, causal):
    q8, k8, v8, sq, sk, sv = _inputs(shape)
    ref, _, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal)
    out, lse, pmax = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, return_lse=False)
    ctx.expect(lse is None and pmax is None, "lse and pmax should be None")
    _same(ctx, out, ref, "out differs when lse is skipped")


@case("out_arg", shape=SHAPES, causal=[False, True])
def out_arg(ctx, shape, causal):
    q8, k8, v8, sq, sk, sv = _inputs(shape)
    B, Hq, _, Nq, _, d = shape
    ref, _, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal)
    buf = torch.zeros(B, Nq, Hq, d, dtype=torch.bfloat16, device="cuda")
    out, _, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, out=buf.transpose(1, 2))
    ctx.expect(out.data_ptr() == buf.data_ptr(), "the given out buffer should be used")
    _same(ctx, out, ref, "out differs from the allocated buffer")


@case("layout_qkv_v_view", shape=SELF_SHAPES, causal=[False, True], tma=[False, True])
def layout_qkv_v_view(ctx, shape, causal, tma):
    q8, k8, v8, sq, sk, sv = _inputs(shape)
    ref, _, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, tma=tma)
    B, H, N, d = v8.shape
    qkv = torch.zeros(B, N, 3, H, d, dtype=torch.uint8, device="cuda")
    qkv[:, :, 2] = v8.transpose(1, 2).view(torch.uint8)
    v = qkv[:, :, 2].view(torch.float8_e4m3fn).transpose(1, 2)
    out, _, _ = attn_fwd(_bshd(q8), _bshd(k8), v, sq, sk, sv, causal=causal, tma=tma)
    _same(ctx, out, ref, "out differs from the contiguous layout")
