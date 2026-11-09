import math

import torch

from tests.fixtures import quantize_qkv, randn_qkv, rel_l2
from tests.registry import case
from warpy_kernels import attn_fwd

CFG = {"pipelined": False, "num_stages": 2, "block_n": 64}
SHAPES = [(2, 4, 2, 512, 1024, 128), (2, 2, 2, 300, 777, 64)]


@case("seqused_matches_truncated", shape=SHAPES, causal=[False, True], tma=[False, True])
def seqused_matches_truncated(ctx, shape, causal, tma):
    B, Hq, Hkv, Nq, Nk, d = shape
    q8, k8, v8, sq, sk, sv = quantize_qkv(*randn_qkv(B, Hq, Hkv, Nq, Nk, d))
    lens = torch.tensor([Nk, Nk // 3 + 5], dtype=torch.int32, device="cuda")
    out, lse, _ = attn_fwd(q8, k8, v8, sq, sk, sv, causal=causal, seqused_k=lens, tma=tma, **CFG)
    for b, n in enumerate(lens.tolist()):
        sl = slice(b, b + 1)
        ref, lse_ref, _ = attn_fwd(q8[sl], k8[sl, :, :n], v8[sl, :, :n], sq[sl], sk[sl], sv[sl], causal=causal,
                                   tma=tma, **CFG)
        ctx.expect(torch.equal(out[sl], ref), f"out of batch item {b} differs from truncated keys")
        ctx.expect(torch.equal(lse[sl], lse_ref), f"lse of batch item {b} differs from truncated keys")


@case("lse_natural", shape=SHAPES, causal=[False, True])
def lse_natural(ctx, shape, causal):
    B, Hq, Hkv, Nq, Nk, d = shape
    args = quantize_qkv(*randn_qkv(B, Hq, Hkv, Nq, Nk, d))
    _, lse2, _ = attn_fwd(*args, causal=causal)
    _, lse_e, _ = attn_fwd(*args, causal=causal, lse_natural=True)
    ctx.expect(torch.allclose(lse_e, lse2 * math.log(2.0), rtol=1e-6, atol=0), "lse is not base-2 lse times ln 2")


@case("trimmed_context", d=[64, 128], tma=[False, True])
def trimmed_context(ctx, d, tma):
    B, H, Nq, S = 2, 4, 1024, 512
    g = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(B, H, Nq, d, device="cuda", generator=g)
    k = torch.randn(B, H, S, d, device="cuda", generator=g)
    v = torch.randn(B, H, S, d, device="cuda", generator=g)
    lens = torch.tensor([77, 300], dtype=torch.int32, device="cuda")
    keep = (int(lens.max()) + 7) // 8 * 8
    pad = torch.arange(S, device="cuda")[None, :] >= lens[:, None]
    k, v = k.masked_fill(pad[:, None, :, None], 0), v.masked_fill(pad[:, None, :, None], 0)
    q8, k8, v8, sq, sk, sv = quantize_qkv(q, k, v)
    full, _, _ = attn_fwd(q8, k8, v8, sq, sk, sv, tma=tma, **CFG)
    trim, lse, _ = attn_fwd(q8, k8[:, :, :keep], v8[:, :, :keep], sq, sk, sv, seqused_k=lens, lse_natural=True,
                            tma=tma, **CFG)
    ln_pad = torch.log((S - lens).float())[:, None, None]
    gain = 1.0 / (1.0 + torch.exp(ln_pad - lse))
    ctx.metric("out", rel_l2(trim.float() * gain[..., None], full), 6e-2)
