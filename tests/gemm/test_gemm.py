import sys
from typing import Any, cast

import torch

import warpy_kernels as wk
from tests.fixtures import rel_l2
from tests.registry import case
from warpy_kernels import utils
from warpy_kernels.sm120 import ptx

GEMM = sys.modules["warpy_kernels.gemm.sm120_gemm"]  # `warpy_kernels.gemm` is shadowed by the function
FP8 = torch.float8_e4m3fn
WS = {"tma": True, "warp_spec": True, "persistent": True}

CONFIGS: dict[str, dict[str, Any]] = {
    "cp.async-bn128": {"bk": 32, "stages": 3},
    "cp.async-bn256": {"bn": 256, "bk": 32, "stages": 3},
    "ws-bn256": {"bn": 256, "bk": 32, "stages": 4, "setmaxnreg": True, "frag_pipe": True, **WS},
}
FP8_CONFIGS: dict[str, dict[str, Any]] = {
    "cp.async-bn128": {"bk": 128, "stages": 2},
    "ws-bn128": {"bn": 128, "bk": 128, "stages": 3, **WS},
    "ws-bn256": {"bn": 256, "bk": 128, "stages": 2, "setmaxnreg": True, "frag_pipe": True, **WS},
}
# the small shape is all stream-K, the large one ends with a partial round
SHAPES = [(256, 512, 1024), (4096, 4096, 2048)]


def _mm(a: torch.Tensor, b: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    return cast(torch.Tensor, wk.gemm(a, b, **kwargs))


def _mm_gelu(a: torch.Tensor, b: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return cast(tuple[torch.Tensor, torch.Tensor], wk.gemm(a, b, act="gelu", store_pre=True, **kwargs))


def _dgelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1 + torch.erf(x * 0.7071067811865476)) + x * torch.exp(-0.5 * x * x) * 0.3989422804014327


@case("epilogue", config=list(CONFIGS), shape=SHAPES)
def epilogue(ctx, config, shape):
    m, n, k = shape
    cfg = CONFIGS[config]
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(m, k, device="cuda", generator=gen).bfloat16()
    b = torch.randn(n, k, device="cuda", generator=gen).bfloat16()
    c0 = torch.randn(m, n, device="cuda", generator=gen).bfloat16()
    prod = a.float() @ b.float().T
    x = prod + c0.float()
    tol = 1e-2  # bf16 rounding is about 4e-3
    ctx.metric("plain", rel_l2(_mm(a, b, **cfg).float(), prod), tol)
    ctx.metric("accum", rel_l2(_mm(a, b, out=c0.clone(), accum=True, **cfg).float(), x), tol)
    post, pre = _mm_gelu(a, b, out=c0.clone(), accum=True, **cfg)
    ctx.metric("gelu", rel_l2(post.float(), torch.nn.functional.gelu(x)), tol)
    ctx.metric("pre", rel_l2(pre.float(), x), tol)
    if "tma" not in cfg:  # `b_trans` is tested on `cp.async` only
        bt = b.T.contiguous()
        ctx.metric("b_trans", rel_l2(_mm(a, bt, b_trans=True, **cfg).float(), prod), tol)
        d = _mm(a, bt, out=c0.clone(), act="dgelu", pre=pre, accum=True, b_trans=True, **cfg)
        ctx.metric("dgelu", rel_l2(d.float(), x * _dgelu(pre.float())), tol)
    ctx.expect(torch.equal(_mm(a, b, **cfg), _mm(a, b, **cfg)), "repeated launches differ")


@case(
    "matches_cublas",
    where=lambda dtype, config, shape: config in (FP8_CONFIGS if dtype == "fp8" else CONFIGS),
    dtype=["fp8", "bf16"],
    config=sorted({*CONFIGS, *FP8_CONFIGS}),
    shape=SHAPES,
)
def matches_cublas(ctx, dtype, config, shape):
    m, n, k = shape
    a = torch.randn(m, k, device="cuda")
    b = torch.randn(n, k, device="cuda")
    if dtype == "fp8":
        a, b = a.to(FP8), b.to(FP8)
        one = torch.ones((), device="cuda")
        want = torch._scaled_mm(a, b.T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)  # pyright: ignore[reportPrivateImportUsage]
        cfg = FP8_CONFIGS[config]
    else:
        a, b = a.bfloat16(), b.bfloat16()
        want = a @ b.T
        cfg = CONFIGS[config]
    ctx.metric("rel_l2", rel_l2(_mm(a, b, **cfg).float(), want.float()), 1e-3)


def _wide(r: int, c: int, gen: torch.Generator) -> torch.Tensor:
    e = torch.rand(r, c, device="cuda", generator=gen) * 17.8 - 9.0  # e4m3 subnormals to near 448
    s = torch.randint(0, 2, (r, c), device="cuda", generator=gen) * 2 - 1
    return s * torch.exp2(e)


def _fp8_inputs(kind: str, m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(0)
    if kind == "normal":
        a = torch.randn(m, k, device="cuda", generator=gen)
        b = torch.randn(n, k, device="cuda", generator=gen)
    elif kind == "wide":
        a, b = _wide(m, k, gen), _wide(n, k, gen)
    else:
        a = torch.randn(m, k // 2, device="cuda", generator=gen) * 16
        b = torch.randn(n, k // 2, device="cuda", generator=gen) * 16
        a, b = torch.cat([a, -a], 1), torch.cat([b, b], 1)  # exact results are 0
    return a.to(FP8), b.to(FP8)


def _with_mma(mma: Any, fn: Any) -> Any:
    setattr(GEMM, "mma_mxf8", mma)  # noqa: B010
    utils._compile_cache.clear()  # the cache key ignores the MMA
    try:
        return fn()
    finally:
        setattr(GEMM, "mma_mxf8", ptx.mma_mxf8)  # noqa: B010
        utils._compile_cache.clear()


@case("fp8_block_scaled_mma", config=list(FP8_CONFIGS), kind=["normal", "wide", "cancel"])
def fp8_block_scaled_mma(ctx, config, kind):
    m, n, k = 512, 512, 4096
    cfg = FP8_CONFIGS[config]
    a, b = _fp8_inputs(kind, m, n, k)
    ad, bd = a.double(), b.double()
    ref = ad @ bd.T
    # subtracting bf16(ref) keeps the low accumulator bits in the bf16 output
    c0 = (-ref).to(torch.bfloat16)
    run = lambda: (_mm(a, b, **cfg), _mm(a, b, out=c0.clone(), accum=True, **cfg))
    new, new_resid = run()
    old, old_resid = _with_mma(ptx.mma_fp8, run)
    ctx.expect(torch.equal(new, old), "output differs from the plain e4m3 MMA")
    ctx.expect(torch.equal(new_resid, old_resid), "accumulator differs from the plain e4m3 MMA")
    err = (new_resid.double() - c0.double() - ref).abs() / ((ad.abs() @ bd.abs().T) * 2.0**-21).clamp_min(1e-300)
    ctx.metric("acc_err_units", err.max().item(), k / 32)  # bound for a truncating 22-bit accumulator
