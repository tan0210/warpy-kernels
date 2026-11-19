# type: ignore

import math

import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64

from ..arch import require
from ..utils import (
    Arg,
    Scalar,
    autotune,
    cached_tensor,
    check_descale,
    compile_cached,
    flat,
    round_up,
    tune_key,
    tune_lookup,
)
from .fwd_base import AttnFwdBase
from .fwd_sm80 import AttnFwdSm80
from .fwd_sm89 import AttnFwdSm89
from .fwd_sm120 import AttnFwdSm120

_CLASSES = {80: AttnFwdSm80, 86: AttnFwdSm80, 89: AttnFwdSm89, 120: AttnFwdSm120}


FWD_CANDIDATES = {
    128: [
        {"pipelined": False, "num_stages": 2, "block_n": 128},
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 3, "block_n": 64},
    ],
    64: [
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 3, "block_n": 128},
        {"pipelined": True, "num_stages": 3, "block_n": 128},
        {"pipelined": False, "num_stages": 2, "block_n": 64},
    ],
}
FWD_DEFAULTS = {
    128: {"pipelined": False, "num_stages": 2, "block_n": 128, "tma": True},
    64: {"pipelined": True, "num_stages": 3, "block_n": 128, "tma": True},
}
# bf16 rows are twice as wide, so fewer and smaller K/V stages fit
FWD_CANDIDATES_BF16 = {
    128: [
        {"pipelined": False, "num_stages": 2, "block_n": 64},
        {"pipelined": True, "num_stages": 3, "block_n": 32},
        {"pipelined": True, "num_stages": 4, "block_n": 32},
    ],
    64: [
        {"pipelined": True, "num_stages": 3, "block_n": 64},
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 2, "block_n": 128},
        {"pipelined": False, "num_stages": 2, "block_n": 64},
        {"pipelined": True, "num_stages": 4, "block_n": 32},
    ],
}
FWD_DEFAULTS_BF16 = {
    128: {"pipelined": False, "num_stages": 2, "block_n": 64},
    64: {"pipelined": True, "num_stages": 3, "block_n": 64},
}
# 64-query CTAs within ~49 KB of shared memory run two per SM below SM120
_BM64_BF16 = {
    128: [{"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 32}],
    64: [
        {"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 64},
        {"block_m": 64, "pipelined": True, "num_stages": 3, "block_n": 32},
        {"block_m": 64, "pipelined": True, "num_stages": 4, "block_n": 32},
        {"warp_rows": 32, "pipelined": False, "num_stages": 3, "block_n": 32},
        {"warp_rows": 32, "pipelined": False, "num_stages": 2, "block_n": 32},
        {"ctas_per_sm": 2, "pipelined": False, "num_stages": 2, "block_n": 32},
        {"ctas_per_sm": 2, "pipelined": False, "num_stages": 2, "block_n": 64},
        {"ctas_per_sm": 2, "pipelined": False, "num_stages": 2, "block_n": 32, "split_kv": True},
    ],
}
FWD_CANDIDATES_BF16_SM89 = {
    d: FWD_CANDIDATES_BF16[d] + _BM64_BF16[d] for d in (64, 128)
}
# e4m3 on SM89, with `cp.async` loads only
FWD_CANDIDATES_SM89 = {
    128: [
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": True, "num_stages": 5, "block_n": 64},
        {"pipelined": False, "num_stages": 4, "block_n": 64},
        {"pipelined": True, "num_stages": 6, "block_n": 32},
        {"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 64},
        {"block_m": 64, "pipelined": False, "num_stages": 4, "block_n": 32},
        {"block_m": 64, "pipelined": True, "num_stages": 4, "block_n": 32},
    ],
    64: [
        {"pipelined": True, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 4, "block_n": 64},
        {"pipelined": False, "num_stages": 3, "block_n": 128},
        {"block_m": 64, "pipelined": False, "num_stages": 4, "block_n": 64},
        {"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 128},
        {"block_m": 64, "pipelined": True, "num_stages": 4, "block_n": 64},
    ],
}
# fastest on an RTX 4090 across n = 1024 to 8192
FWD_DEFAULTS_SM89 = {
    128: {"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 64},
    64: {"pipelined": False, "num_stages": 4, "block_n": 64},
}
FWD_DEFAULTS_BF16_SM89 = {
    128: {"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 32},
    64: {"ctas_per_sm": 2, "pipelined": False, "num_stages": 2, "block_n": 32, "split_kv": True},
}


# causal e4m3 d64 on SM89, 16-18% faster than the d64 default on an RTX 4070 Ti SUPER at n = 1024 to 8192
FWD_DEFAULTS_CAUSAL_SM89 = {
    **FWD_DEFAULTS_SM89,
    64: {"block_m": 64, "pipelined": False, "num_stages": 2, "block_n": 128},
}


def _defaults(arch, bf16, causal=False):
    if bf16:
        return FWD_DEFAULTS_BF16_SM89 if arch != 120 else FWD_DEFAULTS_BF16
    if arch == 120:
        return FWD_DEFAULTS
    return FWD_DEFAULTS_CAUSAL_SM89 if causal else FWD_DEFAULTS_SM89


def _kernel_view(x, n_pad):
    xb = x.view(torch.uint8)
    n = x.shape[2]
    direct = (
        n == n_pad
        and x.stride(3) == 1
        and x.data_ptr() % 16 == 0
        and all(st % 16 == 0 for st in x.stride()[:3])
    )
    return xb if direct else F.pad(xb, (0, 0, 0, n_pad - n)).contiguous()


def make_attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    num_stages=None,
    pipelined=None,
    block_n=None,
    block_m=None,
    warp_rows=None,
    ctas_per_sm=None,
    tile_local=True,
    lpt=None,
    tma=None,
    out=None,
    return_lse=True,
    seqused_k=None,
    lse_natural=False,
    warp_spec=None,
    split_kv=None,
):
    """Prepare an attention forward launch.

    ``launch()`` may not see later in-place changes to the inputs. Call ``make_attn_fwd`` again for new values.

    Args:
        q: ``[B, Hq, Nq, d]`` ``torch.float8_e4m3fn`` or ``torch.bfloat16``. ``d`` is 64 or 128. Any strides work if
            the last one is 1, such as a transposed ``[B, N, H, d]`` tensor. Inputs are copied if a stride or the
            base address is not a multiple of 16 bytes, or if ``Nq`` is not a multiple of 128.
        k: ``[B, Hkv, Nk, d]``, same dtype as ``q`` and the same stride rules. ``Hq`` must be a multiple of ``Hkv``.
            ``Nk`` must be a multiple of ``block_n`` unless ``tma`` is on, or it is copied.
        v: Same shape and dtype as ``k``.
        q_descale: ``[B, Hkv]`` float32 dequantization scale for e4m3 inputs, or ``None`` for 1. bf16 inputs take
            no descales.
        k_descale: Same as ``q_descale``, for ``k``.
        v_descale: Same as ``q_descale``, for ``v``.
        causal: Whether to apply a bottom-right aligned causal mask.
        softmax_scale: Scale on the logits. Defaults to ``1 / sqrt(d)``.
        num_stages: Depth of the K/V shared-memory pipeline, at least 3 when ``pipelined``. ``None`` takes the
            autotuned or built-in choice. See ``autotune_attn_fwd``.
        pipelined: Whether to issue the next tile's QK^T before the current tile's softmax. ``None`` takes the
            autotuned or built-in choice.
        block_n: Keys per tile, 32, 64 or 128. ``None`` takes the autotuned or built-in choice.
        block_m: Queries per CTA, 64 or 128. Always 128 on SM120. ``None`` takes the autotuned or built-in choice.
        warp_rows: Queries per warp, 16 or 32. 32 needs bf16 inputs and ``pipelined=False``. Always 16 on SM120.
            ``None`` takes the autotuned or built-in choice.
        ctas_per_sm: Limit registers so this many CTAs fit on one SM, or 0 for no limit. Ignored on SM120.
            ``None`` takes the autotuned or built-in choice.
        tile_local: Whether to quantize P against each tile's own row max, which keeps long-context tails
            accurate.
        lpt: Whether to launch the longest causal CTAs first. ``None`` enables it for causal attention only.
        tma: Whether to load K and V with TMA instead of ``cp.async``. Q always uses ``cp.async``. ``None``
            takes the autotuned or built-in choice. bf16 inputs always use ``cp.async``.
        out: Output buffer, ``[B, Hq, Nq, d]`` bf16 with last stride 1 and even other strides. ``None`` allocates
            one with the memory layout of ``q``.
        return_lse: Whether to compute ``lse`` and ``pmax``. The backward pass needs them.
        seqused_k: ``[B]`` int32 number of keys each batch item attends to, at most ``Nk``. ``None`` attends to all
            ``Nk``. Causal masking is aligned to this length. The backward pass does not support it.
        lse_natural: Whether ``lse`` is the natural-log log-sum-exp instead of base 2. The backward pass needs
            base 2.
        warp_spec: Whether a dedicated producer warp group issues the K/V TMA loads. Requires ``tma``. ``None``
            takes the autotuned choice, which is off by default.
        split_kv: Whether each tile's K and V load and wait separately. Needs ``pipelined=False``; ignored on
            SM120. ``None`` takes the autotuned or built-in choice.

    Returns:
        ``(launch, out, lse, pmax)``. Calling ``launch()`` runs the kernel and fills the outputs. ``out`` is
        ``[B, Hq, Nq, d]`` bf16, ``lse`` is the ``[B, Hq, Nq]`` base-2 log-sum-exp of the scaled logits
        (``+inf`` for rows with no visible key), and ``pmax`` is the ``[B, Hq, Nq]`` largest probability per row.
        ``lse`` and ``pmax`` are ``None`` if ``return_lse`` is false.

    Raises:
        ValueError: If ``q``, ``k`` or ``v`` has the wrong dtype or shape, a descale is not ``[B, Hkv]`` or is given
            for bf16 inputs, ``out`` has the wrong shape, dtype or strides, or ``seqused_k`` is not an int32 CUDA
            tensor of shape ``[B]``.
        NotImplementedError: If the GPU does not support the dtype of ``q``. See ``warpy_kernels.supports``.
    """
    B, Hq, Nq, d = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
    if not (
        q.dtype == k.dtype == v.dtype
        and q.dtype in (torch.float8_e4m3fn, torch.bfloat16)
    ):
        raise ValueError(
            "q, k and v must all be torch.float8_e4m3fn or all torch.bfloat16"
        )
    bf16 = q.dtype == torch.bfloat16
    arch = require("attn_fwd" if bf16 else "attn_fwd_fp8", q)
    if bf16 and not (q_descale is None and k_descale is None and v_descale is None):
        raise ValueError("descales apply to float8_e4m3fn inputs only")
    if (
        k.dim() != 4
        or v.shape != k.shape
        or k.shape[0] != B
        or k.shape[3] != d
        or d not in (64, 128)
        or Hq % Hkv
    ):
        raise ValueError(
            "need q [B, Hq, Nq, d] and k, v [B, Hkv, Nk, d] with d 64 or 128 and Hq a multiple of Hkv, "
            f"got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    for x, name in (
        (q_descale, "q_descale"),
        (k_descale, "k_descale"),
        (v_descale, "v_descale"),
    ):
        check_descale(x, B, Hkv, name)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    if (
        num_stages is None
        or pipelined is None
        or block_n is None
        or block_m is None
        or warp_rows is None
        or ctas_per_sm is None
        or tma is None
        or warp_spec is None
    ):
        tuned = (
            tune_lookup(
                tune_key(
                    "fwd_bf16" if bf16 else "fwd", d, causal, Nq, Nk, Hq // Hkv, arch
                )
            )
            or _defaults(arch, bf16, causal)[d]
        )
        num_stages = tuned["num_stages"] if num_stages is None else num_stages
        pipelined = tuned["pipelined"] if pipelined is None else pipelined
        block_n = tuned["block_n"] if block_n is None else block_n
        block_m = tuned.get("block_m", 128) if block_m is None else block_m
        warp_rows = tuned.get("warp_rows", 16) if warp_rows is None else warp_rows
        ctas_per_sm = (
            tuned.get("ctas_per_sm", 0) if ctas_per_sm is None else ctas_per_sm
        )
        tma = bool(tuned.get("tma", False)) if tma is None else tma
        warp_spec = (
            bool(tuned.get("warp_spec", False)) if warp_spec is None else warp_spec
        )
    tuned_cfg = (
        tune_lookup(tune_key("fwd_bf16" if bf16 else "fwd", d, causal, Nq, Nk, Hq // Hkv, arch))
        or _defaults(arch, bf16, causal)[d]
    )

    def tuned_get(key):
        return tuned_cfg.get(key, False)

    tma = bool(tma) and not bf16 and arch == 120
    warp_spec = bool(warp_spec) and tma
    if lpt is None:
        lpt = causal and arch == 120  # longest-first order measured slower on the 4090
    dev = q.device
    ones = cached_tensor(
        ("ones", dev, B, Hkv),
        lambda: torch.ones(B, Hkv, dtype=torch.float32, device=dev),
    )
    dq_, dk_, dv_ = [
        (x if x is not None else ones).float().contiguous()
        for x in (q_descale, k_descale, v_descale)
    ]

    nq_pad = round_up(Nq, AttnFwdBase.BM)
    nk_pad = round_up(Nk, block_n)
    qv = _kernel_view(q, nq_pad)
    kv = _kernel_view(k, Nk if tma else nk_pad)  # TMA zero-fills the rows past `Nk`
    vv = _kernel_view(v, Nk if tma else nk_pad)
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    elif (
        out.shape != q.shape
        or out.dtype != torch.bfloat16
        or out.device != dev
        or out.stride(3) != 1
        or any(st % 2 for st in out.stride()[:3])
        or out.data_ptr() % 4
    ):
        raise ValueError(
            "out must be a bf16 tensor shaped like q, with last stride 1 and even other strides"
        )
    if seqused_k is not None and (
        seqused_k.shape != (B,)
        or seqused_k.dtype != torch.int32
        or seqused_k.device != dev
    ):
        raise ValueError("seqused_k must be an int32 CUDA tensor of shape [B]")
    seq = (
        seqused_k.contiguous()
        if seqused_k is not None
        else cached_tensor(
            ("zero_i32", dev), lambda: torch.zeros(1, dtype=torch.int32, device=dev)
        )
    )
    if return_lse:
        lse = torch.empty(B, Hq, nq_pad, dtype=torch.float32, device=dev)
        pmax = torch.empty(B, Hq, nq_pad, dtype=torch.float32, device=dev)
    else:
        lse = pmax = torch.empty(1, dtype=torch.float32, device=dev)

    def dyn(x):
        return Arg(x, 16, True, 3)

    targs = [dyn(qv), dyn(kv), dyn(vv), dyn(out), flat(lse), flat(pmax)]
    targs += [Arg(x) for x in (dq_, dk_, dv_, seq)]
    strides = [Scalar(Int64, st) for x in (qv, kv, vv, out) for st in x.stride()[:3]]
    sargs = [
        Scalar(Float32, softmax_scale),
        *[Scalar(Int32, x) for x in (Nq, Nk, nq_pad, nk_pad, Hq, Hkv, B)],
    ] + strides

    cls = _CLASSES[arch]
    ctor = {
        "head_dim": d,
        "causal": causal,
        "num_stages": num_stages,
        "pipelined": pipelined,
        "block_n": block_n,
        "tile_local": tile_local,
        "lpt": lpt,
        "return_lse": return_lse,
        "has_seqused": seqused_k is not None,
        "lse_natural": lse_natural,
        "bf16": bf16,
        "arch": arch,
    }
    if arch == 120:
        ctor.update(tma=tma, warp_spec=warp_spec)
    else:
        o_vec = out.data_ptr() % 16 == 0 and all(st % 8 == 0 for st in out.stride()[:3])
        ctor.update(
            block_m=block_m,
            o_vec=o_vec,
            warp_rows=warp_rows,
            ctas_per_sm=ctas_per_sm,
            split_kv=bool(tuned_get("split_kv") if split_kv is None else split_kv) and not pipelined,
            # heaviest causal query blocks first within each head; measured faster for d128 only
            rev_m=causal and d == 128,
        )
    compiled = compile_cached(cls, ctor, (B, Hq, Hkv, nq_pad, nk_pad), *targs, *sargs)

    def launch():
        compiled(*targs, *sargs)

    if not return_lse:
        return launch, out, None, None
    return launch, out, lse[:, :, :Nq], pmax[:, :, :Nq]


def attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    num_stages=None,
    pipelined=None,
    block_n=None,
    block_m=None,
    warp_rows=None,
    ctas_per_sm=None,
    tile_local=True,
    lpt=None,
    tma=None,
    out=None,
    return_lse=True,
    seqused_k=None,
    lse_natural=False,
    warp_spec=None,
    split_kv=None,
):
    """Run the attention forward once.

    Args:
        q: As in ``make_attn_fwd``.
        k: As in ``make_attn_fwd``.
        v: As in ``make_attn_fwd``.
        q_descale: As in ``make_attn_fwd``.
        k_descale: As in ``make_attn_fwd``.
        v_descale: As in ``make_attn_fwd``.
        causal: As in ``make_attn_fwd``.
        softmax_scale: As in ``make_attn_fwd``.
        num_stages: As in ``make_attn_fwd``.
        pipelined: As in ``make_attn_fwd``.
        block_n: As in ``make_attn_fwd``.
        block_m: As in ``make_attn_fwd``.
        warp_rows: As in ``make_attn_fwd``.
        ctas_per_sm: As in ``make_attn_fwd``.
        tile_local: As in ``make_attn_fwd``.
        lpt: As in ``make_attn_fwd``.
        tma: As in ``make_attn_fwd``.
        out: As in ``make_attn_fwd``.
        return_lse: As in ``make_attn_fwd``.
        seqused_k: As in ``make_attn_fwd``.
        lse_natural: As in ``make_attn_fwd``.
        warp_spec: As in ``make_attn_fwd``.
        split_kv: As in ``make_attn_fwd``.

    Returns:
        ``(out, lse, pmax)``, as returned by ``make_attn_fwd`` after the launch.

    Raises:
        ValueError: If ``out`` or ``seqused_k`` is invalid. See ``make_attn_fwd``.
    """
    launch, out, lse, pmax = make_attn_fwd(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        causal,
        softmax_scale,
        num_stages,
        pipelined,
        block_n,
        block_m,
        warp_rows,
        ctas_per_sm,
        tile_local,
        lpt,
        tma,
        out,
        return_lse,
        seqused_k,
        lse_natural,
        warp_spec,
        split_kv,
    )
    launch()
    return out, lse, pmax


def autotune_attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    candidates=None,
    iters=10,
    verbose=True,
):
    """Time candidate forward configs for this shape and cache the fastest for later calls.

    Args:
        q: As in ``make_attn_fwd``.
        k: As in ``make_attn_fwd``.
        v: As in ``make_attn_fwd``.
        q_descale: As in ``make_attn_fwd``.
        k_descale: As in ``make_attn_fwd``.
        v_descale: As in ``make_attn_fwd``.
        causal: As in ``make_attn_fwd``.
        softmax_scale: As in ``make_attn_fwd``.
        candidates: Config dicts with keys ``pipelined``, ``num_stages``, ``block_n``, ``tma`` and ``warp_spec``.
            Defaults to every config in ``FWD_CANDIDATES`` for the head dim, with ``cp.async``, TMA, and TMA
            with a producer warp group, to ``FWD_CANDIDATES_SM89`` for e4m3 inputs on SM89, or to
            ``FWD_CANDIDATES_BF16`` (``FWD_CANDIDATES_BF16_SM89`` below SM120) for bf16 inputs.
        iters: Timed launches per candidate.
        verbose: Whether to print one line per candidate.

    Returns:
        ``(best_config, results)``, where ``results`` is a list of ``(config, milliseconds or error string)``.

    Raises:
        RuntimeError: If no candidate runs.
    """
    d = q.shape[-1]
    bf16 = q.dtype == torch.bfloat16
    arch = require("attn_fwd" if bf16 else "attn_fwd_fp8", q)
    if candidates is None:
        candidates = (
            (FWD_CANDIDATES_BF16_SM89 if arch != 120 else FWD_CANDIDATES_BF16)[d]
            if bf16
            else FWD_CANDIDATES_SM89[d]
            if arch != 120
            else FWD_CANDIDATES[d]
            + [dict(c, tma=True) for c in FWD_CANDIDATES[d]]
            + [dict(c, tma=True, warp_spec=True) for c in FWD_CANDIDATES[d]]
        )

    def build(cfg):
        return make_attn_fwd(
            q, k, v, q_descale, k_descale, v_descale, causal, softmax_scale, **cfg
        )[0]

    return autotune(
        "fwd_bf16" if bf16 else "fwd",
        d,
        causal,
        q.shape[2],
        k.shape[2],
        q.shape[1] // k.shape[1],
        candidates,
        build,
        iters,
        verbose,
        arch,
    )
