import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack

from .attn import attn_bwd as sm120_attn_bwd
from .attn import attn_fwd as sm120_attn_fwd
from .gemm import CANDIDATES, FP8_CANDIDATES, GemmSm120
from .utils import compile_cached, flat, time_candidates

_TYPES = {torch.bfloat16: "bf16", torch.float16: "f16"}
SUPPORTED = "compute capability 12.0 (SM120)"


def _bytes_arg(x):
    # read by address, so one compile serves every shape
    return from_dlpack(x.reshape(-1).view(torch.uint8)).mark_layout_dynamic()


def _scale_arg(s):
    s = from_dlpack(s.reshape(-1))
    # vector scales are read by address, so one compile serves every length
    return s.mark_layout_dynamic() if s.shape[0] > 1 else s


def make_gemm(
    a,
    b,
    out=None,
    bk=None,
    stages=3,
    group_m=8,
    min_blocks=1,
    bn=128,
    frag_pipe=False,
    tma=False,
    warp_spec=False,
    persistent=False,
    setmaxnreg=False,
    scale_a=None,
    scale_b=None,
    out_dtype=torch.bfloat16,
    act="",
    store_pre=False,
    pre=None,
    a_trans=False,
    b_trans=False,
    accum=False,
    a_scale_mx=None,
    out_mx=False,
):
    """Prepare ``out = op(a) @ op(b).T * scale_a * scale_b`` on SM120, with vector scales broadcast as ``[M, 1]`` and ``[N]``.

    Args:
        a: ``[M, K]`` bf16, fp16 or ``torch.float8_e4m3fn``, contiguous. ``[K, M]`` if ``a_trans``.
        b: ``[N, K]`` with the dtype of ``a``, contiguous. ``[K, N]`` if ``b_trans``, so the product is
            ``a @ b``.
        out: Optional contiguous ``[M, N]`` output. Allocated if ``None``. Its dtype is the dtype of ``a`` for bf16
            and fp16 inputs, and bf16 or fp16 for fp8 inputs.
        bk: K elements per pipeline stage. 32 or 64 for bf16 and fp16, 64 or 128 for fp8. Defaults to 64 and 128.
        stages: Pipeline depth.
        group_m: Row blocks that sweep all column blocks before the next group, for L2 reuse.
        min_blocks: CTAs per SM to request from the compiler. Must be 1 when ``bn`` is 256.
        bn: Output columns per CTA, 128 or 256.
        frag_pipe: Whether to prefetch the next MMA step's operands into registers while the current step's
            MMAs run. Needs more registers.
        tma: Whether to load A and B with TMA instead of ``cp.async``.
        warp_spec: Whether a dedicated ninth warp issues the TMA loads while the other eight only compute. Needs
            ``tma``.
        persistent: Whether each CTA loops over tiles, so the loads of the next tile overlap the epilogue of the
            current one. Needs ``warp_spec``.
        setmaxnreg: Whether to use a 128-thread producer group that gives up registers to the eight MMA warps,
            which lets them use up to 240 instead of 168. Needs ``warp_spec``.
        scale_a: Float32 CUDA tensor that scales ``a``: one element, or ``M`` elements to scale each output row.
            A one-element scale is only used for fp8 inputs, where it defaults to 1. Row scales apply to every
            input dtype, before ``accum`` and ``act``.
        scale_b: As ``scale_a``, for ``b``, with ``N`` elements to scale each output column.
        out_dtype: Output dtype when ``out`` is ``None`` and the inputs are fp8.
        act: ``"gelu"`` to apply the exact erf GELU to the scaled product before storing it in ``out``, or
            ``"dgelu"`` to multiply the scaled product by the GELU derivative at ``pre`` (the backward of ``"gelu"``).
        store_pre: Whether to also store the product before the activation. Needs ``act="gelu"``.
        pre: The pre-activation values the forward stored, ``[M, N]`` with the dtype of ``out``. Needed for
            ``act="dgelu"``, ignored otherwise.
        a_trans: Whether ``a`` is stored ``[K, M]``. Needs bf16 or fp16 inputs.
        b_trans: Whether ``b`` is stored ``[K, N]``. Same limits as ``a_trans``.
        accum: Whether to add the scaled product to the values ``out`` already holds. The sum is formed before
            the activation, so ``act="gelu"`` gives ``gelu(product + out)`` and ``act="dgelu"`` gives
            ``(product + out) * gelu'(pre)``. Needs ``out``.
        a_scale_mx: ``[M, K / 32]`` uint8 ue8m0 scales of ``a``, one per row and block of 32 along K, as
            ``out_mx`` writes them. Needs fp8 inputs. Combines with ``scale_a`` and ``scale_b``.
        out_mx: Whether to also write the result as MXFP8: e4m3 values ``[M, N]`` and ``[M, N / 32]`` uint8
            ue8m0 scales, one per row and block of 32 columns. The values are the ones ``out`` receives, after
            ``accum`` and ``act``, each block scaled by the smallest power of two that keeps it within 448.

    Returns:
        ``(launch, out)``, or ``(launch, out, pre)`` if ``act="gelu"``, where ``pre`` is ``None`` unless
        ``store_pre`` is set. With ``out_mx``, the e4m3 values and their scales follow. ``launch()`` runs the
        kernel and fills the outputs.

    Raises:
        ValueError: If the dtype is unsupported, the shapes do not match, an input is not contiguous, or
            ``M`` is not a multiple of 128, ``N`` is not a multiple of ``bn`` or ``K`` is not a multiple of ``bk``.
    """
    fp8 = a.dtype == torch.float8_e4m3fn
    if (a.dtype not in _TYPES and not fp8) or b.dtype != a.dtype:
        raise ValueError("a and b must share a bf16, fp16 or float8_e4m3fn dtype")
    odt = (
        out_dtype if fp8 and out is None else out.dtype if out is not None else a.dtype
    )
    if odt not in _TYPES:
        raise ValueError("the output must be bf16 or fp16")
    # 16-bit kernels take the MMA input type from the output type, so the two must agree
    if not fp8 and odt != a.dtype:
        raise ValueError(
            f"out must have the dtype of a and b for 16-bit inputs, got {odt} and {a.dtype}"
        )
    bk = bk or (128 if fp8 else 64)
    if (a_trans or b_trans) and fp8:
        raise ValueError("transposed operands need bf16 or fp16 inputs")
    (m, k) = a.shape[::-1] if a_trans else a.shape
    (n, kb) = b.shape[::-1] if b_trans else b.shape
    if kb != k:
        raise ValueError(f"inner dimensions differ: {k} vs {kb}")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a and b must be contiguous")
    if m % GemmSm120.BM or n % bn or k % bk:
        raise ValueError(
            f"M must be a multiple of 128, N of {bn} and K of {bk}, got {m}, {n}, {k}"
        )
    if accum and out is None:
        raise ValueError("accum needs an out tensor holding the addend")
    if out is None:
        out = torch.empty(m, n, dtype=odt, device=a.device)
    elif out.shape != (m, n) or not out.is_contiguous() or out.data_ptr() % 16:
        raise ValueError("out must be a contiguous [M, N] tensor aligned to 16 bytes")
    if act == "dgelu" and (
        pre is None
        or pre.shape != out.shape
        or pre.dtype != out.dtype
        or not pre.is_contiguous()
    ):
        raise ValueError("act='dgelu' needs a contiguous pre shaped and typed like out")
    tiles = (m // GemmSm120.BM) * (n // bn)
    ctas = (
        torch.cuda.get_device_properties(a.device).multi_processor_count * min_blocks
        if persistent
        else tiles
    )
    # stream-K workspace and flags, one slot per CTA
    sk_ws = (
        torch.empty(
            ctas * GemmSm120.NUM_THREADS * bn // 2, dtype=torch.float32, device=a.device
        )
        if persistent
        else out
    )
    sk_flags = (
        torch.zeros(ctas, dtype=torch.int32, device=a.device) if persistent else out
    )
    pre_out = torch.empty_like(out) if store_pre else None
    pre_in = pre if act == "dgelu" else pre_out
    ones = torch.ones(1, dtype=torch.float32, device=a.device)
    scale_a = ones if scale_a is None else scale_a
    scale_b = ones if scale_b is None else scale_b
    for name, s, size in (("scale_a", scale_a, m), ("scale_b", scale_b, n)):
        if (
            s.dtype != torch.float32
            or not s.is_cuda
            or s.numel() not in (1, size)
            or (s.numel() > 1 and (not s.is_contiguous() or s.data_ptr() % 8))
        ):
            raise ValueError(
                f"{name} must be a float32 CUDA tensor of 1 or {size} elements, contiguous and 8-byte aligned"
            )
    row_scale = scale_a.numel() > 1
    if a_scale_mx is not None and (
        not fp8
        or a_scale_mx.dtype != torch.uint8
        or a_scale_mx.shape != (m, k // 32)
        or not a_scale_mx.is_contiguous()
    ):
        raise ValueError(f"a_scale_mx must be a contiguous [{m}, {k // 32}] uint8 tensor with fp8 inputs")
    q_mx = torch.empty(m, n, dtype=torch.float8_e4m3fn, device=a.device) if out_mx else None
    s_mx = torch.empty(m, n // 32, dtype=torch.uint8, device=a.device) if out_mx else None
    col_scale = scale_b.numel() > 1
    # fp8 is passed as raw bytes
    raw = (lambda x: x.view(torch.uint8)) if fp8 else (lambda x: x)
    dyn = lambda x: from_dlpack(raw(x), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    ops = [dyn(a), dyn(b)] if tma else [flat(raw(a)), flat(raw(b))]
    args = ops + [
        flat(out),
        flat(pre_in if pre_in is not None else out),
        _scale_arg(scale_a),
        _scale_arg(scale_b),
        flat(sk_ws),
        flat(sk_flags),
        _bytes_arg(q_mx if out_mx else out),
        _bytes_arg(s_mx if out_mx else out),
        _bytes_arg(a_scale_mx if a_scale_mx is not None else out),
        Int32(m),
        Int32(n),
        Int32(k),
        Int32(ctas),
    ]
    ctor = (
        _TYPES[odt],
        bk,
        stages,
        group_m,
        min_blocks,
        bn,
        frag_pipe,
        tma,
        warp_spec,
        persistent,
        setmaxnreg,
        fp8,
        act,
        store_pre,
        a_trans,
        b_trans,
        accum,
        row_scale,
        col_scale,
        a_scale_mx is not None,
        out_mx,
    )
    kernel = compile_cached(GemmSm120, ctor, (), *args)

    def launch():
        kernel(*args)

    mx = (q_mx, s_mx) if out_mx else ()
    return (launch, out, pre_out, *mx) if act == "gelu" else (launch, out, *mx)


def gemm(
    a,
    b,
    out=None,
    bk=None,
    stages=3,
    group_m=8,
    min_blocks=1,
    bn=128,
    frag_pipe=False,
    tma=False,
    warp_spec=False,
    persistent=False,
    setmaxnreg=False,
    scale_a=None,
    scale_b=None,
    out_dtype=torch.bfloat16,
    act="",
    store_pre=False,
    pre=None,
    a_trans=False,
    b_trans=False,
    accum=False,
    a_scale_mx=None,
    out_mx=False,
):
    """Compute ``(a @ b.T) * scale_a * scale_b`` on SM120.

    Args:
        a: As in ``make_gemm``.
        b: As in ``make_gemm``.
        out: As in ``make_gemm``.
        bk: As in ``make_gemm``.
        stages: As in ``make_gemm``.
        group_m: As in ``make_gemm``.
        min_blocks: As in ``make_gemm``.
        bn: As in ``make_gemm``.
        frag_pipe: As in ``make_gemm``.
        tma: As in ``make_gemm``.
        warp_spec: As in ``make_gemm``.
        persistent: As in ``make_gemm``.
        setmaxnreg: As in ``make_gemm``.
        scale_a: As in ``make_gemm``.
        scale_b: As in ``make_gemm``.
        out_dtype: As in ``make_gemm``.
        act: As in ``make_gemm``.
        store_pre: As in ``make_gemm``.
        pre: As in ``make_gemm``.
        a_trans: As in ``make_gemm``.
        b_trans: As in ``make_gemm``.
        accum: As in ``make_gemm``.
        a_scale_mx: As in ``make_gemm``.
        out_mx: As in ``make_gemm``.

    Returns:
        ``[M, N]`` in the output dtype, or ``(out, pre)`` if ``act`` is set. With ``out_mx``, a tuple that
        ends with the e4m3 values and their scales.
    """
    launch, out, *rest = make_gemm(
        a,
        b,
        out,
        bk,
        stages,
        group_m,
        min_blocks,
        bn,
        frag_pipe,
        tma,
        warp_spec,
        persistent,
        setmaxnreg,
        scale_a,
        scale_b,
        out_dtype,
        act,
        store_pre,
        pre,
        a_trans,
        b_trans,
        accum,
        a_scale_mx,
        out_mx,
    )
    launch()
    res = (out, *rest) if act == "gelu" or out_mx else out
    return res


def autotune_gemm(a, b, candidates=None, iters=20, verbose=True, **kwargs):
    """Time candidate configs for these shapes.

    Args:
        a: As in ``make_gemm``.
        b: As in ``make_gemm``.
        candidates: Config dicts with the keyword arguments of ``make_gemm``. Defaults to ``FP8_CANDIDATES`` for fp8
            inputs and ``CANDIDATES`` otherwise.
        iters: Timed launches per candidate.
        verbose: Whether to print one line per candidate.
        **kwargs: Extra keyword arguments for ``make_gemm``, such as ``scale_a``.

    Returns:
        ``(best_config, results)``, where ``results`` is a list of ``(config, milliseconds or error string)``.

    Raises:
        RuntimeError: If no candidate runs.
    """
    candidates = candidates or (
        FP8_CANDIDATES if a.dtype == torch.float8_e4m3fn else CANDIDATES
    )
    return time_candidates(
        candidates,
        lambda cfg: make_gemm(a, b, **cfg, **kwargs)[0],
        iters,
        verbose,
        "gemm",
    )


def _check(q, what, dtypes):
    if q.dtype not in dtypes:
        names = " and ".join(str(t).removeprefix("torch.") for t in dtypes)
        raise NotImplementedError(
            f"warpy_kernels has no {what} for {q.dtype} yet; supported: {names} on {SUPPORTED}"
        )
    capability = torch.cuda.get_device_capability(q.device)
    if capability != (12, 0):
        raise NotImplementedError(
            f"warpy_kernels has no {what} for compute capability {capability}; supported: {SUPPORTED}"
        )


def attn_fwd(
    q,
    k,
    v,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    **config,
):
    """Run the attention forward kernel for the dtype of ``q``.

    Args:
        q: ``[B, Hq, Nq, d]`` tensor.
        k: ``[B, Hkv, Nk, d]`` tensor with the same dtype as ``q``.
        v: ``[B, Hkv, Nk, d]`` tensor with the same dtype as ``q``.
        q_descale: ``[B, Hkv]`` float32 dequantization scale, or ``None`` for 1.
        k_descale: Same as ``q_descale``, for ``k``.
        v_descale: Same as ``q_descale``, for ``v``.
        causal: Whether to apply a bottom-right aligned causal mask.
        softmax_scale: Scale on the logits. Defaults to ``1 / sqrt(d)``.
        **config: Kernel-specific options. See ``warpy_kernels.attn.attn_fwd`` for ``float8_e4m3fn`` and ``bfloat16``.

    Returns:
        ``(out, lse, pmax)``. ``out`` is ``[B, Hq, Nq, d]`` bf16, ``lse`` is the ``[B, Hq, Nq]`` base-2
        log-sum-exp, and ``pmax`` is the ``[B, Hq, Nq]`` largest probability per row. Pass all three to
        ``attn_bwd``.

    Raises:
        NotImplementedError: If there is no kernel for the dtype or the GPU.
    """
    _check(q, "forward kernel", (torch.float8_e4m3fn, torch.bfloat16))
    return sm120_attn_fwd(
        q, k, v, q_descale, k_descale, v_descale, causal, softmax_scale, **config
    )


def attn_bwd(
    q,
    k,
    v,
    out,
    lse,
    pmax,
    dout,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    causal=False,
    softmax_scale=None,
    **config,
):
    """Run the attention backward kernel for the outputs of ``attn_fwd``.

    Args:
        q: The ``q`` given to ``attn_fwd``.
        k: The ``k`` given to ``attn_fwd``.
        v: The ``v`` given to ``attn_fwd``.
        out: The ``out`` returned by ``attn_fwd``.
        lse: The ``lse`` returned by ``attn_fwd``.
        pmax: The ``pmax`` returned by ``attn_fwd``.
        dout: Gradient of ``out``, ``[B, Hq, Nq, d]``.
        q_descale: The ``q_descale`` given to ``attn_fwd``.
        k_descale: The ``k_descale`` given to ``attn_fwd``.
        v_descale: The ``v_descale`` given to ``attn_fwd``.
        causal: The ``causal`` given to ``attn_fwd``.
        softmax_scale: The ``softmax_scale`` given to ``attn_fwd``.
        **config: Kernel-specific options. See ``warpy_kernels.attn.attn_bwd`` for ``float8_e4m3fn`` and ``bfloat16``.

    Returns:
        ``(dq, dk, dv)`` in bf16: gradients with respect to the dequantized ``q``, ``k`` and ``v``.

    Raises:
        NotImplementedError: If there is no kernel for the dtype or the GPU.
    """
    _check(q, "backward kernel", (torch.float8_e4m3fn, torch.bfloat16))
    return sm120_attn_bwd(
        q,
        k,
        v,
        out,
        lse,
        pmax,
        dout,
        q_descale,
        k_descale,
        v_descale,
        causal,
        softmax_scale,
        **config,
    )
