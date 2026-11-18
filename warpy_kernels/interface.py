import torch

from .arch import require
from .attn import attn_bwd as _attn_bwd
from .attn import attn_fwd as _attn_fwd
from .gemm.api import autotune_gemm, gemm, make_gemm

__all__ = ["attn_bwd", "attn_fwd", "autotune_gemm", "gemm", "make_gemm"]


def _check(q, what, feature, dtypes):
    if q.dtype not in dtypes:
        names = " and ".join(str(t).removeprefix("torch.") for t in dtypes)
        raise NotImplementedError(
            f"warpy_kernels has no {what} for {q.dtype} yet; supported: {names}"
        )
    require(feature if q.dtype == torch.bfloat16 else f"{feature}_fp8", q)


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
    _check(q, "forward kernel", "attn_fwd", (torch.float8_e4m3fn, torch.bfloat16))
    return _attn_fwd(
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
    _check(q, "backward kernel", "attn_bwd", (torch.float8_e4m3fn, torch.bfloat16))
    return _attn_bwd(
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
