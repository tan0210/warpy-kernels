import torch

ARCHS = {(8, 0): 80, (8, 6): 86, (8, 9): 89, (12, 0): 120}
NAMES = {80: "SM80", 86: "SM86", 89: "SM89", 120: "SM120"}
SUPPORTED = ", ".join(
    f"compute capability {a // 10}.{a % 10} ({NAMES[a]})" for a in ARCHS.values()
)

# usable shared memory bytes per CTA
_SMEM_LIMIT = {80: 166912, 86: 101376, 89: 101376, 120: 101376}

_FEATURES = {
    "gemm": (86, 89, 120),
    "gemm_gelu": (86, 89, 120),
    "gemm_dgelu": (86, 89, 120),
    "gemm_accum": (86, 89, 120),
    "gemm_fp8": (89, 120),
    "gemm_mx": (120,),
    "gemm_mx_out": (89, 120),
    "attn_fwd": (86, 89, 120),
    "attn_bwd": (86, 89, 120),
    "attn_fwd_fp8": (89, 120),
    "attn_bwd_fp8": (89, 120),
}

_device_arch = {}


def device_arch(device=None):
    """Return the architecture of a CUDA device as a number: 80, 86, 89 or 120.

    Args:
        device: CUDA device, index or tensor. Defaults to the current device.

    Returns:
        The architecture number.

    Raises:
        NotImplementedError: If warpy_kernels has no kernels for the device.
    """
    if not torch.cuda.is_available():
        raise NotImplementedError(
            f"warpy_kernels needs a usable CUDA device; supported: {SUPPORTED}"
        )
    if isinstance(device, torch.Tensor):
        device = device.device
    if device is None:
        idx = torch.cuda.current_device()
    elif isinstance(device, int):
        idx = device
    else:
        device = torch.device(device)
        idx = device.index if device.index is not None else torch.cuda.current_device()
    arch = _device_arch.get(idx)
    if arch is None:
        cap = torch.cuda.get_device_capability(idx)
        if cap not in ARCHS:
            raise NotImplementedError(
                f"warpy_kernels has no kernels for compute capability {cap}; supported: {SUPPORTED}"
            )
        arch = _device_arch[idx] = ARCHS[cap]
    return arch


def smem_limit(arch):
    return _SMEM_LIMIT[arch]


def supports(feature, device=None):
    """Return whether a feature runs on a GPU.

    Args:
        feature: One of ``"gemm"``, ``"gemm_gelu"``, ``"gemm_dgelu"``, ``"gemm_accum"``, ``"gemm_fp8"``,
            ``"gemm_mx"`` (MXFP8 inputs), ``"gemm_mx_out"`` (MXFP8 outputs), ``"attn_fwd"``, ``"attn_bwd"``, ``"attn_fwd_fp8"`` and ``"attn_bwd_fp8"``.
        device: CUDA device, index or tensor, or an architecture number such as 86. Defaults to the current
            device.

    Returns:
        ``True`` if the feature runs on that GPU, ``False`` otherwise, including on GPUs warpy_kernels does
        not support at all.

    Raises:
        ValueError: If ``feature`` is not one of the names above.
    """
    if feature not in _FEATURES:
        raise ValueError(f"unknown feature {feature!r}; known: {', '.join(_FEATURES)}")
    if isinstance(device, int) and device in NAMES:
        arch = device
    else:
        try:
            arch = device_arch(device)
        except NotImplementedError:
            return False
    return arch in _FEATURES[feature]


def require(feature, device):
    """Raise unless ``feature`` runs on ``device``.

    Args:
        feature: As in ``supports``.
        device: As in ``supports``.

    Returns:
        The architecture number of ``device``.

    Raises:
        NotImplementedError: If the GPU or the feature on it is unsupported.
    """
    arch = device_arch(device)
    if arch not in _FEATURES[feature]:
        archs = ", ".join(NAMES[a] for a in _FEATURES[feature])
        raise NotImplementedError(
            f"warpy_kernels has no {feature} for {NAMES[arch]}; supported on: {archs}"
        )
    return arch
