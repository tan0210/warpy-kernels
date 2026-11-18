import json
import os

import cuda.bindings.driver as cuda
import torch
import torch.nn.functional as F
from cutlass import Numeric, cute
from cutlass.cute.runtime import from_dlpack, make_fake_stream

_compile_cache = {}


def current_stream():
    """Wrap PyTorch's current CUDA stream for a kernel launch.

    Returns:
        The stream.
    """
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def compile_cached(cls, ctor_args, shape_key, *args):
    """Compile ``cls(*ctor_args)`` for ``args``, reusing an earlier compile of the same configuration.

    Args:
        cls: Kernel class.
        ctor_args: Constructor arguments, positional as a sequence or by keyword as a dict. All of them are part
            of the cache key.
        shape_key: Extra values the compiled kernel depends on, such as problem sizes.
        *args: Arguments the kernel is compiled for.

    Returns:
        The compiled kernel. Each call launches on PyTorch's current CUDA stream at that moment, so launches
        are ordered with PyTorch's own work and are recorded by CUDA graph capture.
    """
    kw = isinstance(ctor_args, dict)
    key = (
        cls.__name__,
        tuple(sorted(ctor_args.items())) if kw else tuple(ctor_args),
        tuple(shape_key),
    )
    if key not in _compile_cache:
        obj = cls(**ctor_args) if kw else cls(*ctor_args)
        _compile_cache[key] = cute.compile(
            obj,
            *[a.cute() if isinstance(a, (Arg, Scalar)) else a for a in args],
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    kernel = _compile_cache[key]
    # TVM-FFI takes torch tensors and launches on PyTorch's current stream
    return lambda *call_args: kernel(*[_raw(a) for a in call_args])


class Arg:
    """A CUDA tensor argument of a compiled kernel, described for the compiler only when it compiles.

    Args:
        t: CUDA tensor.
        align: Assumed byte alignment of the data pointer, or ``None``.
        dynamic: Whether shapes and strides are runtime values.
        leading: Index of the stride-1 dimension when ``dynamic``, or ``None`` to detect it.
    """

    __slots__ = ("align", "dynamic", "leading", "t")

    def __init__(self, t, align=None, dynamic=False, leading=None):
        self.t = t
        self.align = align
        self.dynamic = dynamic
        self.leading = leading

    def cute(self):
        """Return the CuTe tensor the kernel is compiled for.

        Returns:
            The CuTe tensor.
        """
        x = from_dlpack(self.t, assumed_align=self.align)
        if self.dynamic:
            x = x.mark_layout_dynamic(
                **({} if self.leading is None else {"leading_dim": self.leading})
            )
        return x


class Scalar:
    """A scalar kernel argument, typed for the compiler only when it compiles.

    Args:
        ty: CuTe numeric type, such as ``Int32``.
        v: Python value.
    """

    __slots__ = ("ty", "v")

    def __init__(self, ty, v):
        self.ty = ty
        self.v = v

    def cute(self):
        """Return the typed value the kernel is compiled for.

        Returns:
            The CuTe scalar.
        """
        return self.ty(self.v)


_cached = {}


def cached_tensor(key, make):
    """Return a constant tensor made once per ``key``.

    Args:
        key: Hashable key, which should include the device and shape.
        make: Function that creates the tensor.

    Returns:
        The tensor. Callers must not write to it.
    """
    t = _cached.get(key)
    if t is None:
        t = _cached[key] = make()
    return t


def _raw(a):
    if isinstance(a, Arg):
        return a.t
    if isinstance(a, Scalar):
        return a.v
    if isinstance(a, Numeric):
        return a.value
    return a


def flat(x):
    """Describe ``x`` as a flat 16-byte aligned kernel argument.

    Args:
        x: CUDA tensor.

    Returns:
        The argument.
    """
    return Arg(x.reshape(-1), 16, True, 0)


def check_descale(x, batch, h_kv, name):
    """Check that a dequantization scale is ``None`` or a ``[B, Hkv]`` tensor on a CUDA device.

    Args:
        x: The scale, or ``None``.
        batch: Batch size ``B``.
        h_kv: Number of KV heads ``Hkv``.
        name: Argument name for the error message.

    Raises:
        ValueError: If ``x`` is not ``None`` and is not shaped ``[B, Hkv]`` on a CUDA device.
    """
    if x is not None and (tuple(x.shape) != (batch, h_kv) or not x.is_cuda):
        raise ValueError(
            f"{name} must be a CUDA tensor of shape [B, Hkv] = [{batch}, {h_kv}], "
            f"got {tuple(x.shape)} on {x.device}"
        )


def round_up(x, m):
    return (x + m - 1) // m * m


def pad_rows(x, n_pad, value=0):
    n = x.shape[-2] if x.dim() == 4 else x.shape[-1]  # row count
    if n == n_pad:
        return x.contiguous()
    pad = (0, 0, 0, n_pad - n) if x.dim() == 4 else (0, n_pad - n)
    return F.pad(x, pad, value=value).contiguous()


def time_launch_ms(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def time_candidates(candidates, build, iters, verbose, what):
    """Time every candidate config and pick the fastest.

    Args:
        candidates: Config dicts.
        build: Maps a config to a launch function. May raise for configs that do not fit or compile.
        iters: Timed launches per candidate.
        verbose: Whether to print one line per candidate.
        what: Name of the kernel, used in the printed lines and the error.

    Returns:
        ``(best_config, results)``, where ``results`` is a list of ``(config, milliseconds or error string)``.

    Raises:
        RuntimeError: If no candidate runs.
    """
    results = []
    for cfg in candidates:
        try:
            results.append((cfg, time_launch_ms(build(cfg), iters)))
        except Exception as e:  # noqa: BLE001  # skip configs that do not fit or fail to compile
            results.append((cfg, f"{type(e).__name__}: {str(e)[:120]}"))
        if verbose:
            t = results[-1][1]
            print(
                f"  {what} tune {cfg}: {t if isinstance(t, str) else f'{t:.3f} ms'}",
                flush=True,
            )
    timed = [(c, t) for c, t in results if not isinstance(t, str)]
    if not timed:
        raise RuntimeError(f"no {what} config ran: {results}")
    return min(timed, key=lambda ct: ct[1])[0], results


# best attention config per `(kernel, head dim, causal, length bucket, GQA ratio)`
TUNE_CACHE_PATH = os.environ.get(
    "WARPY_KERNELS_TUNE_CACHE", os.path.expanduser("~/.cache/warpy_kernels_tune.json")
)
_tune_cache = None


def _bucket(n):
    return 1 << max(10, (int(n) - 1).bit_length())


def tune_key(kind, d, causal, nq, nk, rep, arch=120):
    # SM120 keys carry no arch suffix, so caches tuned before other archs existed stay valid
    key = (
        f"{kind}|d={d}|causal={int(causal)}|nq={_bucket(nq)}|nk={_bucket(nk)}|rep={rep}"
    )
    return key if arch == 120 else f"{key}|sm{arch}"


def tune_lookup(key):
    global _tune_cache
    if _tune_cache is None:
        try:
            with open(TUNE_CACHE_PATH) as f:
                _tune_cache = json.load(f)
        except (OSError, ValueError):
            _tune_cache = {}
    return _tune_cache.get(key)


def tune_store(key, cfg):
    tune_lookup(key)
    _tune_cache[key] = cfg  # type: ignore
    try:
        os.makedirs(os.path.dirname(TUNE_CACHE_PATH), exist_ok=True)
        with open(TUNE_CACHE_PATH, "w") as f:
            json.dump(_tune_cache, f, indent=1, sort_keys=True)
    except OSError:
        pass


def autotune(kind, d, causal, nq, nk, rep, candidates, build, iters, verbose, arch=120):
    best, results = time_candidates(candidates, build, iters, verbose, kind)
    tune_store(tune_key(kind, d, causal, nq, nk, rep, arch), best)
    return best, results
