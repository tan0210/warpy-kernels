# type: ignore

import math

import torch
from cutlass import Float32, Int32

from ..arch import require
from ..utils import (
    Arg,
    autotune,
    check_descale,
    compile_cached,
    flat,
    pad_rows,
    round_up,
    tune_key,
    tune_lookup,
)
from .bwd_common import BwdDqPost, BwdDqPostFixed, BwdPrep
from .bwd_sm80 import AttnBwdSm80
from .bwd_sm89 import AttnBwdSm89
from .bwd_sm120 import AttnBwdDqSm120, AttnBwdSm120


def make_attn_bwd(
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
    num_stages=None,
    dq_bulk=None,
    dq_interleave=None,
    deterministic=False,
    block_m=None,
    kb_major=None,
    seqused_k=None,
    one_barrier=None,
):
    """Prepare the attention backward.

    ``launch()`` may not see later in-place changes to the inputs. Call ``make_attn_bwd`` again for new values.

    Args:
        q: The forward's ``q``, ``[B, Hq, Nq, d]``, ``torch.float8_e4m3fn`` or ``torch.bfloat16``.
        k: The forward's ``k``, ``[B, Hkv, Nk, d]``, same dtype as ``q``.
        v: The forward's ``v``, ``[B, Hkv, Nk, d]``, same dtype as ``q``.
        out: The forward's ``out`` (bf16).
        lse: The forward's base-2 ``lse``.
        pmax: The forward's largest probability per row. Unused for bf16 inputs.
        dout: Gradient of ``out``, ``[B, Hq, Nq, d]``. Read as bf16; other dtypes are converted first.
        q_descale: ``[B, Hkv]`` float32 dequantization scale for e4m3 inputs, or ``None`` for 1. bf16 inputs take
            no descales.
        k_descale: Same as ``q_descale``, for ``k``.
        v_descale: Same as ``q_descale``, for ``v``.
        causal: Whether to apply a bottom-right aligned causal mask.
        softmax_scale: Scale on the logits. Defaults to ``1 / sqrt(d)``.
        num_stages: Depth of the Q/dO shared-memory pipeline, at least 2. ``None`` takes the autotuned or built-in
            choice. See ``autotune_attn_bwd``.
        dq_bulk: Whether to reduce dQ with one bulk TMA reduce-add per tile instead of per-element atomics.
            ``None`` takes the autotuned or built-in choice. Ignored below SM120.
        dq_interleave: Whether to issue the dQ atomics between the dV and dK matrix multiplies instead of after
            them. ``None`` takes the autotuned or built-in choice. Ignored on SM120.
        one_barrier: Whether to synchronize once per query tile instead of twice, at the cost of a second dS
            buffer. Needs 3 or more stages. ``None`` takes the autotuned or built-in choice. Ignored on SM120.
        deterministic: Whether the gradients must be bitwise reproducible. e4m3 inputs then add dQ in fixed
            point, which SM89 always does. bf16 inputs on SM120 compute dQ in a separate kernel that recomputes
            S and dP. Not supported for bf16 below SM120.
        block_m: Queries per inner iteration: 32 or 64 for e4m3 inputs, 16, 32 or 64 for bf16. ``None`` takes the
            autotuned or built-in choice.
        kb_major: Whether to launch key blocks slowest, so the longest causal CTAs go first. Defaults to
            ``True`` for causal attention and ``False`` otherwise.
        seqused_k: Must be ``None``. The backward attends to all ``Nk`` keys, so it cannot follow a forward that
            was given ``seqused_k``; passing one raises instead of returning wrong gradients.

    Returns:
        ``(launch, finish)``. ``launch()`` prepares ``dout`` and runs the backward kernels. ``finish()`` then
        returns ``(dq, dk, dv)`` in bf16: gradients with respect to the dequantized ``q``, ``k`` and ``v``.

    Raises:
        NotImplementedError: If ``seqused_k`` is given, ``deterministic`` is set for bf16 below SM120, or the GPU does not
            support the dtype of ``q``. See
            ``warpy_kernels.supports``.
        ValueError: If a tensor has the wrong dtype or shape, or a descale is not ``[B, Hkv]`` or is given for
            bf16 inputs.
    """
    if seqused_k is not None:
        raise NotImplementedError(
            "the backward does not support seqused_k; it would compute gradients over all Nk keys"
        )
    if not (
        q.dtype == k.dtype == v.dtype
        and q.dtype in (torch.float8_e4m3fn, torch.bfloat16)
    ):
        raise ValueError(
            "q, k and v must all be torch.float8_e4m3fn or all torch.bfloat16"
        )
    bf16 = q.dtype == torch.bfloat16
    arch = require("attn_bwd" if bf16 else "attn_bwd_fp8", q)
    if bf16 and not (q_descale is None and k_descale is None and v_descale is None):
        raise ValueError("descales apply to float8_e4m3fn inputs only")
    B, Hq, Nq, d = q.shape
    Hkv, Nk = k.shape[1], k.shape[2]
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
    if out.shape != q.shape or dout.shape != q.shape:
        raise ValueError(
            f"out and dout must be shaped like q {tuple(q.shape)}, got {tuple(out.shape)} and {tuple(dout.shape)}"
        )
    if lse.shape != (B, Hq, Nq) or pmax.shape != (B, Hq, Nq):
        raise ValueError(
            f"lse and pmax must be [B, Hq, Nq] = [{B}, {Hq}, {Nq}], got {tuple(lse.shape)} and {tuple(pmax.shape)}"
        )
    for x, name in (
        (q_descale, "q_descale"),
        (k_descale, "k_descale"),
        (v_descale, "v_descale"),
    ):
        check_descale(x, B, Hkv, name)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    if kb_major is None:
        kb_major = causal
    dev = q.device
    ones = torch.ones(B, Hkv, dtype=torch.float32, device=dev)
    sq, sk, sv = [
        (x if x is not None else ones).float()
        for x in (q_descale, k_descale, v_descale)
    ]
    desc = torch.stack([sq, sk, sv], dim=-1).contiguous()
    if (
        num_stages is None
        or dq_bulk is None
        or dq_interleave is None
        or block_m is None
        or one_barrier is None
    ):
        tuned = (
            tune_lookup(
                tune_key(
                    "bwd_bf16" if bf16 else "bwd", d, causal, Nq, Nk, Hq // Hkv, arch
                )
            )
            or _defaults(arch, bf16, causal)[d]
        )
        num_stages = tuned["num_stages"] if num_stages is None else num_stages
        dq_bulk = tuned.get("dq_bulk", False) if dq_bulk is None else dq_bulk
        dq_interleave = (
            tuned.get("dq_interleave", False)
            if dq_interleave is None
            else dq_interleave
        )
        block_m = tuned["block_m"] if block_m is None else block_m
        one_barrier = (
            tuned.get("one_barrier", False) if one_barrier is None else one_barrier
        )
    if deterministic and arch != 120 and not (arch == 89 and not bf16):
        raise NotImplementedError(
            f"deterministic attention backward is not supported on SM{arch}"
            + (" with bf16 inputs" if arch == 89 else "")
        )
    # e4m3 dQ added as integers is reproducible without the separate dQ kernel, which only bf16 on SM120 needs
    fixed_dq = not bf16 and (arch == 89 or deterministic)
    deterministic = deterministic and bf16 and arch == 120
    if deterministic or arch != 120:
        dq_bulk = False
    nq_pad = round_up(Nq, 128)
    nk_pad = round_up(Nk, 128)  # the key block of every backward kernel

    qp, kp, vp = [
        pad_rows(x if bf16 else x.view(torch.uint8), n)
        for x, n in ((q, nq_pad), (k, nk_pad), (v, nk_pad))
    ]
    lp = pad_rows(lse.float(), nq_pad, value=math.inf)
    dout_c, out_c, pmax_c = (
        dout.to(torch.bfloat16).contiguous(),
        out.to(torch.bfloat16).contiguous(),
        pmax.float().contiguous(),
    )
    dvec = torch.zeros(B, Hq, nq_pad, dtype=torch.float32, device=dev)
    if bf16:
        # the kernels read dO as is; the e4m3 residual and dQ scale slots are unused
        do8 = pad_rows(dout_c, nq_pad)
        do8lo, gq = do8, dvec
    else:
        do8 = (torch.zeros if nq_pad != Nq else torch.empty)(
            B, Hq, nq_pad, d, dtype=torch.uint8, device=dev
        )
        # dO's residual term and the per-query dQ scales, plus fixed-point scales where dQ adds as integers
        do8lo = torch.zeros_like(do8)
        gq = torch.zeros(
            B * Hq * nq_pad * (2 if fixed_dq else 1), dtype=torch.float32, device=dev
        )
    # [dO amax, unused, max |V|, unused] per (batch, kv head)
    stats_buf = torch.zeros(B * Hkv * 4, dtype=torch.float32, device=dev)
    stats = stats_buf.view(B, Hkv, 4)
    stats_init = torch.zeros_like(stats_buf)
    if not bf16:
        stats_init.view(B, Hkv, 4)[:, :, 2] = (
            (v.float() * sv[:, :, None, None]).norm(dim=-1).amax(dim=2)
        )
    if fixed_dq:
        stats_init.view(B, Hkv, 4)[:, :, 1] = k.float().abs().amax(dim=(2, 3))
    dk_out = torch.empty(B, Hkv, nk_pad, d, dtype=torch.bfloat16, device=dev)
    dv_out = torch.empty(B, Hkv, nk_pad, d, dtype=torch.bfloat16, device=dev)

    run_main_dq = not deterministic
    dq_acc = (
        torch.empty(B, Hq, nq_pad, d, dtype=torch.float32, device=dev)
        if run_main_dq
        else None
    )
    prep_args = [
        flat(dout_c),
        flat(out_c),
        flat(pmax_c),
        flat(dvec if bf16 else do8),
        flat(dvec if bf16 else do8lo),
        flat(dvec),
        flat(gq),
        Arg(stats),
        flat(dq_acc if dq_acc is not None else dvec),
        Int32(B * Hq * Nq),
        Int32(Nq),
        Int32(nq_pad),
        Int32(Hq),
        Int32(Hkv),
    ]
    prep = compile_cached(
        BwdPrep,
        (d, dq_acc is not None, bf16) + ((True,) if fixed_dq else ()),
        (str(dout_c.dtype), B, Hq, Hkv, Nq, nq_pad),
        *prep_args,
    )

    if deterministic:
        dq_out = torch.empty(B, Hq, nq_pad, d, dtype=torch.bfloat16, device=dev)
        dqk_args = [
            flat(qp),
            flat(kp),
            flat(vp),
            flat(do8),
            flat(do8lo),
            flat(lp),
            flat(dvec),
            flat(gq),
            Arg(stats),
            Arg(desc),
            flat(dq_out),
            Float32(softmax_scale),
            Int32(Nq),
            Int32(Nk),
            Int32(nq_pad),
            Int32(nk_pad),
            Int32(Hq),
            Int32(Hkv),
            Int32(B),
        ]
        dqk = compile_cached(
            AttnBwdDqSm120,
            (d, causal, 2, bf16),
            (B, Hq, Hkv, nq_pad, nk_pad),
            *dqk_args,
        )
    else:
        dq_out = torch.empty(B, Hq, Nq, d, dtype=torch.bfloat16, device=dev)
        post_args = [
            flat(dq_acc),
            Arg(stats),
            Arg(desc),
            flat(dq_out),
            Float32(softmax_scale),
            Int32(Nq),
            Int32(nq_pad),
            Int32(Hq),
            Int32(Hkv),
            Int32(B),
        ]
        if fixed_dq:
            post_args.insert(1, flat(gq))
            post = compile_cached(
                BwdDqPostFixed,
                (d, block_m, dq_bulk),
                (B, Hq, Hkv, Nq, nq_pad),
                *post_args,
            )
        else:
            post = compile_cached(
                BwdDqPost,
                (d, block_m, dq_bulk),
                (B, Hq, Hkv, Nq, nq_pad),
                *post_args,
            )

    scalars = [
        Float32(softmax_scale),
        Int32(Nq),
        Int32(Nk),
        Int32(nq_pad),
        Int32(nk_pad),
        Int32(Hq),
        Int32(Hkv),
        Int32(B),
    ]
    if arch == 120:
        main_args = [
            flat(qp),
            flat(kp),
            flat(vp),
            flat(do8),
            flat(do8lo),
            flat(lp),
            flat(dvec),
            flat(gq),
            Arg(stats),
            Arg(desc),
            flat(dk_out),
            flat(dv_out),
            flat(dq_acc if dq_acc is not None else dvec),
            *scalars,
        ]
        main = compile_cached(
            AttnBwdSm120,
            (d, causal, block_m, num_stages, not run_main_dq, dq_bulk, kb_major, bf16)
            + ((True,) if fixed_dq else ()),
            (B, Hq, Hkv, nq_pad, nk_pad),
            *main_args,
        )
    else:
        main_args = [
            flat(qp),
            flat(kp),
            flat(vp),
            flat(do8),
            flat(do8lo),
            flat(lp),
            flat(dvec),
            flat(gq),
            Arg(stats),
            Arg(desc),
            flat(dk_out),
            flat(dv_out),
            flat(dq_acc),
            *scalars,
        ]
        main = compile_cached(
            AttnBwdSm89 if arch == 89 else AttnBwdSm80,
            {
                "head_dim": d,
                "causal": causal,
                "block_m": block_m,
                "num_stages": num_stages,
                "kb_major": kb_major,
                "arch": arch,
                "dq_interleave": dq_interleave,
                "bf16": bf16,
                "one_barrier": one_barrier and num_stages >= 3,
            },
            (B, Hq, Hkv, nq_pad, nk_pad),
            *main_args,
        )

    def launch():
        # prep zeroes every dQaccum row, including padding
        stats_buf.copy_(stats_init)
        prep(*prep_args)
        main(*main_args)
        if deterministic:
            dqk(*dqk_args)

    def launch_main_only():
        main(*main_args)
        if deterministic:
            dqk(*dqk_args)

    def finish():
        if deterministic:
            return dq_out[:, :, :Nq], dk_out[:, :, :Nk], dv_out[:, :, :Nk]
        post(*post_args)
        return dq_out, dk_out[:, :, :Nk], dv_out[:, :, :Nk]

    launch.main_only = launch_main_only
    return launch, finish


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
    num_stages=None,
    dq_bulk=None,
    dq_interleave=None,
    deterministic=False,
    block_m=None,
    kb_major=None,
    seqused_k=None,
    one_barrier=None,
):
    """Run the attention backward once.

    Args:
        q: As in ``make_attn_bwd``.
        k: As in ``make_attn_bwd``.
        v: As in ``make_attn_bwd``.
        out: As in ``make_attn_bwd``.
        lse: As in ``make_attn_bwd``.
        pmax: As in ``make_attn_bwd``.
        dout: As in ``make_attn_bwd``.
        q_descale: As in ``make_attn_bwd``.
        k_descale: As in ``make_attn_bwd``.
        v_descale: As in ``make_attn_bwd``.
        causal: As in ``make_attn_bwd``.
        softmax_scale: As in ``make_attn_bwd``.
        num_stages: As in ``make_attn_bwd``.
        dq_bulk: As in ``make_attn_bwd``.
        dq_interleave: As in ``make_attn_bwd``.
        deterministic: As in ``make_attn_bwd``.
        block_m: As in ``make_attn_bwd``.
        kb_major: As in ``make_attn_bwd``.
        seqused_k: As in ``make_attn_bwd``.
        one_barrier: As in ``make_attn_bwd``.

    Returns:
        ``(dq, dk, dv)`` in bf16.
    """
    launch, finish = make_attn_bwd(
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
        num_stages,
        dq_bulk=dq_bulk,
        dq_interleave=dq_interleave,
        deterministic=deterministic,
        block_m=block_m,
        kb_major=kb_major,
        seqused_k=seqused_k,
        one_barrier=one_barrier,
    )
    launch()
    return finish()


BWD_DEFAULTS = {
    128: {"num_stages": 3, "dq_bulk": True, "block_m": 32},
    64: {"num_stages": 3, "dq_bulk": True, "block_m": 64},
}
# bf16 Q/dO tiles are twice as wide, so d128 fits 16 queries per tile
BWD_DEFAULTS_BF16 = {
    128: {"num_stages": 3, "dq_bulk": False, "block_m": 16},
    64: {"num_stages": 3, "dq_bulk": True, "block_m": 32},
}
# bf16 below SM120 with 99 KB of shared memory, where d128 fits 16 queries per tile
BWD_DEFAULTS_SM80 = {
    128: {"num_stages": 2, "block_m": 16, "dq_interleave": False},
    64: {"num_stages": 3, "block_m": 32, "dq_interleave": True},
}
BWD_CANDIDATES_SM80 = {
    128: [
        {"block_m": 16, "num_stages": 2, "dq_interleave": il} for il in (False, True)
    ],
    64: [
        {"block_m": bm, "num_stages": st, "dq_interleave": il}
        for bm, st in ((32, 2), (32, 3), (64, 2))
        for il in (False, True)
    ],
}
# e4m3 on SM89, fastest on an RTX 4070 Ti SUPER at n = 2048 and 8192 with dQ added as integers
BWD_DEFAULTS_SM89 = {
    128: {"num_stages": 2, "block_m": 32, "dq_interleave": True, "one_barrier": False},
    64: {"num_stages": 2, "block_m": 64, "dq_interleave": True, "one_barrier": False},
}
# causal tiles finish unevenly, which one barrier per tile tolerates better
BWD_DEFAULTS_CAUSAL_SM89 = {
    128: {"num_stages": 3, "block_m": 32, "dq_interleave": False, "one_barrier": True},
    64: {"num_stages": 3, "block_m": 64, "dq_interleave": False, "one_barrier": True},
}
# d128 with 64 queries spills
BWD_CANDIDATES_SM89 = {
    128: [
        {"block_m": 32, "num_stages": st, "dq_interleave": il, "one_barrier": ob}
        for st in (2, 3)
        for il in (False, True)
        for ob in ((False, True) if st >= 3 else (False,))
    ],
    64: [
        {"block_m": bm, "num_stages": st, "dq_interleave": il, "one_barrier": ob}
        for bm in (32, 64)
        for st in (2, 3)
        for il in (False, True)
        for ob in ((False, True) if st >= 3 else (False,))
    ],
}
BWD_CANDIDATES = {
    128: [
        {"block_m": 32, "num_stages": st, "dq_bulk": bulk}
        for st in (2, 3)
        for bulk in (False, True)
    ],
    64: [
        {"block_m": bm, "num_stages": st, "dq_bulk": bulk}
        for bm in (32, 64)
        for st in (2, 3)
        for bulk in (False, True)
    ],
}
BWD_CANDIDATES_BF16 = {
    128: [
        {"block_m": 16, "num_stages": 2, "dq_bulk": False},
        {"block_m": 16, "num_stages": 2, "dq_bulk": True},
        {"block_m": 16, "num_stages": 3, "dq_bulk": False},
    ],
    64: [
        {"block_m": bm, "num_stages": st, "dq_bulk": bulk}
        for bm in (16, 32)
        for st in (2, 3)
        for bulk in (False, True)
    ]
    + [{"block_m": 64, "num_stages": 2, "dq_bulk": False}],
}


def autotune_attn_bwd(
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
    candidates=None,
    iters=10,
    verbose=True,
):
    """Time candidate backward configs and cache the fastest.

    Each candidate is timed as a full step: prep, kernels and the dQ postprocess.

    Args:
        q: As in ``make_attn_bwd``.
        k: As in ``make_attn_bwd``.
        v: As in ``make_attn_bwd``.
        out: As in ``make_attn_bwd``.
        lse: As in ``make_attn_bwd``.
        pmax: As in ``make_attn_bwd``.
        dout: As in ``make_attn_bwd``.
        q_descale: As in ``make_attn_bwd``.
        k_descale: As in ``make_attn_bwd``.
        v_descale: As in ``make_attn_bwd``.
        causal: As in ``make_attn_bwd``.
        softmax_scale: As in ``make_attn_bwd``.
        candidates: Config dicts with keys ``block_m``, ``num_stages``, ``dq_bulk`` and ``dq_interleave``. Defaults to
            ``BWD_CANDIDATES`` for the head dim, ``BWD_CANDIDATES_BF16`` for bf16 inputs, or
            ``BWD_CANDIDATES_SM80`` (bf16) or ``BWD_CANDIDATES_SM89`` (e4m3) below SM120.
        iters: Timed steps per candidate.
        verbose: Whether to print one line per candidate.

    Returns:
        ``(best_config, results)``, where ``results`` is a list of ``(config, milliseconds or error string)``.

    Raises:
        RuntimeError: If no candidate runs.
    """
    d = q.shape[-1]

    def build(cfg):
        launch, finish = make_attn_bwd(
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
            **cfg,
        )
        return lambda: (launch(), finish())

    bf16 = q.dtype == torch.bfloat16
    arch = require("attn_bwd" if bf16 else "attn_bwd_fp8", q)
    return autotune(
        "bwd_bf16" if bf16 else "bwd",
        d,
        causal,
        q.shape[2],
        k.shape[2],
        q.shape[1] // k.shape[1],
        candidates or candidates_for(arch, bf16)[d],
        build,
        iters,
        verbose,
        arch,
    )


def _defaults(arch, bf16, causal=False):
    if arch != 120:
        if bf16:
            return BWD_DEFAULTS_SM80
        return BWD_DEFAULTS_CAUSAL_SM89 if causal else BWD_DEFAULTS_SM89
    return BWD_DEFAULTS_BF16 if bf16 else BWD_DEFAULTS


def candidates_for(arch, bf16):
    """Return the default backward autotuning candidates for an architecture.

    Args:
        arch: Architecture number, such as 86 or 120.
        bf16: Whether the inputs are bf16 rather than ``torch.float8_e4m3fn``.

    Returns:
        A dict from head dim to a list of config dicts with the keyword arguments of ``make_attn_bwd``.
    """
    if arch != 120:
        return BWD_CANDIDATES_SM80 if bf16 else BWD_CANDIDATES_SM89
    return BWD_CANDIDATES_BF16 if bf16 else BWD_CANDIDATES
