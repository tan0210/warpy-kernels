import argparse
import json
import sys

import cutlass
import torch
from cutlass import Float32, Int32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from warpy_kernels.sm120.ptx import asm, mma_fp8, unpack_struct

THREADS = 256
CHAINS = 8

# kind: (label, FLOP per mma.sync per thread)
KINDS = {
    "fp8": ("FP8 MMA peak", 16 * 8 * 32 * 2 // 32),
    "bf16": ("BF16 MMA peak", 16 * 8 * 16 * 2 // 32),
}


@dsl_user_op
def mma_bf16(a, b, c, *, loc=None, ip=None):
    code = (
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};"
    )
    args = [Int32(x) for x in a] + [Int32(x) for x in b] + [Float32(x) for x in c]
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        args,
        code,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return unpack_struct(res, T.f32(), Float32, 4, loc, ip)


class MmaRate:
    def __init__(self, kind):
        self.kind = kind

    @cute.jit
    def _step(self, a0: Int32, b0: Int32, acc: cute.Tensor):
        for c in cutlass.range_constexpr(CHAINS):
            frag = [acc[4 * c], acc[4 * c + 1], acc[4 * c + 2], acc[4 * c + 3]]
            if cutlass.const_expr(self.kind == "fp8"):
                r = mma_fp8([a0, a0, a0, a0], [b0, b0], frag)
            else:
                r = mma_bf16([a0, a0, a0, a0], [b0, b0], frag)
            for x in cutlass.range_constexpr(4):
                acc[4 * c + x] = r[x]

    @cute.kernel
    def kernel(self, out: cute.Tensor, iters: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        acc = cute.make_rmem_tensor(4 * CHAINS, Float32)
        acc.fill(0.0)
        a0 = Int32(0x30303030) + tidx * 0
        b0 = Int32(0x30303030) + tidx * 0
        for i in cutlass.range(iters, unroll=1):
            self._step(a0, b0, acc)
        s = Float32(0.0)
        for x in cutlass.range_constexpr(4 * CHAINS):
            s = s + acc[x]
        out[bidx * THREADS + tidx] = s

    @cute.jit
    def __call__(self, out: cute.Tensor, iters: Int32, blocks: Int32):
        self.kernel(out, iters).launch(grid=[blocks, 1, 1], block=[THREADS, 1, 1])


def measure(kind, iters, reps):
    """Peak dense ``mma.sync`` throughput in TFLOP/s for ``kind``, the best of ``reps`` timed runs."""
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    blocks = sms * 4
    out = torch.zeros(blocks * THREADS, device="cuda")
    tensor = from_dlpack(out)
    fn = cute.compile(MmaRate(kind), tensor, Int32(iters), Int32(blocks))
    fn(tensor, Int32(iters), Int32(blocks))
    torch.cuda.synchronize()
    flop = blocks * THREADS * iters * CHAINS * KINDS[kind][1]
    best = 0.0
    for _ in range(reps):
        t0, t1 = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        t0.record()
        fn(tensor, Int32(iters), Int32(blocks))
        t1.record()
        torch.cuda.synchronize()
        best = max(best, flop / (t0.elapsed_time(t1) * 1e-3) / 1e12)
    return best


def main():
    parser = argparse.ArgumentParser(
        prog="python -m tests.peak",
        description="Measure the tensor-core peak and print it as JSON for tests.plot.",
    )
    parser.add_argument(
        "kinds",
        nargs="*",
        choices=list(KINDS),
        help="MMA types to measure (default: fp8)",
    )
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    peaks = {}
    for kind in args.kinds or ["fp8"]:
        label = KINDS[kind][0]
        peaks[label] = round(measure(kind, args.iters, args.reps), 1)
        print(f"{label}: {peaks[label]} TF", file=sys.stderr)
    json.dump(
        {"device": torch.cuda.get_device_name(), "peaks": peaks}, sys.stdout, indent=1
    )
    print()


if __name__ == "__main__":
    main()
