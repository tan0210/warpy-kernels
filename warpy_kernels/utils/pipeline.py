# type: ignore

import cutlass
from cutlass import Int32, cute

from ..ptx import warp_sync
from .staged import Staged


class CpAsyncPipeline(Staged):
    """A ring of ``stages`` shared-memory buffers filled by ``cp.async``, ``count`` k-tiles in all.

    ``load(i)`` issues the copies of k-tile ``i`` into buffer ``i % stages``. Every thread of the CTA takes part.
    """

    RUNTIME = ("count",)

    def __init__(self, stages, count, load):
        self.stages = stages
        self.count = count
        self.load = load

    @cute.jit
    def prologue(self):
        for s in cutlass.range_constexpr(self.stages - 1):
            if s < self.count:
                self.load(Int32(s))
            cute.arch.cp_async_commit_group()

    @cute.jit
    def wait(self):
        cute.arch.cp_async_wait_group(self.stages - 2)  # the oldest buffer has landed
        cute.arch.sync_threads()  # for every thread

    @cute.jit
    def issue(self, i):
        if i < self.count:
            self.load(i)
        cute.arch.cp_async_commit_group()

    @cute.jit
    def wait_first(self):
        self.wait()

    def issue_kk(self, kk_n):
        return 0  # issue the next load at the first k-step

    @cute.jit
    def step_begin(self, it):
        self.wait()
        self.issue(it + (self.stages - 1))

    def step_end(self, it, lane):
        pass

    @cute.jit
    def advance(self, it, lane):
        self.wait()


class TmaPipeline(Staged):
    """A ring of ``stages`` shared-memory buffers filled by TMA, with full and empty mbarriers.

    ``full`` points at ``2 * stages`` mbarriers: the full ones, then the empty ones. ``arm_and_load(c)`` is called by
    one warp to wait for buffer ``c % stages``, arm its barrier and issue the copies of use ``c``.
    """

    RUNTIME = ("full", "empty", "count", "producer")

    def __init__(self, full, stages, count, arm_and_load, producer):
        self.full = full
        self.empty = full + stages
        self.stages = stages
        self.count = count
        self.arm_and_load = arm_and_load
        self.producer = producer  # whether this thread is in the issuing warp

    @cute.jit
    def init(self, consumer_warps):
        if cute.arch.thread_idx()[0] == 0:
            for s in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(self.full + s, 1)
                cute.arch.mbarrier_init(self.empty + s, consumer_warps)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

    @cute.jit
    def arm(self, c, tx_bytes):
        s = c % self.stages
        if (
            cute.arch.thread_idx()[0] % 32 == 0
        ):  # lane 0 waits for the buffer and arms its barrier
            cute.arch.mbarrier_wait(self.empty + s, ((c // self.stages) & 1) ^ 1)
            cute.arch.mbarrier_arrive_and_expect_tx(self.full + s, tx_bytes)
        warp_sync()  # TMA copies need the full warp

    @cute.jit
    def wait_full(self, c):
        cute.arch.mbarrier_wait(self.full + c % self.stages, (c // self.stages) & 1)

    @cute.jit
    def release(self, c, lane):
        warp_sync()
        if lane == 0:
            cute.arch.mbarrier_arrive(self.empty + c % self.stages)  # buffer is free

    @cute.jit
    def prologue(self):
        if self.producer:
            for s in cutlass.range_constexpr(self.stages - 1):
                if s < self.count:
                    self.arm_and_load(Int32(s))

    @cute.jit
    def issue(self, i):
        if self.producer and i < self.count:
            self.arm_and_load(i)

    def issue_kk(self, kk_n):
        return kk_n // 2  # late, so every warp has released the buffer

    @cute.jit
    def wait_first(self):
        cute.arch.mbarrier_wait(self.full, 0)

    @cute.jit
    def step_begin(self, it):
        self.issue(it + (self.stages - 1))
        self.wait_full(it)

    @cute.jit
    def step_end(self, it, lane):
        self.release(it, lane)

    @cute.jit
    def advance(self, it, lane):
        self.release(it, lane)
        if it + 1 < self.count:
            self.wait_full(it + 1)
