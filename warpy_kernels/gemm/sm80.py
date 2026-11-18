# type: ignore

from ..ptx import mma_16816
from .base import GemmBase


class GemmSm80(GemmBase):
    """bf16 and fp16 GEMM for SM80 and newer: ``cp.async`` loads and ``mma.sync`` m16n8k16."""

    def _mma(self, a, b, c, sfa):
        return mma_16816(a, b, c, self.ty)
