# type: ignore

from ..ptx import mma_16816
from .fwd_base import AttnFwdBase


class AttnFwdSm80(AttnFwdBase):
    """bf16 attention forward for SM80 and newer: ``cp.async`` loads and ``mma.sync`` m16n8k16."""

    def _mma(self, a, b, c):
        return mma_16816(a, b, c, "bf16")
