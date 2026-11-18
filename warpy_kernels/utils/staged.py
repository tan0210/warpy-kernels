# type: ignore

import copy

from cutlass import extract_mlir_values, new_from_mlir_values


class Staged:
    """Base for helper objects that live across dynamic loops and branches in a kernel.

    ``RUNTIME`` names the fields that hold runtime values; the others stay compile-time.
    """

    RUNTIME = ()

    def __extract_mlir_values__(self):
        values = []
        for f in self.RUNTIME:
            values += extract_mlir_values(getattr(self, f))
        return values

    def __new_from_mlir_values__(self, values):
        new = copy.copy(self)
        i = 0
        for f in self.RUNTIME:
            old = getattr(self, f)
            n = len(extract_mlir_values(old))
            setattr(new, f, new_from_mlir_values(old, values[i : i + n]))
            i += n
        return new
