from cutlass import Int64

from .staged import Staged


def swizzle_offset(row_bytes, row, chunk):
    if row_bytes == 256:  # two 128-byte halves, each with the 128B swizzle
        return row * 256 + ((((chunk & 7) ^ (row & 7)) | (chunk & 8)) << 4)
    if row_bytes == 128:  # 128B swizzle
        return row * 128 + (((row & 7) ^ chunk) << 4)
    return row * 64 + ((((row >> 1) & 3) ^ chunk) << 4)  # 64B swizzle


class SmemTile(Staged):
    """Rows of ``row_bytes`` in shared memory, 16-byte chunks XOR-swizzled per row."""

    RUNTIME = ("base",)

    def __init__(self, base, row_bytes, rows):
        self.base = base
        self.row_bytes = row_bytes
        self.rows = rows
        self.chunks = row_bytes // 16
        self.bytes = rows * row_bytes

    def addr(self, row, chunk):
        return self.base + swizzle_offset(self.row_bytes, row, chunk)

    def at(self, offset):
        return SmemTile(self.base + offset, self.row_bytes, self.rows)


class StagedTile(Staged):
    """``stages`` copies of one ``SmemTile`` shape, back to back."""

    RUNTIME = ("base",)

    def __init__(self, base, row_bytes, rows, stages):
        self.base = base
        self.row_bytes = row_bytes
        self.rows = rows
        self.stages = stages
        self.stage_bytes = rows * row_bytes
        self.bytes = stages * self.stage_bytes

    def stage(self, s):
        return SmemTile(self.base + s * self.stage_bytes, self.row_bytes, self.rows)


class PanelTile(Staged):
    """A ``[rows, width]`` tile stored as 64-column panels of 128B-swizzled rows, for transposed operands."""

    RUNTIME = ("base",)

    def __init__(self, base, rows, width):
        self.base = base
        self.rows = rows
        self.width = width
        self.panel_bytes = rows * 128

    def addr(self, row, chunk):
        return (
            self.base
            + (chunk // 8) * self.panel_bytes
            + swizzle_offset(128, row, chunk % 8)
        )


class GmemRows(Staged):
    """A row-major global matrix by byte address, with ``ld`` elements between rows."""

    RUNTIME = ("base", "ld")

    def __init__(self, base, ld, elem_bytes):
        self.base = Int64(base)
        self.ld = ld
        self.es = elem_bytes

    def addr(self, row, col):
        return self.base + (Int64(row) * Int64(self.ld) + Int64(col)) * self.es

    def offset_rows(self, rows):
        return GmemRows(
            self.base + Int64(rows) * Int64(self.ld) * self.es, self.ld, self.es
        )


class PaddedTile(Staged):
    """Rows of ``row_bytes`` in shared memory, each padded by 16 bytes so 8-row ``ldmatrix`` reads are conflict-free."""

    RUNTIME = ("base",)

    def __init__(self, base, row_bytes, rows):
        self.base = base
        self.stride = row_bytes + 16
        self.rows = rows
        self.bytes = rows * self.stride

    def addr(self, row, byte):
        return self.base + row * self.stride + byte
