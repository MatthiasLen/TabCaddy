from __future__ import annotations

from pathlib import Path

import polars as pl


def scan_feather(path: Path) -> pl.LazyFrame:
    # Compressed IPC files cannot be memory-mapped; disable mmap to avoid noisy fallback warnings.
    return pl.scan_ipc(str(path), memory_map=False)


def read_feather(path: Path) -> pl.DataFrame:
    # Keep read behavior aligned with scan_feather for consistent warning-free IO.
    return pl.read_ipc(str(path), memory_map=False)
