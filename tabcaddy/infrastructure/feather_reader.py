from __future__ import annotations

from pathlib import Path

import polars as pl


def scan_feather(path: Path) -> pl.LazyFrame:
    return pl.scan_ipc(str(path))


def read_feather(path: Path) -> pl.DataFrame:
    return pl.read_ipc(str(path))
