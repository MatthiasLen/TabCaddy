from __future__ import annotations

from pathlib import Path

import polars as pl


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(str(path), infer_schema_length=1000, try_parse_dates=True)


def read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(str(path), try_parse_dates=True)
