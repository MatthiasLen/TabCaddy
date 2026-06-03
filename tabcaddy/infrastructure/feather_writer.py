from __future__ import annotations

from pathlib import Path

import polars as pl


def write_feather(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(path)
