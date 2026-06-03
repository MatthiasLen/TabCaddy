from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl


def write_parquet_dataset(frames: Iterable[pl.DataFrame], output_path: Path) -> list[Path]:
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, frame in enumerate(frames, start=1):
        target = data_path / f"part-{index:03d}.parquet"
        frame.write_parquet(target)
        written.append(target)
    return written
