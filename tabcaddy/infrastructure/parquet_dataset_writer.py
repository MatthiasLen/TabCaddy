from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
from tqdm.auto import tqdm


def write_parquet_dataset(
    frames: Iterable[pl.DataFrame], output_path: Path, total: int | None = None
) -> list[Path]:
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    
    # Write each DataFrame to a separate Parquet file in the dataset
    for index, frame in enumerate(
        tqdm(
            frames,
            total=total,
            desc="Writing parquet files",
            unit="file",
            disable=None,
        ),
        start=1,
    ):
        target = data_path / f"part-{index:03d}.parquet"
        frame.write_parquet(target)
        written.append(target)
    return written
