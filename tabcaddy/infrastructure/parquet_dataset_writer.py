from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
from tqdm.auto import tqdm

# Flush a new Parquet part once the in-memory buffer reaches this size.
# Parquet columnar compression typically achieves 3-5x on tabular data, so
# this targets roughly 64–128 MB on disk per part file.
_TARGET_CHUNK_BYTES = 256 * 1024 * 1024


def write_parquet_dataset(
    frames: Iterable[pl.DataFrame], output_path: Path, total: int | None = None
) -> list[Path]:
    data_path = output_path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    part_index = 1
    buffer: list[pl.DataFrame] = []
    buffer_bytes = 0

    def _flush() -> None:
        nonlocal part_index, buffer, buffer_bytes
        if not buffer:
            return
        chunk = pl.concat(buffer) if len(buffer) > 1 else buffer[0]
        target = data_path / f"part-{part_index:03d}.parquet"
        chunk.write_parquet(target)
        written.append(target)
        part_index += 1
        buffer = []
        buffer_bytes = 0

    for frame in tqdm(
        frames, total=total, desc="Writing parquet files", unit="file", disable=None
    ):
        buffer.append(frame)
        buffer_bytes += frame.estimated_size()
        if buffer_bytes >= _TARGET_CHUNK_BYTES:
            _flush()

    _flush()
    return written
