from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl
from tqdm.auto import tqdm

from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.shared.serialization import analysis_from_dict


SUPPORTED_FILE_SUFFIXES = {".csv", ".feather", ".arrow", ".parquet"}

# Flush a new Parquet part once the in-memory buffer reaches this size.
# Parquet columnar compression typically achieves 3-5x on tabular data, so
# this targets roughly 64-128 MB on disk per part file.
_TARGET_CHUNK_BYTES = 256 * 1024 * 1024


def scan_csv(path: Path) -> pl.LazyFrame:
    return pl.scan_csv(str(path), infer_schema_length=1000, try_parse_dates=True)


def read_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(str(path), try_parse_dates=True)


def write_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def scan_feather(path: Path) -> pl.LazyFrame:
    # Compressed IPC files cannot be memory-mapped; disable mmap to avoid noisy fallback warnings.
    return pl.scan_ipc(str(path), memory_map=False)


def read_feather(path: Path) -> pl.DataFrame:
    # Keep read behavior aligned with scan_feather for consistent warning-free IO.
    return pl.read_ipc(str(path), memory_map=False)


def write_feather(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_ipc(path)


def scan_parquet_dataset(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(path / "data" / "*.parquet"))


def scan_parquet_file(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(path))


def read_parquet_file(path: Path) -> pl.DataFrame:
    return pl.read_parquet(str(path))


def read_compiled_analysis(path: Path) -> DatasetAnalysis | None:
    metadata_file = path / "metadata.json"
    if not metadata_file.exists():
        return None
    return analysis_from_dict(json.loads(metadata_file.read_text(encoding="utf-8")))


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


def scan_dataframe(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return scan_csv(path)
    if suffix in {".feather", ".arrow"}:
        return scan_feather(path)
    if suffix == ".parquet":
        return scan_parquet_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def read_dataframe(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".parquet":
        return read_parquet_file(path)
    if suffix in {".feather", ".arrow"}:
        return read_feather(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def write_dataframe(df: pl.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        write_csv(df, path)
        return
    if suffix == ".parquet":
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return
    if suffix in {".feather", ".arrow"}:
        write_feather(df, path)
        return
    raise ValueError(f"Unsupported file type: {path.suffix}")
