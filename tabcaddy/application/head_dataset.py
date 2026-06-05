from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tabcaddy.domain.models import DatasetSource, SourceType
from tabcaddy.infrastructure.csv_reader import scan_csv
from tabcaddy.infrastructure.feather_reader import scan_feather
from tabcaddy.infrastructure.parquet_dataset_reader import scan_parquet_dataset
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


def _read_head(path: Path, n: int) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return scan_csv(path).head(n).collect()
    if suffix in {".feather", ".arrow"}:
        return scan_feather(path).head(n).collect()
    raise ValueError(f"Unsupported file type: {suffix}")


@dataclass
class FileHeadResult:
    path: Path
    df: pl.DataFrame


@dataclass
class HeadResult:
    # Single file / compiled dataset: one entry. Folder: one entry per file.
    frames: list[FileHeadResult]
    is_folder: bool


class HeadDataset:
    def run(self, source: DatasetSource, n: int) -> HeadResult:
        if source.source_type == SourceType.FILE:
            df = _read_head(source.path, n)
            return HeadResult(frames=[FileHeadResult(source.path, df)], is_folder=False)

        if source.source_type == SourceType.COMPILED_DATASET:
            df = scan_parquet_dataset(source.path).head(n).collect()
            return HeadResult(frames=[FileHeadResult(source.path, df)], is_folder=False)

        # Folder: first row of each of the first n files
        files = iter_dataset_files(source)[:n]
        frames = [FileHeadResult(path, _read_head(path, 1)) for path in files]
        return HeadResult(frames=frames, is_folder=True)
