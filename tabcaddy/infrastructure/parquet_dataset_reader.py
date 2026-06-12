from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.shared.serialization import analysis_from_dict


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
