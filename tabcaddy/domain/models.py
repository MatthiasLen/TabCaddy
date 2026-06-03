from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class SourceType(str, Enum):
    FILE = "file"
    FOLDER = "folder"
    COMPILED_DATASET = "compiled_dataset"


class ProfileMode(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class DiffLevel(str, Enum):
    METADATA = "metadata"
    STATISTICS = "statistics"
    FULL = "full"


@dataclass(frozen=True)
class DatasetSource:
    path: Path
    source_type: SourceType


@dataclass(frozen=True)
class ColumnDefinition:
    name: str
    dtype: str


@dataclass
class SchemaSignature:
    columns: list[ColumnDefinition]
    hash: str
    occurrence_count: int


@dataclass
class DatasetMetadata:
    version: int
    created_at: datetime
    row_count: int
    column_count: int
    source_file_count: int
    schema_hash: str | None
    column_hashes: dict[str, str] | None


@dataclass
class ColumnStatistics:
    dtype: str
    null_rate: float
    unique_estimate: int | None
    min_value: Any | None
    max_value: Any | None
    mean: float | None
    median: float | None
    stddev: float | None
    histogram: list[tuple[str, int]] | None = None


@dataclass
class DatasetStatistics:
    columns: dict[str, ColumnStatistics]


@dataclass
class DatasetAnalysis:
    metadata: DatasetMetadata
    schemas: list[SchemaSignature]
    statistics: DatasetStatistics | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DiffReport:
    metadata_changes: list[str]
    schema_changes: list[str]
    statistics_changes: list[str]
    warnings: list[str] = field(default_factory=list)
