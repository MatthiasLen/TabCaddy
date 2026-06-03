from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tabcaddy.domain.models import ColumnStatistics, DatasetAnalysis, DatasetSource, DatasetStatistics, ProfileMode, SourceType
from tabcaddy.domain.serialization import analysis_from_dict
from tabcaddy.infrastructure.csv_reader import scan_csv
from tabcaddy.infrastructure.feather_reader import scan_feather
from tabcaddy.infrastructure.metadata_builder import MetadataBuilder
from tabcaddy.infrastructure.parquet_dataset_reader import scan_parquet_dataset, scan_parquet_file
from tabcaddy.infrastructure.schema_analyzer import FileSchemaRecord, SchemaAnalyzer
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


@dataclass(frozen=True)
class AnalysisBuildResult:
    analysis: DatasetAnalysis
    files: list[FileSchemaRecord]


def _scan_file(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return scan_csv(path)
    if suffix in {".feather", ".arrow"}:
        return scan_feather(path)
    if suffix == ".parquet":
        return scan_parquet_file(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _is_numeric_dtype(dtype: Any) -> bool:
    probe = getattr(dtype, "is_numeric", None)
    if callable(probe):
        return bool(probe())
    if probe is not None:
        return bool(probe)
    return any(token in str(dtype) for token in ("Int", "UInt", "Float", "Decimal"))


def _is_temporal_dtype(dtype: Any) -> bool:
    return any(token in str(dtype) for token in ("Date", "Datetime", "Time"))


def _normalise_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _format_histogram_bound(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3g}"


class AnalysisBuilder:
    def __init__(self, schema_analyzer: SchemaAnalyzer | None = None, metadata_builder: MetadataBuilder | None = None) -> None:
        self._schema_analyzer = schema_analyzer or SchemaAnalyzer()
        self._metadata_builder = metadata_builder or MetadataBuilder()

    def load_compiled_analysis(self, source: DatasetSource) -> DatasetAnalysis | None:
        metadata_path = source.path / "metadata.json"
        if not metadata_path.exists():
            return None
        return analysis_from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))

    def build(self, source: DatasetSource, profile_mode: ProfileMode) -> AnalysisBuildResult:
        if source.source_type == SourceType.COMPILED_DATASET:
            compiled = self.load_compiled_analysis(source)
            if compiled is not None:
                files = self._schema_analyzer.analyze(source).files
                return AnalysisBuildResult(analysis=compiled, files=files)
        return self.build_file_set(
            files=iter_dataset_files(source),
            base_path=source.path,
            source_type=source.source_type,
            profile_mode=profile_mode,
        )

    def build_file_set(
        self,
        files: list[Path],
        base_path: Path,
        source_type: SourceType,
        profile_mode: ProfileMode,
    ) -> AnalysisBuildResult:
        schema_result = self._schema_analyzer.analyze_files(files, base_path=base_path, source_type=source_type)
        warnings = list(schema_result.warnings)
        if len(schema_result.schemas) > 1:
            warnings.append(f"Schema drift detected across {len(schema_result.schemas)} schema groups.")

        row_count = sum(record.row_count for record in schema_result.files)
        column_names = {column.name for schema in schema_result.schemas for column in schema.columns}
        statistics: DatasetStatistics | None = None
        column_hashes: dict[str, str] | None = None

        if schema_result.files and profile_mode != ProfileMode.QUICK:
            lazyframe = self._build_lazyframe(files, source_type)
            statistics, column_hashes = self._build_statistics(lazyframe, profile_mode)

        metadata = self._metadata_builder.build(
            schema_result=schema_result,
            row_count=row_count,
            column_count=len(column_names),
            column_hashes=column_hashes if profile_mode == ProfileMode.DEEP else None,
        )
        return AnalysisBuildResult(
            analysis=DatasetAnalysis(
                metadata=metadata,
                schemas=schema_result.schemas,
                statistics=statistics,
                warnings=warnings,
            ),
            files=schema_result.files,
        )

    def _build_lazyframe(self, files: list[Path], source_type: SourceType) -> pl.LazyFrame:
        if source_type == SourceType.COMPILED_DATASET:
            return scan_parquet_dataset(files[0].parent.parent)
        lazyframes = [_scan_file(path) for path in files]
        if len(lazyframes) == 1:
            return lazyframes[0]
        return pl.concat(lazyframes, how="diagonal_relaxed")

    def _build_statistics(self, lazyframe: pl.LazyFrame, profile_mode: ProfileMode) -> tuple[DatasetStatistics, dict[str, str] | None]:
        schema = lazyframe.collect_schema()
        expressions: list[pl.Expr] = []
        descriptors: list[tuple[str, str, Any]] = []
        for index, (name, dtype) in enumerate(schema.items()):
            prefix = f"c{index}"
            descriptors.append((prefix, name, dtype))
            expressions.extend(
                [
                    pl.col(name).is_null().mean().fill_null(0.0).alias(f"{prefix}_null_rate"),
                    pl.col(name).min().alias(f"{prefix}_min"),
                    pl.col(name).max().alias(f"{prefix}_max"),
                ]
            )
            if _is_numeric_dtype(dtype):
                expressions.extend(
                    [
                        pl.col(name).mean().alias(f"{prefix}_mean"),
                        pl.col(name).median().alias(f"{prefix}_median"),
                        pl.col(name).std().alias(f"{prefix}_stddev"),
                    ]
                )
            if profile_mode == ProfileMode.DEEP:
                expressions.append(pl.col(name).approx_n_unique().alias(f"{prefix}_unique"))

        values = lazyframe.select(expressions).collect().row(0, named=True) if expressions else {}
        column_hashes = self._build_column_hashes(lazyframe, descriptors) if profile_mode == ProfileMode.DEEP else None
        histograms = self._build_histograms(lazyframe, descriptors) if profile_mode == ProfileMode.DEEP else {}

        columns: dict[str, ColumnStatistics] = {}
        for prefix, name, dtype in descriptors:
            columns[name] = ColumnStatistics(
                dtype=str(dtype),
                null_rate=float(values.get(f"{prefix}_null_rate", 0.0) or 0.0),
                unique_estimate=None if profile_mode != ProfileMode.DEEP else int(values.get(f"{prefix}_unique", 0) or 0),
                min_value=_normalise_value(values.get(f"{prefix}_min")),
                max_value=_normalise_value(values.get(f"{prefix}_max")),
                mean=None if not _is_numeric_dtype(dtype) else _normalise_value(values.get(f"{prefix}_mean")),
                median=None if not (_is_numeric_dtype(dtype) or _is_temporal_dtype(dtype)) else _normalise_value(values.get(f"{prefix}_median")),
                stddev=None if not _is_numeric_dtype(dtype) else _normalise_value(values.get(f"{prefix}_stddev")),
                histogram=histograms.get(name),
            )
        return DatasetStatistics(columns=columns), column_hashes

    def _build_column_hashes(self, lazyframe: pl.LazyFrame, descriptors: list[tuple[str, str, Any]]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for _, name, _ in descriptors:
            digest = sha256()
            series = lazyframe.select(pl.col(name).cast(pl.String).fill_null("<NULL>")).collect().get_column(name)
            for value in series.to_list():
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\0")
            hashes[name] = digest.hexdigest()
        return hashes

    def _build_histograms(self, lazyframe: pl.LazyFrame, descriptors: list[tuple[str, str, Any]]) -> dict[str, list[tuple[str, int]]]:
        histograms: dict[str, list[tuple[str, int]]] = {}
        for _, name, dtype in descriptors:
            if not _is_numeric_dtype(dtype):
                continue
            series = lazyframe.select(pl.col(name).drop_nulls()).collect().get_column(name)
            values = [float(value) for value in series.to_list()]
            if not values:
                continue
            lower = min(values)
            upper = max(values)
            if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-9):
                histograms[name] = [(_format_histogram_bound(lower), len(values))]
                continue
            bin_count = min(8, max(2, math.ceil(math.sqrt(len(values)))))
            counts, edges = np.histogram(values, bins=bin_count)
            histograms[name] = [
                (f"{_format_histogram_bound(float(edges[index]))}..{_format_histogram_bound(float(edges[index + 1]))}", int(count))
                for index, count in enumerate(counts.tolist())
            ]
        return histograms
