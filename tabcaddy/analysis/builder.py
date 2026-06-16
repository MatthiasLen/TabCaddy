from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from hashlib import sha256
from json import JSONDecodeError
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from tabcaddy.analysis.metadata import MetadataBuilder
from tabcaddy.analysis.schema import FileSchemaRecord, SchemaAnalyzer
from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.domain.models import (
    ColumnStatistics,
    DatasetAnalysis,
    DatasetSource,
    DatasetStatistics,
    ProfileMode,
    SourceType,
)
from tabcaddy.shared.dataset_io import scan_dataframe, scan_parquet_dataset
from tabcaddy.shared.serialization import analysis_from_dict


@dataclass(frozen=True)
class AnalysisBuildResult:
    analysis: DatasetAnalysis
    files: list[FileSchemaRecord]


def _is_numeric_dtype(dtype: Any) -> bool:
    probe = getattr(dtype, "is_numeric", None)
    if callable(probe):
        return bool(probe())
    if probe is not None:
        return bool(probe)
    return any(token in str(dtype) for token in ("Int", "UInt", "Float", "Decimal"))


def _is_temporal_dtype(dtype: Any) -> bool:
    probe = getattr(dtype, "is_temporal", None)
    if callable(probe):
        return bool(probe())
    if probe is not None:
        return bool(probe)
    return any(
        token in str(dtype) for token in ("Date", "Datetime", "Time", "Duration")
    )


def _supports_min_max(dtype: Any) -> bool:
    probe = getattr(dtype, "is_nested", None)
    if callable(probe):
        return not bool(probe())
    if probe is not None:
        return not bool(probe)
    dtype_str = str(dtype)
    return "List(" not in dtype_str and "Struct(" not in dtype_str


def _supports_approx_n_unique(dtype: Any) -> bool:
    return _supports_min_max(dtype)


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


def _get_temporal_format(dtype: Any) -> str:
    dtype_str = str(dtype)
    if "Datetime" in dtype_str:
        return (
            "%Y-%m-%dT%H:%M:%S%:z" if "time_zone" in dtype_str else "%Y-%m-%dT%H:%M:%S"
        )
    if "Date" in dtype_str:
        return "%Y-%m-%d"
    if "Time" in dtype_str:
        return "%H:%M:%S"
    return "%Y-%m-%d"


def _format_histogram_bound(value: float) -> str:
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3g}"


class AnalysisBuilder:
    def __init__(
        self,
        schema_analyzer: SchemaAnalyzer | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        self._schema_analyzer = schema_analyzer or SchemaAnalyzer()
        self._metadata_builder = metadata_builder or MetadataBuilder()

    def load_compiled_analysis(self, source: DatasetSource) -> DatasetAnalysis | None:
        metadata_path = source.path / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return analysis_from_dict(payload)
        except (KeyError, TypeError, ValueError):
            return None

    def load_compiled_result(self, source: DatasetSource) -> AnalysisBuildResult | None:
        compiled = self.load_compiled_analysis(source)
        if compiled is None:
            return None
        schema_result = self._schema_analyzer.analyze(source)
        return AnalysisBuildResult(analysis=compiled, files=schema_result.files)

    def build(
        self, source: DatasetSource, profile_mode: ProfileMode
    ) -> AnalysisBuildResult:
        if source.source_type == SourceType.COMPILED_DATASET:
            compiled_result = self.load_compiled_result(source)
            if compiled_result is not None:
                return compiled_result

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
        schema_result = self._schema_analyzer.analyze_files(
            files, base_path=base_path, source_type=source_type
        )
        warnings = list(schema_result.warnings)
        if len(schema_result.schemas) > 1:
            warnings.append(
                f"Schema drift detected across {len(schema_result.schemas)} schema groups."
            )

        row_count = sum(record.row_count for record in schema_result.files)
        column_names = {
            column.name for schema in schema_result.schemas for column in schema.columns
        }
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

    def _build_lazyframe(
        self, files: list[Path], source_type: SourceType
    ) -> pl.LazyFrame:
        if source_type == SourceType.COMPILED_DATASET:
            return scan_parquet_dataset(files[0].parent.parent)
        lazyframes = [scan_dataframe(path) for path in files]
        if len(lazyframes) == 1:
            return lazyframes[0]
        return pl.concat(lazyframes, how="diagonal_relaxed")

    def _build_statistics(
        self, lazyframe: pl.LazyFrame, profile_mode: ProfileMode
    ) -> tuple[DatasetStatistics, dict[str, str] | None]:
        schema = lazyframe.collect_schema()
        expressions: list[pl.Expr] = []
        descriptors: list[tuple[str, str, Any]] = []

        for index, (name, dtype) in enumerate(schema.items()):
            prefix = f"c{index}"
            is_numeric = _is_numeric_dtype(dtype)
            is_temporal = _is_temporal_dtype(dtype)
            supports_min_max = _supports_min_max(dtype)
            supports_approx_n_unique = _supports_approx_n_unique(dtype)
            descriptors.append((prefix, name, dtype))

            expressions.append(
                pl.col(name)
                .is_null()
                .mean()
                .fill_null(0.0)
                .alias(f"{prefix}_null_rate")
            )

            if supports_min_max:
                expressions.extend(
                    [
                        (
                            pl.col(name).min().dt.to_string(_get_temporal_format(dtype))
                            if is_temporal
                            else pl.col(name).min()
                        ).alias(f"{prefix}_min"),
                        (
                            pl.col(name).max().dt.to_string(_get_temporal_format(dtype))
                            if is_temporal
                            else pl.col(name).max()
                        ).alias(f"{prefix}_max"),
                    ]
                )

            if is_numeric:
                expressions.extend(
                    [
                        pl.col(name).count().alias(f"{prefix}_count"),
                        pl.col(name).mean().alias(f"{prefix}_mean"),
                        pl.col(name).median().alias(f"{prefix}_median"),
                        pl.col(name).std().alias(f"{prefix}_stddev"),
                    ]
                )

            if is_temporal:
                expressions.append(
                    pl.col(name)
                    .median()
                    .dt.to_string(_get_temporal_format(dtype))
                    .alias(f"{prefix}_median")
                )

            if profile_mode == ProfileMode.DEEP and supports_approx_n_unique:
                expressions.append(
                    pl.col(name).approx_n_unique().alias(f"{prefix}_unique")
                )

        values = (
            lazyframe.select(expressions).collect().row(0, named=True)
            if expressions
            else {}
        )
        if profile_mode == ProfileMode.DEEP:
            column_hashes, histograms = self._build_deep_profiles(
                lazyframe, descriptors, values
            )
        else:
            column_hashes = None
            histograms = {}

        columns: dict[str, ColumnStatistics] = {}
        for prefix, name, dtype in descriptors:
            is_numeric = _is_numeric_dtype(dtype)
            is_temporal = _is_temporal_dtype(dtype)
            columns[name] = ColumnStatistics(
                dtype=str(dtype),
                null_rate=float(values.get(f"{prefix}_null_rate", 0.0) or 0.0),
                unique_estimate=None
                if profile_mode != ProfileMode.DEEP
                or not _supports_approx_n_unique(dtype)
                else int(values.get(f"{prefix}_unique", 0) or 0),
                min_value=(
                    values.get(f"{prefix}_min")
                    if is_temporal
                    else _normalise_value(values.get(f"{prefix}_min"))
                ),
                max_value=(
                    values.get(f"{prefix}_max")
                    if is_temporal
                    else _normalise_value(values.get(f"{prefix}_max"))
                ),
                mean=None
                if not is_numeric
                else _normalise_value(values.get(f"{prefix}_mean")),
                median=None
                if not (is_numeric or is_temporal)
                else (
                    values.get(f"{prefix}_median")
                    if is_temporal
                    else _normalise_value(values.get(f"{prefix}_median"))
                ),
                stddev=None
                if not is_numeric
                else _normalise_value(values.get(f"{prefix}_stddev")),
                histogram=histograms.get(name),
            )
        return DatasetStatistics(columns=columns), column_hashes

    def _build_deep_profiles(
        self,
        lazyframe: pl.LazyFrame,
        descriptors: list[tuple[str, str, Any]],
        values: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, list[tuple[str, int]]]]:
        hashable_columns = [
            (name, dtype)
            for _, name, dtype in descriptors
            if _supports_approx_n_unique(dtype)
        ]
        hash_digests = {name: sha256() for name, _ in hashable_columns}
        string_expressions = [
            pl.col(name).cast(pl.String).fill_null("<NULL>").alias(name)
            for name, _ in hashable_columns
        ]
        histogram_counts: dict[str, np.ndarray] = {}
        histogram_edges: dict[str, np.ndarray] = {}
        histograms: dict[str, list[tuple[str, int]]] = {}

        for prefix, name, dtype in descriptors:
            if not _is_numeric_dtype(dtype):
                continue
            non_null_count = int(values.get(f"{prefix}_count", 0) or 0)
            if non_null_count == 0:
                continue
            lower = float(values[f"{prefix}_min"])
            upper = float(values[f"{prefix}_max"])
            if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-9):
                histograms[name] = [(_format_histogram_bound(lower), non_null_count)]
                continue
            bin_count = min(8, max(2, math.ceil(math.sqrt(non_null_count))))
            histogram_edges[name] = np.linspace(lower, upper, num=bin_count + 1)
            histogram_counts[name] = np.zeros(bin_count, dtype=np.int64)

        for batch in lazyframe.collect_batches():
            if batch.height == 0:
                continue
            string_batch = batch.select(string_expressions)
            for name, digest in hash_digests.items():
                col_values: list[str] = string_batch.get_column(name).to_list()
                digest.update(b"\0".join(v.encode() for v in col_values))
                digest.update(b"\0")
            for name, counts in histogram_counts.items():
                series = batch.get_column(name).drop_nulls()
                if series.len() == 0:
                    continue
                numeric_values = np.asarray(
                    series.cast(pl.Float64).to_numpy(), dtype=float
                )
                edges = histogram_edges[name]
                bucket_indexes = (
                    np.searchsorted(edges, numeric_values, side="right") - 1
                )
                bucket_indexes = np.clip(bucket_indexes, 0, len(counts) - 1)
                counts += np.bincount(bucket_indexes, minlength=len(counts))

        for name, counts in histogram_counts.items():
            edges = histogram_edges[name]
            histograms[name] = [
                (
                    f"{_format_histogram_bound(float(edges[index]))}..{_format_histogram_bound(float(edges[index + 1]))}",
                    int(count),
                )
                for index, count in enumerate(counts.tolist())
            ]
        return (
            {name: digest.hexdigest() for name, digest in hash_digests.items()},
            histograms,
        )
