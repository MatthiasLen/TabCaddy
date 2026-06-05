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

from tabcaddy.domain.models import (
    ColumnStatistics,
    DatasetAnalysis,
    DatasetSource,
    DatasetStatistics,
    ProfileMode,
    SourceType,
)
from tabcaddy.domain.serialization import analysis_from_dict
from tabcaddy.infrastructure.csv_reader import scan_csv
from tabcaddy.infrastructure.feather_reader import scan_feather
from tabcaddy.infrastructure.metadata_builder import MetadataBuilder
from tabcaddy.infrastructure.parquet_dataset_reader import (
    scan_parquet_dataset,
    scan_parquet_file,
)
from tabcaddy.infrastructure.schema_analyzer import FileSchemaRecord, SchemaAnalyzer
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


@dataclass(frozen=True)
class AnalysisBuildResult:
    """Result of a dataset analysis operation.

    Attributes:
        analysis: The computed DatasetAnalysis containing metadata, schemas, and statistics.
        files: List of file schema records analyzed during the process.
    """

    analysis: DatasetAnalysis
    files: list[FileSchemaRecord]


def _scan_file(path: Path) -> pl.LazyFrame:
    """Scan a file and return a lazy Polars DataFrame.

    Dispatches to the appropriate reader based on file extension.
    Supports CSV, Feather (Arrow IPC), and Parquet formats.

    Args:
        path: Path to the file to scan.

    Returns:
        A lazy Polars DataFrame ready for query optimization and execution.

    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return scan_csv(path)
    if suffix in {".feather", ".arrow"}:
        return scan_feather(path)
    if suffix == ".parquet":
        return scan_parquet_file(path)
    raise ValueError(
        f"Unsupported file type: {path.suffix}. Supported: .csv, .feather, .arrow, .parquet"
    )


def _is_numeric_dtype(dtype: Any) -> bool:
    # Try modern Polars API first
    probe = getattr(dtype, "is_numeric", None)
    if callable(probe):
        return bool(probe())
    # Fall back to attribute access
    if probe is not None:
        return bool(probe)
    # String-based detection as last resort
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


def _normalise_value(value: Any) -> Any:
    """Normalize a Python value for serialization.

    Args:
        value: A value to normalize, typically from a Polars aggregation.

    Returns:
        The normalized value suitable for JSON serialization.
    """
    if value is None:
        return None
    # NaN is not JSON-serializable
    if isinstance(value, float) and math.isnan(value):
        return None
    # Convert Decimal to float for serialization
    if isinstance(value, Decimal):
        return float(value)
    # Use ISO 8601 format for temporal types
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _get_temporal_format(dtype: Any) -> str:
    """Get the appropriate format string for a temporal dtype.

    Date types use YYYY-MM-DD format, Datetime with timezone uses ISO 8601
    with timezone, and Time uses HH:MM:SS format.
    """
    dtype_str = str(dtype)
    if "Datetime" in dtype_str:
        return (
            "%Y-%m-%dT%H:%M:%S%:z" if "time_zone" in dtype_str else "%Y-%m-%dT%H:%M:%S"
        )
    if "Date" in dtype_str:
        return "%Y-%m-%d"
    if "Time" in dtype_str:
        return "%H:%M:%S"
    return "%Y-%m-%d"  # Fallback for unknown temporal types


def _format_histogram_bound(value: float) -> str:
    """Format a histogram bin boundary for readable display."""
    # Display whole numbers as integers
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    # Use 3-significant-figure notation for other numbers
    return f"{value:.3g}"


class AnalysisBuilder:
    """Orchestrates the computation of dataset analysis including metadata and statistics.

    Analyzes one or more files to extract:
    - Schema information (column names, types, structure)
    - Metadata (row/column counts, source file counts, schema hashes)
    - Statistical summaries (null rates, min/max, mean, histograms, etc.)
    - Column content hashes (for deep profiling mode)

    Supports three profiling modes:
    - QUICK: Schema only, no statistics
    - STANDARD: Schema and basic statistics (null rate, min/max)
    - DEEP: All statistics plus cardinality and column hashes

    Can analyze raw data folders or pre-compiled parquet datasets.
    """

    def __init__(
        self,
        schema_analyzer: SchemaAnalyzer | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        """Initialize the AnalysisBuilder with optional custom dependencies.

        Args:
            schema_analyzer: Optional custom SchemaAnalyzer. Defaults to new instance.
            metadata_builder: Optional custom MetadataBuilder. Defaults to new instance.
        """
        self._schema_analyzer = schema_analyzer or SchemaAnalyzer()
        self._metadata_builder = metadata_builder or MetadataBuilder()

    def load_compiled_analysis(self, source: DatasetSource) -> DatasetAnalysis | None:
        """Load a pre-computed analysis from a compiled dataset.

        Compiled datasets store their analysis in metadata.json at the root.
        This allows rapid re-analysis without recomputing statistics.
        """
        metadata_path = source.path / "metadata.json"
        if not metadata_path.exists():
            return None
        return analysis_from_dict(json.loads(metadata_path.read_text(encoding="utf-8")))

    def load_compiled_result(self, source: DatasetSource) -> AnalysisBuildResult | None:
        """Load a pre-computed analysis and pair it with current file records."""
        compiled = self.load_compiled_analysis(source)
        if compiled is None:
            return None
        schema_result = self._schema_analyzer.analyze(source)
        return AnalysisBuildResult(analysis=compiled, files=schema_result.files)

    def build(
        self, source: DatasetSource, profile_mode: ProfileMode
    ) -> AnalysisBuildResult:
        """Analyze a dataset from a DatasetSource.

        Dispatches to either load_compiled_analysis (for pre-compiled datasets)
        or build_file_set (for raw folders and files).

        Args:
            source: The dataset source to analyze (file, folder, or compiled dataset).
            profile_mode: The profiling depth (QUICK, STANDARD, or DEEP).

        Returns:
            An AnalysisBuildResult with computed analysis and file records.
        """
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
        """Analyze a set of files and compute their dataset analysis.

        Core orchestration method that:
        1. Analyzes schema across all files
        2. Detects schema drift (multiple schemas)
        3. Computes statistics if requested (skipped for QUICK mode)
        4. Builds metadata and analysis result

        Args:
            files: List of file paths to analyze.
            base_path: Base directory for relative path computation.
            source_type: Type of source (FILE, FOLDER, COMPILED_DATASET).
            profile_mode: Profiling depth level.

        Returns:
            An AnalysisBuildResult with complete analysis.
        """
        # Analyze schemas across all files
        schema_result = self._schema_analyzer.analyze_files(
            files, base_path=base_path, source_type=source_type
        )
        warnings = list(schema_result.warnings)
        # Alert if multiple distinct schemas are detected
        if len(schema_result.schemas) > 1:
            warnings.append(
                f"Schema drift detected across {len(schema_result.schemas)} schema groups."
            )

        # Aggregate row and column counts
        row_count = sum(record.row_count for record in schema_result.files)
        column_names = {
            column.name for schema in schema_result.schemas for column in schema.columns
        }
        statistics: DatasetStatistics | None = None
        column_hashes: dict[str, str] | None = None

        # Compute statistics unless QUICK mode (schema-only)
        if schema_result.files and profile_mode != ProfileMode.QUICK:
            lazyframe = self._build_lazyframe(files, source_type)
            statistics, column_hashes = self._build_statistics(lazyframe, profile_mode)

        # Assemble the complete metadata
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
        """Build a lazy DataFrame from one or more files.

        For compiled datasets, scans the parquet partitions directly.
        For raw data, scans each file and concatenates using diagonal_relaxed
        to handle schema drift gracefully (adds null columns for missing fields).

        Args:
            files: List of file paths to load.
            source_type: Type of data source.

        Returns:
            A lazy Polars DataFrame ready for query execution.
        """
        # Compiled datasets are pre-partitioned parquet files
        if source_type == SourceType.COMPILED_DATASET:
            return scan_parquet_dataset(files[0].parent.parent)
        # Scan individual files based on their format
        lazyframes = [_scan_file(path) for path in files]
        # Single file: return directly
        if len(lazyframes) == 1:
            return lazyframes[0]
        # Multiple files: concatenate with diagonal_relaxed to handle schema drift
        # (adds null columns for fields not present in all files)
        return pl.concat(lazyframes, how="diagonal_relaxed")

    def _build_statistics(
        self, lazyframe: pl.LazyFrame, profile_mode: ProfileMode
    ) -> tuple[DatasetStatistics, dict[str, str] | None]:
        """Compute column statistics from a lazy DataFrame.

        Builds expressions for all columns to compute in a single Polars query:
        - All columns: null_rate, min, max
        - Numeric columns: mean, median, stddev, (optional) histogram
        - Temporal columns: median (formatted as ISO string)
        - Deep mode: approx_n_unique cardinality, column content hashes

        Uses prefix-based aliasing to avoid name collisions in the aggregated result,
        then unpacks values and applies appropriate normalization per type.

        Args:
            lazyframe: Lazy DataFrame to analyze.
            profile_mode: Profiling depth (STANDARD or DEEP).

        Returns:
            Tuple of (DatasetStatistics with computed column stats, optional column_hashes dict).
        """
        schema = lazyframe.collect_schema()
        expressions: list[pl.Expr] = []
        descriptors: list[tuple[str, str, Any]] = []

        # Process each column in the schema
        for index, (name, dtype) in enumerate(schema.items()):
            # Use numeric prefix to avoid alias collisions in aggregated result
            prefix = f"c{index}"
            is_numeric = _is_numeric_dtype(dtype)
            is_temporal = _is_temporal_dtype(dtype)
            descriptors.append((prefix, name, dtype))

            # Universal statistics: computed for all columns
            expressions.extend(
                [
                    # Null rate: fraction of null values
                    pl.col(name)
                    .is_null()
                    .mean()
                    .fill_null(0.0)
                    .alias(f"{prefix}_null_rate"),
                    # Min: format temporal values as ISO strings to avoid Python zoneinfo panic
                    (
                        pl.col(name).min().dt.to_string(_get_temporal_format(dtype))
                        if is_temporal
                        else pl.col(name).min()
                    ).alias(f"{prefix}_min"),
                    # Max: format temporal values as ISO strings
                    (
                        pl.col(name).max().dt.to_string(_get_temporal_format(dtype))
                        if is_temporal
                        else pl.col(name).max()
                    ).alias(f"{prefix}_max"),
                ]
            )

            # Numeric-specific statistics
            if is_numeric:
                expressions.extend(
                    [
                        pl.col(name).count().alias(f"{prefix}_count"),
                        pl.col(name).mean().alias(f"{prefix}_mean"),
                        pl.col(name).median().alias(f"{prefix}_median"),
                        pl.col(name).std().alias(f"{prefix}_stddev"),
                    ]
                )

            # Temporal-specific statistics
            if is_temporal:
                # Median for temporal: format as ISO string to cross Python boundary safely
                expressions.append(
                    pl.col(name)
                    .median()
                    .dt.to_string(_get_temporal_format(dtype))
                    .alias(f"{prefix}_median")
                )

            # Deep profile: cardinality and hashes (computed later)
            if profile_mode == ProfileMode.DEEP:
                expressions.append(
                    pl.col(name).approx_n_unique().alias(f"{prefix}_unique")
                )

        # Execute all aggregations in a single query for efficiency
        values = (
            lazyframe.select(expressions).collect().row(0, named=True)
            if expressions
            else {}
        )
        # Deep profiles need one additional streamed pass for hashes and histogram bins.
        if profile_mode == ProfileMode.DEEP:
            column_hashes, histograms = self._build_deep_profiles(
                lazyframe, descriptors, values
            )
        else:
            column_hashes = None
            histograms = {}

        # Unpack aggregation results and construct ColumnStatistics for each column
        columns: dict[str, ColumnStatistics] = {}
        for prefix, name, dtype in descriptors:
            is_numeric = _is_numeric_dtype(dtype)
            is_temporal = _is_temporal_dtype(dtype)
            # Temporal min/max/median are already ISO strings from dt.to_string()
            columns[name] = ColumnStatistics(
                dtype=str(dtype),
                null_rate=float(values.get(f"{prefix}_null_rate", 0.0) or 0.0),
                unique_estimate=None
                if profile_mode != ProfileMode.DEEP
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
        """Build deep-profile hashes and histograms with one streamed batch pass.

        The scalar statistics query already computed min/max and non-null counts.
        This method reuses those results to avoid rescanning the dataset once per
        column for hashes and histograms.
        """
        hash_digests = {name: sha256() for _, name, _ in descriptors}
        string_expressions = [
            pl.col(name).cast(pl.String).fill_null("<NULL>").alias(name)
            for _, name, _ in descriptors
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
                # Batch all row updates into a single digest.update() call per column
                # per batch; b"\0".join(...) + b"\0" produces the same byte sequence
                # as updating each value followed by a null separator individually.
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
