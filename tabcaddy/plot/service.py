from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
import math
from pathlib import Path
from typing import Any, Literal

import polars as pl

from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.domain.models import DatasetSource, SourceType
from tabcaddy.shared.dataset_io import scan_dataframe, scan_parquet_dataset


_SUPPORTED_AGGREGATIONS = {"mean", "median", "min", "max", "sum", "count"}
_NAIVE_UNIX_EPOCH = datetime(1970, 1, 1)
_DEFAULT_FOLDER_MAX_FILES = 5


@dataclass
class PlotResult:
    chart_kind: Literal["line", "scatter"]
    x_column: str
    y_column: str
    x_axis_kind: Literal["numeric", "temporal", "categorical"]
    x_axis_time_unit: Literal["epoch_seconds"] | None
    x_axis_timezone: Literal["UTC"] | None
    row_count: int
    plotted_rows: int
    dropped_rows: int
    duplicate_x_count: int
    sorted_x: bool
    auto_sorted: bool
    aggregated: bool
    line_x_values: list[float] = field(default_factory=list)
    line_values: list[float] = field(default_factory=list)
    scatter_points: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PlotFileResult:
    path: Path
    result: PlotResult


@dataclass
class PlotRunResult:
    plots: list[PlotFileResult]
    total_files: int
    plotted_files: int
    skipped_files: int
    warnings: list[str] = field(default_factory=list)


class PlotDataset:
    def run(
        self,
        source: DatasetSource,
        x_column: str,
        y_column: str,
        *,
        kind: Literal["auto", "line", "scatter"] = "auto",
        aggregate_x: Literal["mean", "median", "min", "max", "sum", "count"]
        | None = None,
        fail_on_x_duplicates: bool = False,
        fail_on_unsorted_x: bool = False,
        folder_max_files: int = _DEFAULT_FOLDER_MAX_FILES,
    ) -> PlotRunResult:
        if folder_max_files < 1:
            raise ValueError("--folder-max-files must be greater than or equal to 1.")

        if source.source_type == SourceType.FOLDER:
            return self._run_for_folder(
                source,
                x_column,
                y_column,
                kind=kind,
                aggregate_x=aggregate_x,
                fail_on_x_duplicates=fail_on_x_duplicates,
                fail_on_unsorted_x=fail_on_unsorted_x,
                folder_max_files=folder_max_files,
            )

        lazyframe = self._build_lazyframe(source)
        plot = self._run_for_lazyframe(
            lazyframe,
            x_column,
            y_column,
            kind=kind,
            aggregate_x=aggregate_x,
            fail_on_x_duplicates=fail_on_x_duplicates,
            fail_on_unsorted_x=fail_on_unsorted_x,
        )

        return PlotRunResult(
            plots=[PlotFileResult(path=source.path, result=plot)],
            total_files=1,
            plotted_files=1,
            skipped_files=0,
        )

    def _run_for_folder(
        self,
        source: DatasetSource,
        x_column: str,
        y_column: str,
        *,
        kind: Literal["auto", "line", "scatter"],
        aggregate_x: Literal["mean", "median", "min", "max", "sum", "count"] | None,
        fail_on_x_duplicates: bool,
        fail_on_unsorted_x: bool,
        folder_max_files: int,
    ) -> PlotRunResult:
        files = iter_dataset_files(source)
        if not files:
            raise ValueError(f"No supported files found under: {source.path}")

        selected_files = files[:folder_max_files]
        file_results: list[PlotFileResult] = []
        for path in selected_files:
            lazyframe = scan_dataframe(path)
            result = self._run_for_lazyframe(
                lazyframe,
                x_column,
                y_column,
                kind=kind,
                aggregate_x=aggregate_x,
                fail_on_x_duplicates=fail_on_x_duplicates,
                fail_on_unsorted_x=fail_on_unsorted_x,
            )
            file_results.append(PlotFileResult(path=path, result=result))

        warnings: list[str] = []
        skipped_files = max(0, len(files) - len(selected_files))
        if skipped_files > 0:
            warnings.append(
                (
                    f"Folder contains {len(files)} files; plotted first {len(selected_files)}. "
                    f"Increase --folder-max-files to at least {len(files)} to plot all files."
                )
            )

        return PlotRunResult(
            plots=file_results,
            total_files=len(files),
            plotted_files=len(file_results),
            skipped_files=skipped_files,
            warnings=warnings,
        )

    def _run_for_lazyframe(
        self,
        lazyframe: pl.LazyFrame,
        x_column: str,
        y_column: str,
        *,
        kind: Literal["auto", "line", "scatter"],
        aggregate_x: Literal["mean", "median", "min", "max", "sum", "count"] | None,
        fail_on_x_duplicates: bool,
        fail_on_unsorted_x: bool,
    ) -> PlotResult:
        schema = lazyframe.collect_schema()
        if x_column not in schema:
            raise ValueError(f"Column not found: {x_column}")
        if y_column not in schema:
            raise ValueError(f"Column not found: {y_column}")

        x_dtype = schema[x_column]
        y_dtype = schema[y_column]
        x_axis_kind = self._infer_x_axis_kind(x_dtype)
        x_axis_time_unit: Literal["epoch_seconds"] | None = None
        x_axis_timezone: Literal["UTC"] | None = None
        if x_axis_kind == "temporal":
            x_axis_time_unit = "epoch_seconds"
            x_axis_timezone = "UTC"

        if y_dtype.is_nested():
            raise ValueError(
                f"Column '{y_column}' is not plottable (nested type: {y_dtype})."
            )

        warnings: list[str] = []

        # Boolean cannot be cast directly to Float64 in Polars; route via Int8 (true=1, false=0).
        y_to_float: pl.Expr
        if y_dtype == pl.Boolean:
            y_to_float = pl.col(y_column).cast(pl.Int8).cast(pl.Float64)
        else:
            y_to_float = pl.col(y_column).cast(pl.Float64, strict=False)

        frame = lazyframe.select(
            [
                pl.col(x_column).alias("_x"),
                pl.col(y_column).alias("_y_raw"),
                y_to_float.alias("_y"),
            ]
        ).collect(engine="auto")

        row_count = frame.height
        cast_failed = frame.filter(
            pl.col("_y_raw").is_not_null() & pl.col("_y").is_null()
        ).height
        if cast_failed:
            warnings.append(
                f"Dropped {cast_failed} rows where '{y_column}' could not be cast to numeric."
            )

        filtered = frame.filter(
            pl.col("_x").is_not_null()
            & pl.col("_y").is_not_null()
            & pl.col("_y").is_finite()
        ).select(["_x", "_y"])

        dropped_rows = row_count - filtered.height
        if filtered.height == 0:
            raise ValueError(
                "No plottable rows remain after filtering null/invalid values."
            )

        duplicate_x_count = self._duplicate_count(filtered)
        sorted_x = self._is_sorted(filtered)

        chart_kind = self._resolve_kind(
            kind,
            x_dtype,
            duplicate_x_count=duplicate_x_count,
            sorted_x=sorted_x,
        )
        if (
            kind == "auto"
            and chart_kind == "scatter"
            and self._is_numeric_dtype(x_dtype)
        ):
            if duplicate_x_count > 0:
                warnings.append(
                    "Auto-selected scatter because numeric x-values contain duplicates."
                )
            elif not sorted_x:
                warnings.append(
                    "Auto-selected scatter because numeric x-values are not monotonic."
                )

        if aggregate_x is not None and aggregate_x not in _SUPPORTED_AGGREGATIONS:
            raise ValueError(
                f"Unsupported aggregation: {aggregate_x}. Use one of: {', '.join(sorted(_SUPPORTED_AGGREGATIONS))}."
            )

        aggregated = False
        if aggregate_x is not None:
            filtered = self._aggregate_by_x(filtered, aggregate_x)
            warnings.append(
                f"Applied aggregation '{aggregate_x}' across duplicate x-values before plotting."
            )
            aggregated = True
            duplicate_x_count = self._duplicate_count(filtered)
            sorted_x = self._is_sorted(filtered)
        elif duplicate_x_count > 0 and (chart_kind == "line" or fail_on_x_duplicates):
            raise ValueError(
                "Duplicate x-values detected. Use --aggregate-x to combine duplicates "
                "or choose --kind scatter."
            )

        auto_sorted = False
        if chart_kind == "line" and not sorted_x:
            if fail_on_unsorted_x:
                raise ValueError(
                    "x-values are not sorted. Re-run without --fail-on-unsorted-x "
                    "to auto-sort, or sort upstream."
                )
            filtered = filtered.sort("_x")
            auto_sorted = True
            sorted_x = True
            warnings.append("x-values were auto-sorted for line plot rendering.")

        if chart_kind == "line":
            line_x_values = self._line_x_values(filtered)
            line_values = [
                float(value)
                for value in filtered.get_column("_y").to_list()
                if value is not None and math.isfinite(float(value))
            ]
            if len(line_x_values) != len(line_values):
                raise ValueError(
                    "Line plots require numeric or temporal x-values. "
                    "Use --kind scatter for categorical x."
                )
            if not line_values:
                raise ValueError("No numeric y-values available for line plot.")
            return PlotResult(
                chart_kind="line",
                x_column=x_column,
                y_column=y_column,
                x_axis_kind=x_axis_kind,
                x_axis_time_unit=x_axis_time_unit,
                x_axis_timezone=x_axis_timezone,
                row_count=row_count,
                plotted_rows=len(line_values),
                dropped_rows=dropped_rows,
                duplicate_x_count=duplicate_x_count,
                sorted_x=sorted_x,
                auto_sorted=auto_sorted,
                aggregated=aggregated,
                line_x_values=line_x_values,
                line_values=line_values,
                warnings=warnings,
            )

        scatter_points, scatter_warnings = self._to_scatter_points(
            filtered, x_dtype=x_dtype
        )
        warnings.extend(scatter_warnings)
        if not scatter_points:
            raise ValueError("No finite points available for scatter plot.")

        return PlotResult(
            chart_kind="scatter",
            x_column=x_column,
            y_column=y_column,
            x_axis_kind=x_axis_kind,
            x_axis_time_unit=x_axis_time_unit,
            x_axis_timezone=x_axis_timezone,
            row_count=row_count,
            plotted_rows=len(scatter_points),
            dropped_rows=dropped_rows,
            duplicate_x_count=duplicate_x_count,
            sorted_x=sorted_x,
            auto_sorted=auto_sorted,
            aggregated=aggregated,
            scatter_points=scatter_points,
            warnings=warnings,
        )

    def _build_lazyframe(self, source: DatasetSource) -> pl.LazyFrame:
        if source.source_type == SourceType.COMPILED_DATASET:
            return scan_parquet_dataset(source.path)
        if source.source_type == SourceType.FILE:
            return scan_dataframe(source.path)

        files = iter_dataset_files(source)
        if not files:
            raise ValueError(f"No supported files found under: {source.path}")
        lazyframes = [scan_dataframe(path) for path in files]
        if len(lazyframes) == 1:
            return lazyframes[0]
        return pl.concat(lazyframes, how="diagonal_relaxed")

    def _resolve_kind(
        self,
        kind: Literal["auto", "line", "scatter"],
        x_dtype: pl.DataType,
        *,
        duplicate_x_count: int,
        sorted_x: bool,
    ) -> Literal["line", "scatter"]:
        if kind in {"line", "scatter"}:
            return kind
        if self._is_temporal_dtype(x_dtype):
            return "line"
        if self._is_numeric_dtype(x_dtype):
            if duplicate_x_count > 0:
                return "scatter"
            if not sorted_x:
                return "scatter"
            return "line"
        return "scatter"

    def _infer_x_axis_kind(
        self, dtype: pl.DataType
    ) -> Literal["numeric", "temporal", "categorical"]:
        if self._is_temporal_dtype(dtype):
            return "temporal"
        if self._is_numeric_dtype(dtype):
            return "numeric"
        return "categorical"

    def _duplicate_count(self, frame: pl.DataFrame) -> int:
        x_values = frame.get_column("_x")
        return max(0, frame.height - x_values.n_unique())

    def _is_sorted(self, frame: pl.DataFrame) -> bool:
        return bool(frame.get_column("_x").is_sorted())

    def _aggregate_by_x(self, frame: pl.DataFrame, aggregate_x: str) -> pl.DataFrame:
        if aggregate_x == "count":
            agg_expr = pl.col("_y").count().cast(pl.Float64).alias("_y")
        else:
            agg_expr = getattr(pl.col("_y"), aggregate_x)().alias("_y")
        return frame.group_by("_x").agg(agg_expr).sort("_x")

    def _to_scatter_points(
        self, frame: pl.DataFrame, *, x_dtype: pl.DataType
    ) -> tuple[list[tuple[float, float]], list[str]]:
        warnings: list[str] = []
        x_values = frame.get_column("_x").to_list()
        y_values = [float(value) for value in frame.get_column("_y").to_list()]

        numeric_x: list[float]
        if self._is_temporal_dtype(x_dtype) or self._is_numeric_dtype(x_dtype):
            numeric_x = []
            for value in x_values:
                converted = self._to_numeric_x(value)
                if converted is None or not math.isfinite(converted):
                    continue
                numeric_x.append(converted)
            if len(numeric_x) != len(y_values):
                # Rebuild aligned points while dropping rows that cannot be projected.
                points = [
                    (x, y)
                    for value, y in zip(x_values, y_values, strict=False)
                    for x in [self._to_numeric_x(value)]
                    if x is not None and math.isfinite(x) and math.isfinite(y)
                ]
                return points, warnings
            points = [
                (x, y)
                for x, y in zip(numeric_x, y_values, strict=False)
                if math.isfinite(x) and math.isfinite(y)
            ]
            return points, warnings

        mapping: dict[str, float] = {}
        numeric_x = []
        for value in x_values:
            key = str(value)
            if key not in mapping:
                mapping[key] = float(len(mapping))
            numeric_x.append(mapping[key])
        warnings.append(
            "Scatter plot encoded categorical x-values to ordinal positions."
        )
        points = [
            (x, y)
            for x, y in zip(numeric_x, y_values, strict=False)
            if math.isfinite(x) and math.isfinite(y)
        ]
        return points, warnings

    def _to_numeric_x(self, value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return (value - _NAIVE_UNIX_EPOCH).total_seconds()
            return value.timestamp()
        if isinstance(value, date):
            return (
                datetime(value.year, value.month, value.day) - _NAIVE_UNIX_EPOCH
            ).total_seconds()
        if isinstance(value, time):
            return (
                value.hour * 3600.0
                + value.minute * 60.0
                + value.second
                + value.microsecond / 1_000_000.0
            )
        return None

    def _line_x_values(self, frame: pl.DataFrame) -> list[float]:
        result: list[float] = []
        for value in frame.get_column("_x").to_list():
            converted = self._to_numeric_x(value)
            if converted is None or not math.isfinite(converted):
                return []
            result.append(converted)
        return result

    def _is_numeric_dtype(self, dtype: pl.DataType) -> bool:
        probe = getattr(dtype, "is_numeric", None)
        if callable(probe):
            return bool(probe())
        if probe is not None:
            return bool(probe)
        return any(token in str(dtype) for token in ("Int", "UInt", "Float", "Decimal"))

    def _is_temporal_dtype(self, dtype: pl.DataType) -> bool:
        probe = getattr(dtype, "is_temporal", None)
        if callable(probe):
            return bool(probe())
        if probe is not None:
            return bool(probe)
        return any(
            token in str(dtype) for token in ("Date", "Datetime", "Time", "Duration")
        )
