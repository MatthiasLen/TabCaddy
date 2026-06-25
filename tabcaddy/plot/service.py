from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
import math
from pathlib import Path
import re
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy import stats

from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.domain.models import DatasetSource, SourceType
from tabcaddy.shared.dataset_io import scan_dataframe, scan_parquet_dataset
from tabcaddy.shared.histogram import build_numeric_histogram


_SUPPORTED_AGGREGATIONS = {"mean", "median", "min", "max", "sum", "count"}
_NAIVE_UNIX_EPOCH = datetime(1970, 1, 1)
_MIN_OUTLIER_POINTS = 8
_MIN_BUCKET_POINTS = 6
_MAX_OUTLIER_BUCKETS = 24
_ROBUST_Z_THRESHOLD = 3.5
_EPSILON = 1e-12
_FILTER_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_-]*)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$"
)
_FILTER_OPERATORS = {
    "==": lambda lhs, rhs: lhs == rhs,
    "!=": lambda lhs, rhs: lhs != rhs,
    ">": lambda lhs, rhs: lhs > rhs,
    ">=": lambda lhs, rhs: lhs >= rhs,
    "<": lambda lhs, rhs: lhs < rhs,
    "<=": lambda lhs, rhs: lhs <= rhs,
}


def _format_epoch_seconds_histogram_bound(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    try:
        dt = datetime.fromtimestamp(value, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return f"{value:.3g}"
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0:
        return dt.strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d %H:%M:%SZ")


@dataclass
class PlotResult:
    chart_kind: Literal["line", "scatter", "histogram"]
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
    line_interpolation: Literal["linear", "nearest"] | None
    line_x_values: list[float] = field(default_factory=list)
    line_values: list[float] = field(default_factory=list)
    scatter_points: list[tuple[float, float]] = field(default_factory=list)
    scatter_inlier_points: list[tuple[float, float]] = field(default_factory=list)
    scatter_outlier_points: list[tuple[float, float]] = field(default_factory=list)
    histogram_bins: list[tuple[str, int]] = field(default_factory=list)
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
        line_interpolation: Literal["linear", "nearest"] = "linear",
        fail_on_x_duplicates: bool = False,
        fail_on_unsorted_x: bool = False,
        folder_max_files: int | None = None,
        filter_expr: str | None = None,
    ) -> PlotRunResult:
        if folder_max_files is not None and folder_max_files < 1:
            raise ValueError("-n must be greater than or equal to 1.")

        if source.source_type == SourceType.FOLDER:
            return self._run_for_folder(
                source,
                x_column,
                y_column,
                kind=kind,
                aggregate_x=aggregate_x,
                line_interpolation=line_interpolation,
                fail_on_x_duplicates=fail_on_x_duplicates,
                fail_on_unsorted_x=fail_on_unsorted_x,
                folder_max_files=1 if folder_max_files is None else folder_max_files,
                filter_expr=filter_expr,
            )

        lazyframe = self._build_lazyframe(source)
        plot = self._run_for_lazyframe(
            lazyframe,
            x_column,
            y_column,
            kind=kind,
            aggregate_x=aggregate_x,
            line_interpolation=line_interpolation,
            fail_on_x_duplicates=fail_on_x_duplicates,
            fail_on_unsorted_x=fail_on_unsorted_x,
            filter_expr=filter_expr,
        )

        return PlotRunResult(
            plots=[PlotFileResult(path=source.path, result=plot)],
            total_files=1,
            plotted_files=1,
            skipped_files=0,
        )

    def run_histogram(
        self,
        source: DatasetSource,
        column: str,
        *,
        folder_max_files: int | None = None,
        filter_expr: str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> PlotRunResult:
        if folder_max_files is not None and folder_max_files < 1:
            raise ValueError("-n must be greater than or equal to 1.")

        if source.source_type == SourceType.FOLDER:
            if folder_max_files is None:
                return self._run_histogram_for_folder_aggregate(
                    source,
                    column,
                    filter_expr=filter_expr,
                    progress_callback=progress_callback,
                )
            return self._run_histogram_for_folder(
                source,
                column,
                folder_max_files=folder_max_files,
                filter_expr=filter_expr,
            )

        lazyframe = self._build_lazyframe(source)
        plot = self._run_histogram_for_lazyframe(
            lazyframe,
            column,
            filter_expr=filter_expr,
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
        line_interpolation: Literal["linear", "nearest"],
        fail_on_x_duplicates: bool,
        fail_on_unsorted_x: bool,
        folder_max_files: int,
        filter_expr: str | None,
    ) -> PlotRunResult:
        files = iter_dataset_files(source)
        if not files:
            raise ValueError(f"No supported files found under: {source.path}")

        if filter_expr is not None:
            self._parse_filter_expression(filter_expr)

        selected_files = files[:folder_max_files]
        file_results: list[PlotFileResult] = []
        warnings: list[str] = []
        failed_files = 0
        for path in selected_files:
            try:
                lazyframe = scan_dataframe(path)
                result = self._run_for_lazyframe(
                    lazyframe,
                    x_column,
                    y_column,
                    kind=kind,
                    aggregate_x=aggregate_x,
                    line_interpolation=line_interpolation,
                    fail_on_x_duplicates=fail_on_x_duplicates,
                    fail_on_unsorted_x=fail_on_unsorted_x,
                    filter_expr=filter_expr,
                )
            except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
                failed_files += 1
                warnings.append(f"Skipped {path.name}: {error}")
                continue

            file_results.append(PlotFileResult(path=path, result=result))

        capped_files = max(0, len(files) - len(selected_files))
        skipped_files = capped_files + failed_files
        if failed_files > 0:
            warnings.insert(
                0,
                f"Skipped {failed_files} file(s) due to plotting errors.",
            )
        if capped_files > 0:
            warnings.append(
                (
                    f"Folder contains {len(files)} files; plotted first {len(selected_files)}. "
                    f"Increase -n to at least {len(files)} to plot all files."
                )
            )

        return PlotRunResult(
            plots=file_results,
            total_files=len(files),
            plotted_files=len(file_results),
            skipped_files=skipped_files,
            warnings=warnings,
        )

    def _run_histogram_for_folder_aggregate(
        self,
        source: DatasetSource,
        column: str,
        *,
        filter_expr: str | None,
        progress_callback: Callable[[int, int], None] | None,
    ) -> PlotRunResult:
        files = iter_dataset_files(source)
        if not files:
            raise ValueError(f"No supported files found under: {source.path}")

        if filter_expr is not None:
            self._parse_filter_expression(filter_expr)

        selected_lazyframes: list[pl.LazyFrame] = []
        missing_column_files: list[str] = []
        failed_files: list[str] = []
        total_files = len(files)
        for index, path in enumerate(files, start=1):
            try:
                lazyframe = scan_dataframe(path)
                schema = lazyframe.collect_schema()
            except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
                failed_files.append(f"Skipped {path.name}: {error}")
                if progress_callback is not None:
                    progress_callback(index, total_files)
                continue

            if column not in schema:
                missing_column_files.append(path.name)
                if progress_callback is not None:
                    progress_callback(index, total_files)
                continue

            column_dtype = schema[column]
            if column_dtype.is_nested():
                failed_files.append(
                    (
                        f"Skipped {path.name}: Column '{column}' is not plottable "
                        f"(nested type: {column_dtype})."
                    )
                )
                if progress_callback is not None:
                    progress_callback(index, total_files)
                continue

            selected_lazyframes.append(lazyframe)
            if progress_callback is not None:
                progress_callback(index, total_files)

        if not selected_lazyframes:
            raise ValueError(f"Column not found in any folder file: {column}")

        if len(selected_lazyframes) == 1:
            combined = selected_lazyframes[0]
        else:
            combined = pl.concat(selected_lazyframes, how="diagonal_relaxed")

        plot = self._run_histogram_for_lazyframe(
            combined,
            column,
            filter_expr=filter_expr,
        )

        warning_messages: list[str] = []
        if missing_column_files:
            sample = ", ".join(missing_column_files[:3])
            remainder = len(missing_column_files) - 3
            suffix = f", +{remainder} more" if remainder > 0 else ""
            warning_messages.append(
                (
                    f"Skipped {len(missing_column_files)} file(s) missing "
                    f"column '{column}': {sample}{suffix}."
                )
            )
        if failed_files:
            warning_messages.append(
                f"Skipped {len(failed_files)} file(s) due to plotting errors."
            )
            warning_messages.extend(failed_files)

        if warning_messages:
            plot.warnings = warning_messages + plot.warnings

        return PlotRunResult(
            plots=[PlotFileResult(path=source.path, result=plot)],
            total_files=len(files),
            plotted_files=1,
            skipped_files=0,
            warnings=[],
        )

    def _run_histogram_for_folder(
        self,
        source: DatasetSource,
        column: str,
        *,
        folder_max_files: int,
        filter_expr: str | None,
    ) -> PlotRunResult:
        files = iter_dataset_files(source)
        if not files:
            raise ValueError(f"No supported files found under: {source.path}")

        if filter_expr is not None:
            self._parse_filter_expression(filter_expr)

        selected_files = files[:folder_max_files]
        file_results: list[PlotFileResult] = []
        warnings: list[str] = []
        failed_files = 0
        for path in selected_files:
            try:
                lazyframe = scan_dataframe(path)
                result = self._run_histogram_for_lazyframe(
                    lazyframe,
                    column,
                    filter_expr=filter_expr,
                )
            except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
                failed_files += 1
                warnings.append(f"Skipped {path.name}: {error}")
                continue

            file_results.append(PlotFileResult(path=path, result=result))

        capped_files = max(0, len(files) - len(selected_files))
        skipped_files = capped_files + failed_files
        if failed_files > 0:
            warnings.insert(
                0,
                f"Skipped {failed_files} file(s) due to plotting errors.",
            )
        if capped_files > 0:
            warnings.append(
                (
                    f"Folder contains {len(files)} files; plotted first {len(selected_files)}. "
                    f"Increase -n to at least {len(files)} to plot all files."
                )
            )

        return PlotRunResult(
            plots=file_results,
            total_files=len(files),
            plotted_files=len(file_results),
            skipped_files=skipped_files,
            warnings=warnings,
        )

    def _run_histogram_for_lazyframe(
        self,
        lazyframe: pl.LazyFrame,
        column: str,
        *,
        filter_expr: str | None,
    ) -> PlotResult:
        schema = lazyframe.collect_schema()
        if column not in schema:
            raise ValueError(f"Column not found: {column}")

        column_dtype = schema[column]
        if column_dtype.is_nested():
            raise ValueError(
                f"Column '{column}' is not plottable (nested type: {column_dtype})."
            )
        x_axis_kind, x_axis_time_unit, x_axis_timezone = self._axis_metadata(
            column_dtype
        )

        working = lazyframe
        warnings: list[str] = []
        if filter_expr is not None:
            working = lazyframe.filter(
                self._build_filter_predicate(filter_expr, schema=schema)
            )

        frame = working.select([pl.col(column).alias("_value_raw")]).collect(
            engine="auto"
        )
        row_count = frame.height

        raw_values = frame.get_column("_value_raw").to_list()
        if x_axis_kind == "temporal":
            converted_values = [self._to_numeric_x(value) for value in raw_values]
        else:
            cast_values = (
                frame.get_column("_value_raw").cast(pl.Float64, strict=False).to_list()
            )
            converted_values = [
                float(value) if value is not None else None for value in cast_values
            ]

        cast_failed = sum(
            1
            for raw_value, converted in zip(raw_values, converted_values, strict=False)
            if raw_value is not None and converted is None
        )
        if cast_failed > 0:
            warnings.append(
                (
                    f"Dropped {cast_failed} rows where '{column}' could not be cast to "
                    "numeric/temporal values for histogram."
                )
            )

        filtered_values = [
            value
            for value in converted_values
            if value is not None and math.isfinite(value)
        ]
        plotted_rows = len(filtered_values)
        dropped_rows = row_count - plotted_rows
        if plotted_rows == 0:
            raise ValueError(
                f"No plottable numeric values remain for histogram column '{column}'."
            )

        bound_formatter: Callable[[float], str] | None = None
        if x_axis_time_unit == "epoch_seconds" and x_axis_timezone == "UTC":
            bound_formatter = _format_epoch_seconds_histogram_bound

        bins = build_numeric_histogram(
            filtered_values,
            bound_formatter=bound_formatter,
        )
        if not bins:
            raise ValueError(
                f"Unable to build histogram for column '{column}' from finite numeric values."
            )

        return PlotResult(
            chart_kind="histogram",
            x_column=column,
            y_column=column,
            x_axis_kind=x_axis_kind,
            x_axis_time_unit=x_axis_time_unit,
            x_axis_timezone=x_axis_timezone,
            row_count=row_count,
            plotted_rows=plotted_rows,
            dropped_rows=dropped_rows,
            duplicate_x_count=0,
            sorted_x=False,
            auto_sorted=False,
            aggregated=False,
            line_interpolation=None,
            histogram_bins=bins,
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
        line_interpolation: Literal["linear", "nearest"],
        fail_on_x_duplicates: bool,
        fail_on_unsorted_x: bool,
        filter_expr: str | None,
    ) -> PlotResult:
        schema = lazyframe.collect_schema()
        x_dtype, y_dtype = self._resolve_plot_dtypes(
            schema, x_column=x_column, y_column=y_column
        )
        x_axis_kind, x_axis_time_unit, x_axis_timezone = self._axis_metadata(x_dtype)
        filtered, row_count, dropped_rows, warnings = self._prepare_plot_frame(
            lazyframe,
            x_column=x_column,
            y_column=y_column,
            y_dtype=y_dtype,
            filter_expr=filter_expr,
            schema=schema,
        )

        duplicate_x_count = self._duplicate_count(filtered)
        sorted_x = self._is_sorted(filtered)
        chart_kind = self._resolve_kind(
            kind,
            x_dtype,
            duplicate_x_count=duplicate_x_count,
            sorted_x=sorted_x,
        )
        warnings.extend(
            self._auto_kind_warnings(
                kind,
                chart_kind,
                x_dtype,
                duplicate_x_count=duplicate_x_count,
                sorted_x=sorted_x,
            )
        )

        filtered, duplicate_x_count, sorted_x, aggregated = self._apply_x_handling(
            filtered,
            chart_kind=chart_kind,
            aggregate_x=aggregate_x,
            fail_on_x_duplicates=fail_on_x_duplicates,
            warnings=warnings,
        )
        filtered, sorted_x, auto_sorted = self._prepare_line_order(
            filtered,
            chart_kind=chart_kind,
            sorted_x=sorted_x,
            fail_on_unsorted_x=fail_on_unsorted_x,
            warnings=warnings,
        )

        if chart_kind == "line":
            return self._build_line_result(
                filtered,
                x_column=x_column,
                y_column=y_column,
                x_axis_kind=x_axis_kind,
                x_axis_time_unit=x_axis_time_unit,
                x_axis_timezone=x_axis_timezone,
                row_count=row_count,
                dropped_rows=dropped_rows,
                duplicate_x_count=duplicate_x_count,
                sorted_x=sorted_x,
                auto_sorted=auto_sorted,
                aggregated=aggregated,
                line_interpolation=line_interpolation,
                warnings=warnings,
            )
        return self._build_scatter_result(
            filtered,
            x_dtype=x_dtype,
            x_column=x_column,
            y_column=y_column,
            x_axis_kind=x_axis_kind,
            x_axis_time_unit=x_axis_time_unit,
            x_axis_timezone=x_axis_timezone,
            row_count=row_count,
            dropped_rows=dropped_rows,
            duplicate_x_count=duplicate_x_count,
            sorted_x=sorted_x,
            auto_sorted=auto_sorted,
            aggregated=aggregated,
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
            if duplicate_x_count > 0:
                return "scatter"
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

    def _resolve_plot_dtypes(
        self,
        schema: pl.Schema,
        *,
        x_column: str,
        y_column: str,
    ) -> tuple[pl.DataType, pl.DataType]:
        if x_column not in schema:
            raise ValueError(f"Column not found: {x_column}")
        if y_column not in schema:
            raise ValueError(f"Column not found: {y_column}")

        x_dtype = schema[x_column]
        y_dtype = schema[y_column]
        if y_dtype.is_nested():
            raise ValueError(
                f"Column '{y_column}' is not plottable (nested type: {y_dtype})."
            )
        return x_dtype, y_dtype

    def _axis_metadata(
        self, x_dtype: pl.DataType
    ) -> tuple[
        Literal["numeric", "temporal", "categorical"],
        Literal["epoch_seconds"] | None,
        Literal["UTC"] | None,
    ]:
        x_axis_kind = self._infer_x_axis_kind(x_dtype)
        if x_axis_kind == "temporal":
            if self._is_time_dtype(x_dtype):
                return x_axis_kind, None, None
            if self._is_duration_dtype(x_dtype):
                return x_axis_kind, None, None
            return x_axis_kind, "epoch_seconds", "UTC"
        return x_axis_kind, None, None

    def _is_duration_dtype(self, dtype: pl.DataType) -> bool:
        if dtype == pl.Duration:
            return True
        base_type = getattr(dtype, "base_type", None)
        if callable(base_type):
            try:
                return base_type() == pl.Duration
            except TypeError:
                return False
        return "Duration" in str(dtype)

    def _is_datetime_dtype(self, dtype: pl.DataType) -> bool:
        if dtype == pl.Datetime:
            return True
        base_type = getattr(dtype, "base_type", None)
        if callable(base_type):
            try:
                return base_type() == pl.Datetime
            except TypeError:
                return False
        return "Datetime" in str(dtype)

    def _is_time_dtype(self, dtype: pl.DataType) -> bool:
        if dtype == pl.Time:
            return True
        base_type = getattr(dtype, "base_type", None)
        if callable(base_type):
            try:
                return base_type() == pl.Time
            except TypeError:
                return False
        return "Time" in str(dtype)

    def _prepare_plot_frame(
        self,
        lazyframe: pl.LazyFrame,
        *,
        x_column: str,
        y_column: str,
        y_dtype: pl.DataType,
        filter_expr: str | None,
        schema: pl.Schema,
    ) -> tuple[pl.DataFrame, int, int, list[str]]:
        warnings: list[str] = []
        filtered_lazyframe = lazyframe
        if filter_expr is not None:
            filtered_lazyframe = lazyframe.filter(
                self._build_filter_predicate(filter_expr, schema=schema)
            )
        y_to_float = self._y_to_float_expr(y_column, y_dtype)
        frame = filtered_lazyframe.select(
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
        if filtered.height == 0:
            raise ValueError(
                "No plottable rows remain after filtering null/invalid values."
            )
        return filtered, row_count, row_count - filtered.height, warnings

    def _y_to_float_expr(self, y_column: str, y_dtype: pl.DataType) -> pl.Expr:
        # Boolean cannot be cast directly to Float64 in Polars; route via Int8.
        if y_dtype == pl.Boolean:
            return pl.col(y_column).cast(pl.Int8).cast(pl.Float64)
        return pl.col(y_column).cast(pl.Float64, strict=False)

    def _build_filter_predicate(self, expression: str, *, schema: pl.Schema) -> pl.Expr:
        column, operator, raw_value = self._parse_filter_expression(expression)
        if column not in schema:
            raise ValueError(f"Column not found in --filter: {column}")
        column_dtype = schema[column]
        return _FILTER_OPERATORS[operator](
            pl.col(column),
            self._parse_filter_value(raw_value, dtype=column_dtype),
        )

    def _parse_filter_expression(self, expression: str) -> tuple[str, str, str]:
        match = _FILTER_PATTERN.match(expression)
        if match is None:
            raise ValueError(
                "Invalid --filter expression. Expected format: COLUMN OP VALUE "
                "with OP in ==, !=, >, >=, <, <=."
            )
        return match.group(1), match.group(2), match.group(3)

    def _parse_filter_value(
        self, value: str, *, dtype: pl.DataType
    ) -> bool | int | float | str | date | datetime | time:
        stripped = value.strip()
        if (
            len(stripped) >= 2
            and stripped[0] == stripped[-1]
            and stripped[0]
            in {
                '"',
                "'",
            }
        ):
            stripped = stripped[1:-1]

        if self._is_string_like_dtype(dtype):
            return stripped

        if dtype == pl.Date:
            try:
                return date.fromisoformat(stripped)
            except ValueError as error:
                raise ValueError(
                    "Invalid --filter value for Date column. Expected ISO-8601 date "
                    "(YYYY-MM-DD)."
                ) from error

        if self._is_time_dtype(dtype):
            try:
                parsed_time = time.fromisoformat(stripped)
            except ValueError as error:
                raise ValueError(
                    "Invalid --filter value for Time column. Expected ISO-8601 "
                    "time (HH:MM[:SS[.ffffff]])."
                ) from error
            if parsed_time.tzinfo is not None:
                raise ValueError(
                    "Invalid --filter value for Time column. Time literals with "
                    "timezone offsets are not supported."
                )
            return parsed_time

        if self._is_datetime_dtype(dtype):
            try:
                parsed = datetime.fromisoformat(stripped)
            except ValueError as error:
                raise ValueError(
                    "Invalid --filter value for Datetime column. Expected ISO-8601 "
                    "datetime (for example 2026-01-01T12:34:56 or 2026-01-01)."
                ) from error
            timezone_name = getattr(dtype, "time_zone", None)
            if timezone_name:
                if parsed.tzinfo is None:
                    if timezone_name == "UTC":
                        return parsed.replace(tzinfo=UTC)
                    try:
                        return parsed.replace(tzinfo=ZoneInfo(timezone_name))
                    except Exception as error:
                        raise ValueError(
                            "Invalid timezone metadata for Datetime filter comparison: "
                            f"{timezone_name}"
                        ) from error
                if timezone_name == "UTC":
                    return parsed.astimezone(UTC)
                try:
                    return parsed.astimezone(ZoneInfo(timezone_name))
                except Exception as error:
                    raise ValueError(
                        "Invalid timezone metadata for Datetime filter comparison: "
                        f"{timezone_name}"
                    ) from error
            if parsed.tzinfo is not None:
                return parsed.astimezone(UTC).replace(tzinfo=None)
            return parsed

        if dtype == pl.Boolean:
            lowered = stripped.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            raise ValueError(
                "Invalid --filter value for Boolean column. Expected true or false."
            )

        if self._is_numeric_dtype(dtype):
            if "Decimal" in str(dtype):
                try:
                    return Decimal(stripped)
                except Exception:
                    # Fall back to integer/float parsing for compatibility.
                    pass
            for parser in (int, float):
                try:
                    return parser(stripped)
                except ValueError:
                    continue
            raise ValueError(
                "Invalid --filter value for numeric column. Expected a numeric literal."
            )

        lowered = stripped.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"

        for parser in (int, float):
            try:
                return parser(stripped)
            except ValueError:
                continue
        return stripped

    def _auto_kind_warnings(
        self,
        kind: Literal["auto", "line", "scatter"],
        chart_kind: Literal["line", "scatter"],
        x_dtype: pl.DataType,
        *,
        duplicate_x_count: int,
        sorted_x: bool,
    ) -> list[str]:
        if kind != "auto" or chart_kind != "scatter":
            return []
        is_numeric_x = self._is_numeric_dtype(x_dtype)
        is_temporal_x = self._is_temporal_dtype(x_dtype)
        if not is_numeric_x and not is_temporal_x:
            return []
        if duplicate_x_count > 0:
            if is_temporal_x:
                return [
                    "Auto-selected scatter because temporal x-values contain duplicates."
                ]
            return [
                "Auto-selected scatter because numeric x-values contain duplicates."
            ]
        if is_numeric_x and not sorted_x:
            return ["Auto-selected scatter because numeric x-values are not monotonic."]
        return []

    def _apply_x_handling(
        self,
        frame: pl.DataFrame,
        *,
        chart_kind: Literal["line", "scatter"],
        aggregate_x: Literal["mean", "median", "min", "max", "sum", "count"] | None,
        fail_on_x_duplicates: bool,
        warnings: list[str],
    ) -> tuple[pl.DataFrame, int, bool, bool]:
        duplicate_x_count = self._duplicate_count(frame)
        sorted_x = self._is_sorted(frame)

        if aggregate_x is not None:
            if aggregate_x not in _SUPPORTED_AGGREGATIONS:
                raise ValueError(
                    f"Unsupported aggregation: {aggregate_x}. Use one of: {', '.join(sorted(_SUPPORTED_AGGREGATIONS))}."
                )
            frame = self._aggregate_by_x(frame, aggregate_x)
            warnings.append(
                f"Applied aggregation '{aggregate_x}' across duplicate x-values before plotting."
            )
            return frame, self._duplicate_count(frame), self._is_sorted(frame), True

        if duplicate_x_count > 0 and (chart_kind == "line" or fail_on_x_duplicates):
            raise ValueError(
                "Duplicate x-values detected. Use --aggregate-x to combine duplicates "
                "or choose --kind scatter."
            )
        return frame, duplicate_x_count, sorted_x, False

    def _prepare_line_order(
        self,
        frame: pl.DataFrame,
        *,
        chart_kind: Literal["line", "scatter"],
        sorted_x: bool,
        fail_on_unsorted_x: bool,
        warnings: list[str],
    ) -> tuple[pl.DataFrame, bool, bool]:
        if chart_kind != "line" or sorted_x:
            return frame, sorted_x, False
        if fail_on_unsorted_x:
            raise ValueError(
                "x-values are not sorted. Re-run without --fail-on-x-unsorted "
                "to auto-sort, or sort upstream."
            )
        warnings.append("x-values were auto-sorted for line plot rendering.")
        return frame.sort("_x"), True, True

    def _build_line_result(
        self,
        frame: pl.DataFrame,
        *,
        x_column: str,
        y_column: str,
        x_axis_kind: Literal["numeric", "temporal", "categorical"],
        x_axis_time_unit: Literal["epoch_seconds"] | None,
        x_axis_timezone: Literal["UTC"] | None,
        row_count: int,
        dropped_rows: int,
        duplicate_x_count: int,
        sorted_x: bool,
        auto_sorted: bool,
        aggregated: bool,
        line_interpolation: Literal["linear", "nearest"],
        warnings: list[str],
    ) -> PlotResult:
        if x_axis_kind == "categorical":
            raise ValueError(
                "Line plots require numeric or temporal x-values. "
                "Use --kind scatter for categorical x."
            )

        line_x_values, line_values, dropped_x_rows = self._line_points(frame)
        if dropped_x_rows > 0:
            warnings.append(
                f"Line plot dropped {dropped_x_rows} rows with non-plottable x-values."
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
            dropped_rows=dropped_rows + dropped_x_rows,
            duplicate_x_count=duplicate_x_count,
            sorted_x=sorted_x,
            auto_sorted=auto_sorted,
            aggregated=aggregated,
            line_interpolation=line_interpolation,
            line_x_values=line_x_values,
            line_values=line_values,
            warnings=warnings,
        )

    def _build_scatter_result(
        self,
        frame: pl.DataFrame,
        *,
        x_dtype: pl.DataType,
        x_column: str,
        y_column: str,
        x_axis_kind: Literal["numeric", "temporal", "categorical"],
        x_axis_time_unit: Literal["epoch_seconds"] | None,
        x_axis_timezone: Literal["UTC"] | None,
        row_count: int,
        dropped_rows: int,
        duplicate_x_count: int,
        sorted_x: bool,
        auto_sorted: bool,
        aggregated: bool,
        warnings: list[str],
    ) -> PlotResult:
        scatter_points, scatter_warnings = self._to_scatter_points(
            frame, x_dtype=x_dtype
        )
        warnings.extend(scatter_warnings)
        if not scatter_points:
            raise ValueError("No finite points available for scatter plot.")
        scatter_inlier_points, scatter_outlier_points = self._split_scatter_outliers(
            scatter_points
        )
        if scatter_outlier_points:
            warnings.append(
                (
                    "Scatter classified "
                    f"{len(scatter_outlier_points)} outlier points "
                    "using robust local y-support."
                )
            )
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
            line_interpolation=None,
            scatter_points=scatter_points,
            scatter_inlier_points=scatter_inlier_points,
            scatter_outlier_points=scatter_outlier_points,
            warnings=warnings,
        )

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

        if self._is_temporal_dtype(x_dtype) or self._is_numeric_dtype(x_dtype):
            points = [
                (x, y)
                for value, y in zip(x_values, y_values, strict=False)
                for x in [self._to_numeric_x(value)]
                if x is not None
                if math.isfinite(x) and math.isfinite(y)
            ]
            if len(points) != len(y_values):
                warnings.append(
                    "Scatter plot dropped rows with non-plottable numeric/temporal x-values."
                )
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
        if isinstance(value, timedelta):
            return value.total_seconds()
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

    def _split_scatter_outliers(
        self, points: list[tuple[float, float]]
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        if len(points) < _MIN_OUTLIER_POINTS:
            return points, []

        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        if math.isclose(min_x, max_x, rel_tol=1e-9, abs_tol=1e-9):
            return self._split_points_by_mask(
                points,
                self._robust_outlier_mask([y for _, y in points]),
            )

        bucket_count = min(_MAX_OUTLIER_BUCKETS, max(1, int(math.sqrt(len(points)))))
        bucket_width = (max_x - min_x) / bucket_count
        if bucket_width <= 0:
            return self._split_points_by_mask(
                points,
                self._robust_outlier_mask([y for _, y in points]),
            )

        buckets: list[list[tuple[int, float]]] = [[] for _ in range(bucket_count)]
        for index, (x, y) in enumerate(points):
            bucket_index = min(
                bucket_count - 1,
                max(0, int((x - min_x) / bucket_width)),
            )
            buckets[bucket_index].append((index, y))

        outlier_mask = [False] * len(points)
        sparse_indices: list[int] = []
        for bucket in buckets:
            if len(bucket) < _MIN_BUCKET_POINTS:
                sparse_indices.extend(index for index, _ in bucket)
                continue
            bucket_values = [value for _, value in bucket]
            bucket_mask = self._robust_outlier_mask(
                bucket_values,
                min_points=_MIN_BUCKET_POINTS,
            )
            for (index, _), is_outlier in zip(bucket, bucket_mask, strict=False):
                if is_outlier:
                    outlier_mask[index] = True

        if len(sparse_indices) >= _MIN_OUTLIER_POINTS:
            sparse_values = [points[index][1] for index in sparse_indices]
            sparse_mask = self._robust_outlier_mask(sparse_values)
            for index, is_outlier in zip(sparse_indices, sparse_mask, strict=False):
                if is_outlier:
                    outlier_mask[index] = True

        return self._split_points_by_mask(points, outlier_mask)

    def _robust_outlier_mask(
        self,
        values: list[float],
        *,
        min_points: int = _MIN_OUTLIER_POINTS,
    ) -> list[bool]:
        if len(values) < min_points:
            return [False] * len(values)

        values_array = np.asarray(values, dtype=float)
        median = float(np.nanmedian(values_array))
        mad = float(
            stats.median_abs_deviation(
                values_array,
                scale="normal",
                nan_policy="omit",
            )
        )
        if math.isfinite(mad) and mad > _EPSILON:
            robust_z = np.abs((values_array - median) / mad)
            return (robust_z > _ROBUST_Z_THRESHOLD).tolist()

        iqr = float(stats.iqr(values_array, rng=(25, 75), nan_policy="omit"))
        if iqr <= _EPSILON:
            return [False] * len(values)
        q1 = float(np.nanpercentile(values_array, 25))
        q3 = float(np.nanpercentile(values_array, 75))
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return ((values_array < lower) | (values_array > upper)).tolist()

    def _split_points_by_mask(
        self,
        points: list[tuple[float, float]],
        outlier_mask: list[bool],
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        inliers: list[tuple[float, float]] = []
        outliers: list[tuple[float, float]] = []
        for point, is_outlier in zip(points, outlier_mask, strict=False):
            if is_outlier:
                outliers.append(point)
            else:
                inliers.append(point)
        return inliers, outliers

    def _line_points(self, frame: pl.DataFrame) -> tuple[list[float], list[float], int]:
        x_values = frame.get_column("_x").to_list()
        y_values = frame.get_column("_y").to_list()

        plotted_x: list[float] = []
        plotted_y: list[float] = []
        dropped = 0
        for x_value, y_value in zip(x_values, y_values, strict=False):
            converted_x = self._to_numeric_x(x_value)
            if converted_x is None or not math.isfinite(converted_x):
                dropped += 1
                continue

            numeric_y = float(y_value)
            if not math.isfinite(numeric_y):
                dropped += 1
                continue

            plotted_x.append(converted_x)
            plotted_y.append(numeric_y)

        return plotted_x, plotted_y, dropped

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

    def _is_string_like_dtype(self, dtype: pl.DataType) -> bool:
        probe = getattr(dtype, "is_string", None)
        if callable(probe):
            return bool(probe())
        if probe is not None:
            return bool(probe)
        return any(
            token in str(dtype)
            for token in (
                "String",
                "Utf8",
                "Categorical",
                "Enum",
            )
        )
