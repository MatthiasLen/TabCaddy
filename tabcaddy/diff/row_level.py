from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
import math
from pathlib import Path
from typing import Any

import polars as pl

from tabcaddy.domain.models import RowChangeExample, RowDiffSummary, RowFieldDelta


@dataclass(frozen=True)
class RowDiffResult:
    summary: RowDiffSummary
    updated_examples: list[RowChangeExample]
    added_key_samples: list[dict[str, Any]]
    removed_key_samples: list[dict[str, Any]]


def compare_rows_by_key(
    left_path: Path,
    right_path: Path,
    key_columns: tuple[str, ...],
    *,
    max_examples: int,
    source_path: str | None = None,
) -> RowDiffResult:
    if not key_columns:
        raise ValueError("Row-level diff requires at least one key column.")
    _validate_unique_keys(key_columns)

    left = _scan_frame(left_path)
    right = _scan_frame(right_path)
    left_schema = left.collect_schema()
    right_schema = right.collect_schema()

    _validate_keys_exist(
        left_schema,
        right_schema,
        key_columns,
        left_path,
        right_path,
    )
    _validate_key_dtypes(left_schema, right_schema, key_columns)
    _raise_on_duplicate_keys(left, key_columns, left_path)
    _raise_on_duplicate_keys(right, key_columns, right_path)

    payload_columns = sorted(
        (set(left_schema.names()) | set(right_schema.names())) - set(key_columns)
    )

    left_aligned = _align_payload(left, left_schema, key_columns, payload_columns)
    right_aligned = _align_payload(right, right_schema, key_columns, payload_columns)

    left_keys = left_aligned.select(list(key_columns))
    right_keys = right_aligned.select(list(key_columns))

    added_keys = right_keys.join(
        left_keys, on=list(key_columns), how="anti", nulls_equal=True
    )
    removed_keys = left_keys.join(
        right_keys, on=list(key_columns), how="anti", nulls_equal=True
    )
    common_keys = left_keys.join(
        right_keys, on=list(key_columns), how="inner", nulls_equal=True
    )

    left_common = left_aligned.join(
        common_keys, on=list(key_columns), how="inner", nulls_equal=True
    ).rename({column: f"left__{column}" for column in payload_columns})
    right_common = right_aligned.join(
        common_keys, on=list(key_columns), how="inner", nulls_equal=True
    ).rename({column: f"right__{column}" for column in payload_columns})

    merged = left_common.join(
        right_common,
        on=list(key_columns),
        how="inner",
        nulls_equal=True,
    )

    changed_expr = _build_changed_row_expr(
        payload_columns,
        left_schema,
        right_schema,
    )

    common_rows = _count_rows(merged)
    updated_rows = _count_rows(merged.filter(changed_expr))
    unchanged_rows = common_rows - updated_rows

    added_rows = _count_rows(added_keys)
    removed_rows = _count_rows(removed_keys)

    updated_sample = (
        merged.filter(changed_expr)
        .select(
            [
                *[pl.col(column) for column in key_columns],
                *[pl.col(f"left__{column}") for column in payload_columns],
                *[pl.col(f"right__{column}") for column in payload_columns],
            ]
        )
        .limit(max_examples)
        .collect(engine="streaming")
    )

    updated_examples: list[RowChangeExample] = []
    for row in updated_sample.iter_rows(named=True):
        deltas: list[RowFieldDelta] = []
        for column in payload_columns:
            left_value = row.get(f"left__{column}")
            right_value = row.get(f"right__{column}")
            if _values_differ(left_value, right_value):
                deltas.append(
                    RowFieldDelta(
                        column=column,
                        left_value=_normalize_value(left_value),
                        right_value=_normalize_value(right_value),
                    )
                )
        updated_examples.append(
            RowChangeExample(
                key={
                    column: _normalize_value(row.get(column)) for column in key_columns
                },
                deltas=deltas,
                source_path=source_path,
            )
        )

    summary = RowDiffSummary(
        key_columns=list(key_columns),
        added_rows=added_rows,
        removed_rows=removed_rows,
        updated_rows=updated_rows,
        unchanged_rows=unchanged_rows,
    )

    return RowDiffResult(
        summary=summary,
        updated_examples=updated_examples,
        added_key_samples=_key_samples(added_keys, key_columns, max_examples),
        removed_key_samples=_key_samples(removed_keys, key_columns, max_examples),
    )


def _validate_keys_exist(
    left_schema: pl.Schema,
    right_schema: pl.Schema,
    key_columns: tuple[str, ...],
    left_path: Path,
    right_path: Path,
) -> None:
    left_columns = set(left_schema.names())
    right_columns = set(right_schema.names())
    left_missing = [column for column in key_columns if column not in left_columns]
    right_missing = [column for column in key_columns if column not in right_columns]
    if not left_missing and not right_missing:
        return
    raise ValueError(
        "Missing key columns for row-level diff. "
        f"left={left_path}: {left_missing or 'none'}, "
        f"right={right_path}: {right_missing or 'none'}"
    )


def _validate_unique_keys(key_columns: tuple[str, ...]) -> None:
    duplicates: list[str] = []
    seen: set[str] = set()
    for column in key_columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    if not duplicates:
        return
    raise ValueError(
        "Duplicate key columns for row-level diff: " + ", ".join(duplicates)
    )


def _validate_key_dtypes(
    left_schema: pl.Schema,
    right_schema: pl.Schema,
    key_columns: tuple[str, ...],
) -> None:
    mismatches: list[str] = []
    for column in key_columns:
        left_dtype = left_schema.get(column)
        right_dtype = right_schema.get(column)
        if left_dtype != right_dtype:
            mismatches.append(f"{column}: {left_dtype} vs {right_dtype}")
    if not mismatches:
        return
    raise ValueError(
        "Incompatible key column types for row-level diff: " + "; ".join(mismatches)
    )


def _raise_on_duplicate_keys(
    frame: pl.LazyFrame,
    key_columns: tuple[str, ...],
    source_path: Path,
) -> None:
    duplicate = (
        frame.lazy()
        .group_by(list(key_columns))
        .agg(pl.len().alias("_row_count"))
        .filter(pl.col("_row_count") > 1)
        .limit(1)
        .collect(engine="streaming")
    )
    if duplicate.is_empty():
        return

    values = duplicate.row(0, named=True)
    key_values = ", ".join(f"{column}={values[column]!r}" for column in key_columns)
    raise ValueError(
        f"Duplicate key rows detected in {source_path}: {key_values}. "
        "Row-level diff requires unique keys."
    )


def _align_payload(
    frame: pl.LazyFrame,
    schema: pl.Schema,
    key_columns: tuple[str, ...],
    payload_columns: list[str],
) -> pl.LazyFrame:
    available = set(schema.names())
    select_exprs: list[pl.Expr] = [pl.col(column) for column in key_columns]
    for column in payload_columns:
        if column in available:
            select_exprs.append(pl.col(column))
        else:
            select_exprs.append(pl.lit(None).alias(column))
    return frame.select(select_exprs)


def _key_samples(
    keys_frame: pl.LazyFrame,
    key_columns: tuple[str, ...],
    max_examples: int,
) -> list[dict[str, Any]]:
    sample = keys_frame.limit(max_examples).collect(engine="streaming")
    return [
        {
            column: _normalize_value(value)
            for column, value in zip(key_columns, row, strict=True)
        }
        for row in sample.iter_rows()
    ]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _values_differ(left_value: Any, right_value: Any) -> bool:
    if left_value is None and right_value is None:
        return False
    if isinstance(left_value, float) and isinstance(right_value, float):
        if math.isnan(left_value) and math.isnan(right_value):
            return False
    return left_value != right_value


def _build_changed_row_expr(
    payload_columns: list[str],
    left_schema: pl.Schema,
    right_schema: pl.Schema,
) -> pl.Expr:
    changed_exprs: list[pl.Expr] = []
    left_dtypes = left_schema
    right_dtypes = right_schema
    for column in payload_columns:
        left_col = pl.col(f"left__{column}")
        right_col = pl.col(f"right__{column}")
        left_dtype = left_dtypes.get(column)
        right_dtype = right_dtypes.get(column)

        if left_dtype == right_dtype:
            equal_expr = left_col.eq_missing(right_col)
            if left_dtype in {pl.Float32, pl.Float64}:
                equal_expr = equal_expr | (left_col.is_nan() & right_col.is_nan())
        else:
            # Different physical dtypes should still compare deterministically.
            equal_expr = left_col.cast(pl.Utf8, strict=False).eq_missing(
                right_col.cast(pl.Utf8, strict=False)
            )

        changed_exprs.append(~equal_expr)

    if not changed_exprs:
        return pl.lit(False)
    return pl.any_horizontal(changed_exprs)


def _count_rows(frame: pl.LazyFrame) -> int:
    return frame.select(pl.len().alias("rows")).collect(engine="streaming").item()


def _scan_frame(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.scan_csv(str(path), infer_schema_length=1000, try_parse_dates=True)
    if suffix == ".parquet":
        return pl.scan_parquet(str(path))
    if suffix in {".feather", ".arrow"}:
        return pl.scan_ipc(str(path), memory_map=False)
    raise ValueError(f"Unsupported file type for row-level diff: {path.suffix}")
