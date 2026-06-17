from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
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

    left = _load_frame(left_path)
    right = _load_frame(right_path)

    _validate_keys_exist(left, right, key_columns, left_path, right_path)
    _raise_on_duplicate_keys(left, key_columns, left_path)
    _raise_on_duplicate_keys(right, key_columns, right_path)

    payload_columns = sorted((set(left.columns) | set(right.columns)) - set(key_columns))

    left_aligned = _align_payload(left, key_columns, payload_columns)
    right_aligned = _align_payload(right, key_columns, payload_columns)

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

    updated_examples: list[RowChangeExample] = []
    updated_rows = 0
    unchanged_rows = 0
    for row in merged.iter_rows(named=True):
        deltas: list[RowFieldDelta] = []
        for column in payload_columns:
            left_value = row.get(f"left__{column}")
            right_value = row.get(f"right__{column}")
            if left_value != right_value:
                deltas.append(
                    RowFieldDelta(
                        column=column,
                        left_value=_normalize_value(left_value),
                        right_value=_normalize_value(right_value),
                    )
                )
        if deltas:
            updated_rows += 1
            if len(updated_examples) < max_examples:
                updated_examples.append(
                    RowChangeExample(
                        key={
                            column: _normalize_value(row.get(column))
                            for column in key_columns
                        },
                        deltas=deltas,
                        source_path=source_path,
                    )
                )
        else:
            unchanged_rows += 1

    summary = RowDiffSummary(
        key_columns=list(key_columns),
        added_rows=added_keys.height,
        removed_rows=removed_keys.height,
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
    left: pl.DataFrame,
    right: pl.DataFrame,
    key_columns: tuple[str, ...],
    left_path: Path,
    right_path: Path,
) -> None:
    left_missing = [column for column in key_columns if column not in left.columns]
    right_missing = [column for column in key_columns if column not in right.columns]
    if not left_missing and not right_missing:
        return
    raise ValueError(
        "Missing key columns for row-level diff. "
        f"left={left_path}: {left_missing or 'none'}, "
        f"right={right_path}: {right_missing or 'none'}"
    )


def _raise_on_duplicate_keys(
    frame: pl.DataFrame,
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
    frame: pl.DataFrame,
    key_columns: tuple[str, ...],
    payload_columns: list[str],
) -> pl.DataFrame:
    with_missing = frame.with_columns(
        [
            pl.lit(None).alias(column)
            for column in payload_columns
            if column not in frame.columns
        ]
    )
    return with_missing.select([*key_columns, *payload_columns])


def _key_samples(
    keys_frame: pl.DataFrame,
    key_columns: tuple[str, ...],
    max_examples: int,
) -> list[dict[str, Any]]:
    return [
        {
            column: _normalize_value(value)
            for column, value in zip(key_columns, row, strict=True)
        }
        for row in keys_frame.head(max_examples).iter_rows()
    ]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return value


def _load_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in {".feather", ".arrow"}:
        return pl.read_ipc(path, memory_map=False)
    raise ValueError(f"Unsupported file type for row-level diff: {path.suffix}")
