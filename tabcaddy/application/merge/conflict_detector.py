from __future__ import annotations

from pathlib import Path

import polars as pl


class MergeConflictDetector:
    def find_conflicting_key(
        self,
        frame: pl.LazyFrame,
        key_columns: tuple[str, ...],
    ) -> dict[str, object] | None:
        schema_names = frame.collect_schema().names()
        payload_columns = [
            column for column in schema_names if column not in key_columns
        ]
        if not payload_columns:
            return None

        conflict = (
            frame.group_by(key_columns)
            .agg(pl.struct(payload_columns).n_unique().alias("_payload_variants"))
            .filter(pl.col("_payload_variants") > 1)
            .limit(1)
            .collect(engine="streaming")
        )
        if conflict.is_empty():
            return None

        return conflict.row(0, named=True)

    def format_conflicting_key(
        self,
        values: dict[str, object],
        key_columns: tuple[str, ...],
    ) -> str:
        return ", ".join(f"{column}={values[column]!r}" for column in key_columns)

    def raise_on_conflicting_keys(
        self,
        frame: pl.LazyFrame,
        key_columns: tuple[str, ...],
        source: Path,
        target: Path,
    ) -> None:
        values = self.find_conflicting_key(frame, key_columns)
        if values is None:
            return

        key_values = self.format_conflicting_key(values, key_columns)
        raise ValueError(
            f"Conflicting duplicate key detected while merging {source} into {target}: {key_values}"
        )
