from __future__ import annotations

from pathlib import Path

import polars as pl


class MergeConflictDetector:
    def raise_on_conflicting_keys(
        self,
        frame: pl.LazyFrame,
        key_columns: tuple[str, ...],
        source: Path,
        target: Path,
    ) -> None:
        schema_names = frame.collect_schema().names()
        payload_columns = [
            column for column in schema_names if column not in key_columns
        ]
        if not payload_columns:
            return

        conflict = (
            frame.group_by(key_columns)
            .agg(pl.struct(payload_columns).n_unique().alias("_payload_variants"))
            .filter(pl.col("_payload_variants") > 1)
            .limit(1)
            .collect(engine="streaming")
        )
        if conflict.is_empty():
            return

        values = conflict.row(0, named=True)
        key_values = ", ".join(
            f"{column}={values[column]!r}" for column in key_columns
        )
        raise ValueError(
            f"Conflicting duplicate key detected while merging {source} into {target}: {key_values}"
        )