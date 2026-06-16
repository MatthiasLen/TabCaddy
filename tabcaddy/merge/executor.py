from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tabcaddy.merge.common import (
    MergeStrategy,
    PreparedOperation,
    align_lazyframe_to_schema,
    cast_lazyframe,
    scan_dataframe,
    stage_lazyframe,
    temp_path,
)
from tabcaddy.merge.conflict_detector import MergeConflictDetector


@dataclass(frozen=True)
class StagedWrite:
    destination: Path
    staged_path: Path


class MergeExecutor:
    def __init__(
        self,
        conflict_detector: MergeConflictDetector | None = None,
    ) -> None:
        self._conflict_detector = conflict_detector or MergeConflictDetector()

    def execute_all(
        self,
        operations: list[PreparedOperation],
        on_columns: tuple[str, ...],
        strategy: MergeStrategy,
        transaction_root: Path | None,
    ) -> list[Path]:
        staged_writes: list[StagedWrite] = []
        temp_root = self._prepare_temp_root(transaction_root)

        try:
            for operation in operations:
                staged_writes.append(
                    self._stage_operation(
                        operation,
                        on_columns=on_columns,
                        strategy=strategy,
                        transaction_root=transaction_root,
                        temp_root=temp_root,
                    )
                )

            for staged_write in staged_writes:
                staged_write.destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_write.staged_path, staged_write.destination)
        finally:
            if temp_root is not None:
                shutil.rmtree(temp_root, ignore_errors=True)

        return [operation.destination for operation in operations]

    def _prepare_temp_root(self, transaction_root: Path | None) -> Path | None:
        if transaction_root is None:
            return None

        temp_root = transaction_root.with_name(f".{transaction_root.name}.tmp")
        shutil.rmtree(temp_root, ignore_errors=True)
        return temp_root

    def _stage_operation(
        self,
        operation: PreparedOperation,
        on_columns: tuple[str, ...],
        strategy: MergeStrategy,
        transaction_root: Path | None,
        temp_root: Path | None,
    ) -> StagedWrite:
        staged_path = self._staged_path(
            destination=operation.destination,
            transaction_root=transaction_root,
            temp_root=temp_root,
        )

        if operation.target is None:
            stage_lazyframe(
                lazy_frame=scan_dataframe(operation.source),
                destination=staged_path,
                format_hint=operation.destination,
            )
            return StagedWrite(
                destination=operation.destination, staged_path=staged_path
            )

        if operation.validation is None:
            raise ValueError("_stage_operation requires validation")

        source_frame = scan_dataframe(operation.source)
        if operation.validation.cast_source_to_target_schema:
            source_frame = cast_lazyframe(
                source_frame,
                operation.validation.target_schema,
            )

        target_frame = scan_dataframe(operation.target)
        target_frame = align_lazyframe_to_schema(
            target_frame,
            operation.validation.effective_schema,
        )
        source_frame = align_lazyframe_to_schema(
            source_frame,
            operation.validation.effective_schema,
        )

        if strategy == "append":
            merged = self._merge_append(target_frame, source_frame)
        else:
            merged = self._merge_upsert(target_frame, source_frame, on_columns)

        if strategy == "append" and on_columns:
            self._conflict_detector.raise_on_conflicting_keys(
                frame=merged,
                key_columns=on_columns,
                source=operation.source,
                target=operation.target,
            )

        stage_lazyframe(
            lazy_frame=merged,
            destination=staged_path,
            format_hint=operation.destination,
        )
        return StagedWrite(destination=operation.destination, staged_path=staged_path)

    def _merge_append(
        self,
        target_frame: pl.LazyFrame,
        source_frame: pl.LazyFrame,
    ) -> pl.LazyFrame:
        schema_names = target_frame.collect_schema().names()
        if not schema_names:
            return pl.concat([target_frame, source_frame], how="vertical")

        source_additions = source_frame.join(
            target_frame, on=schema_names, how="anti", nulls_equal=True
        ).unique(subset=schema_names, maintain_order=True)
        return pl.concat([target_frame, source_additions], how="vertical")

    def _merge_upsert(
        self,
        target_frame: pl.LazyFrame,
        source_frame: pl.LazyFrame,
        on_columns: tuple[str, ...],
    ) -> pl.LazyFrame:
        if not on_columns:
            raise ValueError("--strategy upsert requires at least one --on column.")

        key_columns = list(on_columns)
        source_deduplicated = source_frame.unique(
            subset=key_columns,
            keep="last",
            maintain_order=True,
        )
        source_keys = source_deduplicated.select(key_columns).unique(
            maintain_order=True
        )
        target_retained = target_frame.join(source_keys, on=key_columns, how="anti")
        return pl.concat([target_retained, source_deduplicated], how="vertical")

    def _staged_path(
        self,
        destination: Path,
        transaction_root: Path | None,
        temp_root: Path | None,
    ) -> Path:
        if transaction_root is None or temp_root is None:
            return temp_path(destination)
        return temp_root / destination.relative_to(transaction_root)
