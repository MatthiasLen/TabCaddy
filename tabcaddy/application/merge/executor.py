from __future__ import annotations

import polars as pl

from tabcaddy.application.merge.common import (
    PreparedOperation,
    cast_lazyframe,
    scan_dataframe,
    write_lazyframe,
)
from tabcaddy.application.merge.conflict_detector import MergeConflictDetector


class MergeExecutor:
    def __init__(
        self,
        conflict_detector: MergeConflictDetector | None = None,
    ) -> None:
        self._conflict_detector = conflict_detector or MergeConflictDetector()

    def execute(
        self,
        operation: PreparedOperation,
        on_columns: tuple[str, ...],
        inplace: bool,
    ) -> None:
        if operation.target is None:
            self._execute_copy(operation, inplace)
            return
        self._execute_merge(operation, on_columns)

    def _execute_copy(self, operation: PreparedOperation, inplace: bool) -> None:
        write_lazyframe(
            lazy_frame=scan_dataframe(operation.source),
            destination=operation.destination,
            inplace=inplace,
            format_hint=operation.destination,
        )

    def _execute_merge(
        self,
        operation: PreparedOperation,
        on_columns: tuple[str, ...],
    ) -> None:
        if operation.target is None:
            raise ValueError("_execute_merge requires target to be set")
        if operation.validation is None:
            raise ValueError("_execute_merge requires validation")

        source_frame = scan_dataframe(operation.source)
        if operation.validation.cast_source_to_target_schema:
            source_frame = cast_lazyframe(
                source_frame,
                operation.validation.target_schema,
            )

        deduplicated = pl.concat(
            [scan_dataframe(operation.target), source_frame],
            how="vertical",
        ).unique(maintain_order=True)

        if on_columns:
            self._conflict_detector.raise_on_conflicting_keys(
                frame=deduplicated,
                key_columns=on_columns,
                source=operation.source,
                target=operation.target,
            )

        write_lazyframe(
            lazy_frame=deduplicated,
            destination=operation.destination,
            inplace=operation.destination == operation.target,
            format_hint=operation.destination,
        )
