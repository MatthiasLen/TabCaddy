from __future__ import annotations

from pathlib import Path

import polars as pl

from tabcaddy.analysis.sources import SUPPORTED_FILE_SUFFIXES
from tabcaddy.merge.common import (
    PlannedOperation,
    PreparedOperation,
    ValidationResult,
    cast_lazyframe,
    is_binary,
    is_csv,
    scan_dataframe,
)
from tabcaddy.merge.conflict_detector import MergeConflictDetector


class MergeValidator:
    def __init__(
        self,
        conflict_detector: MergeConflictDetector | None = None,
    ) -> None:
        self._conflict_detector = conflict_detector or MergeConflictDetector()

    def prepare_operations(
        self,
        operations: list[PlannedOperation],
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> list[PreparedOperation]:
        if not operations:
            raise ValueError("No supported source files found to merge.")
        if out is not None and len(operations) > 1 and out.exists() and out.is_file():
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        destinations: set[Path] = set()
        prepared_operations: list[PreparedOperation] = []
        for operation in operations:
            self._validate_destination(operation.destination, destinations, inplace)
            prepared_operations.append(
                self._prepare_operation(operation, on_columns, ignore_filetype)
            )
        return prepared_operations

    def preview_operations(
        self,
        operations: list[PlannedOperation],
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> tuple[list[str], bool]:
        if not operations:
            raise ValueError("No supported source files found to merge.")
        if out is not None and len(operations) > 1 and out.exists() and out.is_file():
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        destinations: set[Path] = set()
        lines: list[str] = []
        has_issues = False
        for operation in operations:
            issue = self._destination_issue(
                destination=operation.destination,
                destinations=destinations,
                inplace=inplace,
            )
            if issue is None:
                issue = self._preview_operation_issue(
                    operation=operation,
                    on_columns=on_columns,
                    ignore_filetype=ignore_filetype,
                )
            if issue is not None:
                has_issues = True
            lines.append(self._preview_line(operation, issue))
        return lines, has_issues

    def _validate_destination(
        self,
        destination: Path,
        destinations: set[Path],
        inplace: bool,
    ) -> None:
        issue = self._destination_issue(destination, destinations, inplace)
        if issue is None:
            return
        if issue.startswith("Output path already exists"):
            raise FileExistsError(issue)
        raise ValueError(issue)

    def _prepare_operation(
        self,
        operation: PlannedOperation,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> PreparedOperation:
        if operation.target is None:
            return PreparedOperation(
                source=operation.source,
                target=None,
                destination=operation.destination,
            )

        validation = self._validate_merge_pair(
            source=operation.source,
            target=operation.target,
            on_columns=on_columns,
            ignore_filetype=ignore_filetype,
        )
        if validation.conflicting_columns:
            conflict_list = ", ".join(validation.conflicting_columns)
            raise ValueError(
                f"Schema mismatch between {operation.source} and {operation.target}: "
                f"incompatible types for {conflict_list}"
            )

        return PreparedOperation(
            source=operation.source,
            target=operation.target,
            destination=operation.destination,
            validation=validation,
        )

    def _destination_issue(
        self,
        destination: Path,
        destinations: set[Path],
        inplace: bool,
    ) -> str | None:
        if destination in destinations:
            return (
                f"Multiple source files resolve to the same destination: {destination}"
            )
        destinations.add(destination)

        if destination.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
            return f"Unsupported output file type: {destination.suffix}"
        if not inplace and destination.exists():
            return f"Output path already exists: {destination}"
        return None

    def _preview_operation_issue(
        self,
        operation: PlannedOperation,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> str | None:
        if operation.kind != "merge" or operation.target is None:
            return None

        try:
            validation = self._validate_merge_pair(
                source=operation.source,
                target=operation.target,
                on_columns=on_columns,
                ignore_filetype=ignore_filetype,
            )
        except ValueError as error:
            return str(error)

        if validation.conflicting_columns:
            conflict_list = ", ".join(validation.conflicting_columns)
            return (
                f"Schema mismatch between {operation.source} and {operation.target}: "
                f"incompatible types for {conflict_list}"
            )

        source_frame = scan_dataframe(operation.source)
        if validation.cast_source_to_target_schema:
            try:
                source_frame = self._cast_source_frame(
                    source_frame,
                    validation=validation,
                )
            except ValueError as error:
                return str(error)

        if not on_columns:
            return None

        conflict = self._conflict_detector.find_conflicting_key(
            frame=pl.concat(
                [scan_dataframe(operation.target), source_frame],
                how="vertical",
            ).unique(maintain_order=True),
            key_columns=on_columns,
        )
        if conflict is None:
            return None

        key_values = self._conflict_detector.format_conflicting_key(
            conflict,
            on_columns,
        )
        return (
            f"Conflicting duplicate key detected while merging {operation.source} "
            f"into {operation.target}: {key_values}"
        )

    def _cast_source_frame(
        self,
        source_frame: pl.LazyFrame,
        validation: ValidationResult,
    ) -> pl.LazyFrame:
        cast_frame = cast_lazyframe(source_frame, validation.target_schema)
        try:
            cast_frame.collect(engine="streaming")
        except pl.exceptions.PolarsError as error:
            raise ValueError(str(error)) from error
        return cast_frame

    def _preview_line(self, operation: PlannedOperation, issue: str | None) -> str:
        parts = [operation.kind.upper()]
        if operation.kind == "target_passthrough":
            parts.append(f"target={operation.source}")
        else:
            parts.append(f"source={operation.source}")
        if operation.target is not None:
            parts.append(f"target={operation.target}")
        parts.append(f"destination={operation.destination}")

        if (
            operation.target is not None
            and operation.source.suffix != operation.target.suffix
        ):
            parts.append(
                f"cast={operation.source.suffix.lower()}->{operation.target.suffix.lower()}"
            )
        if issue is not None:
            parts.append(f"issue={issue}")
        return " ".join(parts)

    def _validate_merge_pair(
        self,
        source: Path,
        target: Path,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> ValidationResult:
        source_schema = scan_dataframe(source).collect_schema()
        target_schema = scan_dataframe(target).collect_schema()

        if list(source_schema.keys()) != list(target_schema.keys()):
            raise ValueError(
                f"Schema mismatch between {source} and {target}: column layouts differ"
            )

        missing_keys = [column for column in on_columns if column not in target_schema]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Merge key columns not found in both files: {missing}")

        cast_source = (
            ignore_filetype
            and is_csv(source)
            and is_binary(target)
            and source.suffix.lower() != target.suffix.lower()
        )
        conflicting_columns = [
            column
            for column in target_schema
            if source_schema[column] != target_schema[column] and not cast_source
        ]
        return ValidationResult(
            target_schema=target_schema,
            cast_source_to_target_schema=cast_source,
            conflicting_columns=conflicting_columns,
        )
