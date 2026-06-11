from __future__ import annotations

from pathlib import Path

from tabcaddy.application.merge.common import (
    PlannedOperation,
    PreparedOperation,
    ValidationResult,
    is_binary,
    is_csv,
    scan_dataframe,
)
from tabcaddy.infrastructure.source_resolver import SUPPORTED_FILE_SUFFIXES


class MergeValidator:
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

    def _validate_destination(
        self,
        destination: Path,
        destinations: set[Path],
        inplace: bool,
    ) -> None:
        if destination in destinations:
            raise ValueError(
                f"Multiple source files resolve to the same destination: {destination}"
            )
        destinations.add(destination)

        if destination.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
            raise ValueError(f"Unsupported output file type: {destination.suffix}")
        if not inplace and destination.exists():
            raise FileExistsError(f"Output path already exists: {destination}")

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
