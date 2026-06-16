from __future__ import annotations

from pathlib import Path

import polars as pl

from tabcaddy.analysis.sources import SUPPORTED_FILE_SUFFIXES
from tabcaddy.merge.common import (
    SchemaEvolution,
    MergeStrategy,
    PlannedOperation,
    PreparedOperation,
    ValidationResult,
    align_lazyframe_to_schema,
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
        strategy: MergeStrategy,
        ignore_filetype: bool,
        schema_evolution: SchemaEvolution = "strict",
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
                self._prepare_operation(
                    operation,
                    on_columns,
                    strategy,
                    ignore_filetype,
                    schema_evolution,
                )
            )
        return prepared_operations

    def preview_operations(
        self,
        operations: list[PlannedOperation],
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        strategy: MergeStrategy,
        ignore_filetype: bool,
        schema_evolution: SchemaEvolution = "strict",
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
            validation: ValidationResult | None = None
            issue = self._destination_issue(
                destination=operation.destination,
                destinations=destinations,
                inplace=inplace,
            )
            if issue is None:
                issue, validation = self._preview_operation_issue(
                    operation=operation,
                    on_columns=on_columns,
                    strategy=strategy,
                    ignore_filetype=ignore_filetype,
                    schema_evolution=schema_evolution,
                )
            if issue is not None:
                has_issues = True
            lines.append(
                self._preview_line(
                    operation,
                    issue,
                    strategy,
                    on_columns,
                    validation,
                    schema_evolution,
                )
            )
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
        strategy: MergeStrategy,
        ignore_filetype: bool,
        schema_evolution: SchemaEvolution,
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
            strategy=strategy,
            ignore_filetype=ignore_filetype,
            schema_evolution=schema_evolution,
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
        strategy: MergeStrategy,
        ignore_filetype: bool,
        schema_evolution: SchemaEvolution,
    ) -> tuple[str | None, ValidationResult | None]:
        if operation.kind != "merge" or operation.target is None:
            return None, None

        try:
            validation = self._validate_merge_pair(
                source=operation.source,
                target=operation.target,
                on_columns=on_columns,
                strategy=strategy,
                ignore_filetype=ignore_filetype,
                schema_evolution=schema_evolution,
            )
        except ValueError as error:
            return str(error), None

        if validation.conflicting_columns:
            conflict_list = ", ".join(validation.conflicting_columns)
            return (
                f"Schema mismatch between {operation.source} and {operation.target}: "
                f"incompatible types for {conflict_list}"
            ), validation

        try:
            target_frame, source_frame = self._aligned_merge_frames(
                source=operation.source,
                target=operation.target,
                validation=validation,
            )
        except ValueError as error:
            return str(error), validation

        if not on_columns or strategy != "append":
            return None, validation

        conflict = self._conflict_detector.find_conflicting_key(
            frame=self._build_append_preview_frame(
                target_frame=target_frame,
                source_frame=source_frame,
            ),
            key_columns=on_columns,
        )
        if conflict is None:
            return None, validation

        key_values = self._conflict_detector.format_conflicting_key(
            conflict,
            on_columns,
        )
        return (
            f"Conflicting duplicate key detected while merging {operation.source} "
            f"into {operation.target}: {key_values}"
        ), validation

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

    def _preview_line(
        self,
        operation: PlannedOperation,
        issue: str | None,
        strategy: MergeStrategy,
        on_columns: tuple[str, ...],
        validation: ValidationResult | None,
        schema_evolution: SchemaEvolution,
    ) -> str:
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
        if operation.kind == "merge":
            parts.append(f"strategy={strategy}")
            parts.append(f"schema_evolution={schema_evolution}")
            if schema_evolution == "allow-additive" and validation is not None:
                parts.append(f"added_columns={len(validation.schema_added_columns)}")
            summary = self._preview_merge_summary(
                operation,
                strategy,
                on_columns,
                validation,
            )
            if summary is not None:
                parts.append(summary)
        if issue is not None:
            parts.append(f"issue={issue}")
        return " ".join(parts)

    def _preview_merge_summary(
        self,
        operation: PlannedOperation,
        strategy: MergeStrategy,
        on_columns: tuple[str, ...],
        validation: ValidationResult | None,
    ) -> str | None:
        if operation.target is None or validation is None:
            return None

        try:
            target_frame, source_frame = self._aligned_merge_frames(
                source=operation.source,
                target=operation.target,
                validation=validation,
            )
            schema_names = validation.effective_schema.names()

            if strategy == "append":
                if not schema_names:
                    return None
                target_unique = target_frame.unique(
                    subset=schema_names,
                    maintain_order=True,
                )
                inserts = (
                    source_frame.join(target_unique, on=schema_names, how="anti")
                    .select(pl.len().alias("rows"))
                    .collect(engine="streaming")
                    .item()
                )
                return f"insert_rows={inserts}"

            if not on_columns:
                return None
            key_columns = list(on_columns)
            upsert_keys = source_frame.select(key_columns).unique(maintain_order=True)
            replaced = (
                target_frame.join(upsert_keys, on=key_columns, how="semi")
                .select(pl.len().alias("rows"))
                .collect(engine="streaming")
                .item()
            )
            inserted = (
                upsert_keys.join(
                    target_frame.select(key_columns).unique(maintain_order=True),
                    on=key_columns,
                    how="anti",
                )
                .select(pl.len().alias("rows"))
                .collect(engine="streaming")
                .item()
            )
            return f"insert_rows={inserted} replace_rows={replaced}"
        except pl.exceptions.PolarsError:
            return None

    def _build_append_preview_frame(
        self,
        target_frame: pl.LazyFrame,
        source_frame: pl.LazyFrame,
    ) -> pl.LazyFrame:
        columns = target_frame.collect_schema().names()
        if not columns:
            return pl.concat([target_frame, source_frame], how="vertical")

        source_additions = source_frame.join(
            target_frame, on=columns, how="anti", nulls_equal=True
        ).unique(subset=columns, maintain_order=True)
        return pl.concat([target_frame, source_additions], how="vertical")

    def _validate_merge_pair(
        self,
        source: Path,
        target: Path,
        on_columns: tuple[str, ...],
        strategy: MergeStrategy,
        ignore_filetype: bool,
        schema_evolution: SchemaEvolution,
    ) -> ValidationResult:
        source_schema = scan_dataframe(source).collect_schema()
        target_schema = scan_dataframe(target).collect_schema()

        if schema_evolution == "strict" and list(source_schema.keys()) != list(
            target_schema.keys()
        ):
            raise ValueError(
                f"Schema mismatch between {source} and {target}: column layouts differ"
            )

        missing_keys = [
            column
            for column in on_columns
            if column not in target_schema or column not in source_schema
        ]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Merge key columns not found in both files: {missing}")
        if strategy == "upsert" and not on_columns:
            raise ValueError("--strategy upsert requires at least one --on column.")

        cast_source = (
            ignore_filetype
            and is_csv(source)
            and is_binary(target)
            and source.suffix.lower() != target.suffix.lower()
        )
        conflicting_columns = [
            column
            for column in target_schema
            if column in source_schema
            and source_schema[column] != target_schema[column]
            and not cast_source
        ]

        effective_schema_items = list(target_schema.items())
        schema_added_columns = tuple(
            column for column in source_schema.keys() if column not in target_schema
        )
        for column in schema_added_columns:
            effective_schema_items.append((column, source_schema[column]))

        return ValidationResult(
            target_schema=target_schema,
            effective_schema=pl.Schema(effective_schema_items),
            schema_added_columns=schema_added_columns,
            cast_source_to_target_schema=cast_source,
            conflicting_columns=conflicting_columns,
        )

    def _aligned_merge_frames(
        self,
        source: Path,
        target: Path,
        validation: ValidationResult,
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        target_frame = align_lazyframe_to_schema(
            scan_dataframe(target),
            validation.effective_schema,
        )
        source_frame = scan_dataframe(source)
        if validation.cast_source_to_target_schema:
            source_frame = self._cast_source_frame(
                source_frame,
                validation=validation,
            )
        source_frame = align_lazyframe_to_schema(
            source_frame,
            validation.effective_schema,
        )
        return target_frame, source_frame
