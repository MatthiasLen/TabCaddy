from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from tabcaddy.infrastructure.source_resolver import SUPPORTED_FILE_SUFFIXES


_BINARY_FILE_SUFFIXES = {".parquet", ".feather", ".arrow"}
_CSV_SUFFIX = ".csv"
_FEATHER_ARROW_SUFFIXES = {".feather", ".arrow"}


@dataclass(frozen=True)
class _PlannedOperation:
    source: Path
    target: Path | None
    destination: Path


class MergeDatasets:
    def run(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> list[Path]:
        source_path = self._resolve_existing_path(source, role="source")
        target_path = self._resolve_existing_path(target, role="target")
        output_path = out.expanduser().resolve() if out is not None else None

        # Exactly one of --out or --inplace must be specified
        if inplace == (output_path is not None):
            raise ValueError("Provide exactly one of --out or --inplace.")

        operations = self._plan_operations(
            source=source_path,
            target=target_path,
            out=output_path,
            inplace=inplace,
            ignore_filetype=ignore_filetype,
        )
        self._validate_operations(
            operations=operations,
            out=output_path,
            inplace=inplace,
            on_columns=on_columns,
            ignore_filetype=ignore_filetype,
        )

        written: list[Path] = []
        for operation in operations:
            if operation.target is None:
                self._execute_copy(operation, inplace=inplace)
            else:
                self._execute_merge(
                    operation=operation,
                    inplace=inplace,
                    on_columns=on_columns,
                    ignore_filetype=ignore_filetype,
                )
            written.append(operation.destination)
        return written

    def _plan_operations(
        self,
        source: Path,
        target: Path,
        out: Path | None,
        inplace: bool,
        ignore_filetype: bool,
    ) -> list[_PlannedOperation]:
        # File-to-file merge
        if source.is_file() and target.is_file():
            if not self._supports_merge_pair(source, target, ignore_filetype):
                raise ValueError(
                    "Source and target file types must match unless --ignore-filetype is provided."
                )
            if out is None or out.is_dir():
                raise ValueError(
                    "File-to-file merge requires --out to point to a file."
                )
            return [_PlannedOperation(source=source, target=target, destination=out)]

        # Mixed file-to-folder scenarios
        if source.is_dir() and target.is_file():
            raise ValueError("Folder-to-file merge is not supported.")

        if source.is_file() and target.is_dir():
            destination = self._single_file_destination(
                source=source,
                target_folder=target,
                out=out,
                inplace=inplace,
            )
            matched = self._find_directory_match(source, target, ignore_filetype)
            return [
                _PlannedOperation(
                    source=source,
                    target=matched,
                    destination=matched
                    if inplace and matched is not None
                    else destination,
                )
            ]

        # Folder-to-folder merge
        if out is not None and out.exists() and not out.is_dir():
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        source_files = self._iter_supported_files(source)
        target_index = self._build_directory_index(target, ignore_filetype)
        operations: list[_PlannedOperation] = []

        for file_path in source_files:
            matched = self._match_indexed_file(
                source=file_path,
                target_index=target_index,
                ignore_filetype=ignore_filetype,
            )

            if inplace:
                destination = (
                    matched
                    if matched is not None
                    else target / file_path.relative_to(source)
                )
            else:
                assert out is not None
                destination = (
                    out / matched.relative_to(target)
                    if matched is not None
                    else out / file_path.relative_to(source)
                )

            operations.append(
                _PlannedOperation(
                    source=file_path,
                    target=matched,
                    destination=destination,
                )
            )
        return operations

    def _validate_operations(
        self,
        operations: list[_PlannedOperation],
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> None:
        if not operations:
            raise ValueError("No supported source files found to merge.")

        if out is not None and len(operations) > 1 and out.suffix:
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        destinations: set[Path] = set()
        for operation in operations:
            if operation.destination in destinations:
                raise ValueError(
                    f"Multiple source files resolve to the same destination: {operation.destination}"
                )
            destinations.add(operation.destination)

            if operation.destination.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
                raise ValueError(
                    f"Unsupported output file type: {operation.destination.suffix}"
                )

            if not inplace and operation.destination.exists():
                raise FileExistsError(
                    f"Output path already exists: {operation.destination}"
                )

            if operation.target is None:
                continue

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

    def _execute_copy(self, operation: _PlannedOperation, inplace: bool) -> None:
        lazy_frame = self._scan_dataframe(operation.source)
        self._write_lazyframe(
            lazy_frame=lazy_frame,
            destination=operation.destination,
            inplace=inplace,
            format_hint=operation.destination,
        )

    def _execute_merge(
        self,
        operation: _PlannedOperation,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> None:
        assert operation.target is not None, "_execute_merge requires target to be set"

        validation = self._validate_merge_pair(
            source=operation.source,
            target=operation.target,
            on_columns=on_columns,
            ignore_filetype=ignore_filetype,
        )

        source_frame = self._scan_dataframe(operation.source)
        if validation.cast_source_to_target_schema:
            source_frame = self._cast_lazyframe(source_frame, validation.target_schema)

        target_frame = self._scan_dataframe(operation.target)

        # Concatenate frames and remove exact duplicates while preserving order
        merged = pl.concat([target_frame, source_frame], how="vertical")
        deduplicated = merged.unique(maintain_order=True)

        if on_columns:
            self._raise_on_conflicting_keys(
                frame=deduplicated,
                key_columns=on_columns,
                source=operation.source,
                target=operation.target,
            )

        self._write_lazyframe(
            lazy_frame=deduplicated,
            destination=operation.destination,
            inplace=inplace,
            format_hint=operation.destination,
        )

    def _raise_on_conflicting_keys(
        self,
        frame: pl.LazyFrame,
        key_columns: tuple[str, ...],
        source: Path,
        target: Path,
    ) -> None:
        schema_names = frame.collect_schema().names()
        payload_columns = [col for col in schema_names if col not in key_columns]

        if not payload_columns:
            return

        # Check if any key has multiple distinct payloads (indicating a conflict)
        conflict = (
            frame.select(
                [
                    *(pl.col(col) for col in key_columns),
                    pl.struct(payload_columns).alias("_payload"),
                ]
            )
            .group_by(key_columns)
            .agg(pl.col("_payload").n_unique().alias("_variants"))
            .filter(pl.col("_variants") > 1)
            .limit(1)
            .collect()
        )

        if conflict.is_empty():
            return

        values = conflict.row(0, named=True)
        key_values = ", ".join(f"{col}={values[col]!r}" for col in key_columns)
        raise ValueError(
            f"Conflicting duplicate key detected while merging {source} into {target}: {key_values}"
        )

    def _validate_merge_pair(
        self,
        source: Path,
        target: Path,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> _ValidationResult:
        source_schema = self._scan_dataframe(source).collect_schema()
        target_schema = self._scan_dataframe(target).collect_schema()

        # Verify column layout matches
        if list(source_schema.keys()) != list(target_schema.keys()):
            raise ValueError(
                f"Schema mismatch between {source} and {target}: column layouts differ"
            )

        # Verify merge key columns exist
        missing_keys = [col for col in on_columns if col not in target_schema]
        if missing_keys:
            missing = ", ".join(missing_keys)
            raise ValueError(f"Merge key columns not found in both files: {missing}")

        # Check if source needs casting to match target
        cast_source = (
            ignore_filetype
            and self._is_csv(source)
            and self._is_binary(target)
            and source.suffix.lower() != target.suffix.lower()
        )

        # Identify type mismatches that won't be resolved by casting
        conflicting_columns = [
            col
            for col in target_schema
            if source_schema[col] != target_schema[col] and not cast_source
        ]

        return _ValidationResult(
            target_schema=target_schema,
            cast_source_to_target_schema=cast_source,
            conflicting_columns=conflicting_columns,
        )

    def _single_file_destination(
        self,
        source: Path,
        target_folder: Path,
        out: Path | None,
        inplace: bool,
    ) -> Path:
        if inplace:
            return target_folder / source.name
        if out is None:
            raise ValueError("Provide --out unless --inplace is selected.")
        if out.exists() and out.is_dir():
            return out / source.name
        if not out.exists() and not out.suffix:
            return out / source.name
        return out

    def _build_directory_index(
        self, target_folder: Path, ignore_filetype: bool
    ) -> dict[tuple[str, str | None], Path]:
        duplicates: defaultdict[tuple[str, str | None], list[Path]] = defaultdict(list)
        for path in self._iter_supported_files(target_folder, allow_empty=True):
            key = self._match_key(path, ignore_filetype)
            duplicates[key].append(path)

        # Check for ambiguous matches (multiple files with same key)
        for paths in duplicates.values():
            if len(paths) > 1:
                sample = ", ".join(str(p) for p in paths)
                raise ValueError(f"Ambiguous target match detected: {sample}")

        return {key: paths[0] for key, paths in duplicates.items()}

    def _find_directory_match(
        self, source: Path, target_folder: Path, ignore_filetype: bool
    ) -> Path | None:
        return self._match_indexed_file(
            source=source,
            target_index=self._build_directory_index(target_folder, ignore_filetype),
            ignore_filetype=ignore_filetype,
        )

    def _match_indexed_file(
        self,
        source: Path,
        target_index: dict[tuple[str, str | None], Path],
        ignore_filetype: bool,
    ) -> Path | None:
        return target_index.get(self._match_key(source, ignore_filetype))

    def _match_key(self, path: Path, ignore_filetype: bool) -> tuple[str, str | None]:
        if ignore_filetype:
            return (path.stem, None)
        return (path.stem, path.suffix.lower())

    def _supports_merge_pair(
        self, source: Path, target: Path, ignore_filetype: bool
    ) -> bool:
        return ignore_filetype or source.suffix.lower() == target.suffix.lower()

    def _write_lazyframe(
        self,
        lazy_frame: pl.LazyFrame,
        destination: Path,
        inplace: bool,
        format_hint: Path,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        target_path = destination if not inplace else self._temp_path(destination)

        if target_path.exists():
            target_path.unlink()

        try:
            self._sink_lazyframe(lazy_frame, target_path, format_hint)
            if inplace:
                os.replace(target_path, destination)
        except pl.exceptions.PolarsError as error:
            self._cleanup_temp_file(target_path)
            raise ValueError(str(error)) from error
        except Exception:
            self._cleanup_temp_file(target_path)
            raise

    def _cleanup_temp_file(self, path: Path) -> None:
        """Remove temp file if it exists, silently ignore if already gone."""
        if path.exists():
            path.unlink()

    def _sink_lazyframe(
        self, lazy_frame: pl.LazyFrame, destination: Path, format_hint: Path
    ) -> None:
        suffix = format_hint.suffix.lower()
        if suffix == _CSV_SUFFIX:
            lazy_frame.sink_csv(str(destination))
        elif suffix == ".parquet":
            lazy_frame.sink_parquet(str(destination))
        elif suffix in _FEATHER_ARROW_SUFFIXES:
            lazy_frame.sink_ipc(str(destination))
        else:
            raise ValueError(f"Unsupported file type: {format_hint.suffix}")

    def _cast_lazyframe(self, frame: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
        """Cast frame columns to match target schema types."""
        return frame.with_columns(
            pl.col(col).cast(dtype, strict=True) for col, dtype in schema.items()
        )

    def _scan_dataframe(self, path: Path) -> pl.LazyFrame:
        suffix = path.suffix.lower()
        if suffix == _CSV_SUFFIX:
            return pl.scan_csv(
                str(path), infer_schema_length=1000, try_parse_dates=True
            )
        if suffix == ".parquet":
            return pl.scan_parquet(str(path))
        if suffix in _FEATHER_ARROW_SUFFIXES:
            return pl.scan_ipc(str(path), memory_map=False)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def _iter_supported_files(
        self, folder: Path, allow_empty: bool = False
    ) -> list[Path]:
        """Recursively find all supported data files in a folder."""
        files = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_SUFFIXES
        )
        if not files and not allow_empty:
            raise ValueError(f"No supported files found under: {folder}")
        return files

    def _resolve_existing_path(self, path: Path, role: str) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"{role.capitalize()} does not exist: {resolved}")
        if (
            resolved.is_file()
            and resolved.suffix.lower() not in SUPPORTED_FILE_SUFFIXES
        ):
            raise ValueError(f"Unsupported file type: {resolved.suffix}")
        return resolved

    def _is_binary(self, path: Path) -> bool:
        return path.suffix.lower() in _BINARY_FILE_SUFFIXES

    def _is_csv(self, path: Path) -> bool:
        return path.suffix.lower() == ".csv"

    def _temp_path(self, destination: Path) -> Path:
        return destination.with_name(f".{destination.name}.tmp")


@dataclass(frozen=True)
class _ValidationResult:
    target_schema: pl.Schema
    cast_source_to_target_schema: bool
    conflicting_columns: list[str]
