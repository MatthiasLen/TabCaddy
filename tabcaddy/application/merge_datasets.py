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


@dataclass(frozen=True)
class _MatchKey:
    relative_parent: str | None
    stem: str
    suffix: str | None


@dataclass(frozen=True)
class _PreparedOperation:
    source: Path
    target: Path | None
    destination: Path
    validation: _ValidationResult | None = None


@dataclass(frozen=True)
class _ValidationResult:
    target_schema: pl.Schema
    cast_source_to_target_schema: bool
    conflicting_columns: list[str]


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
        prepared_operations = self._prepare_operations(
            operations=operations,
            out=output_path,
            inplace=inplace,
            on_columns=on_columns,
            ignore_filetype=ignore_filetype,
        )

        written: list[Path] = []
        for operation in prepared_operations:
            if operation.target is None:
                self._execute_copy(operation, inplace=inplace)
            else:
                self._execute_merge(
                    operation=operation,
                    on_columns=on_columns,
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

        if source.is_dir() and target.is_file():
            raise ValueError("Folder-to-file merge is not supported.")

        if source.is_file() and target.is_dir():
            target_index = self._build_directory_index(target, ignore_filetype)
            matched = self._match_indexed_file(
                source=source,
                target_index=target_index,
                ignore_filetype=ignore_filetype,
            )
            return [
                _PlannedOperation(
                    source=source,
                    target=matched,
                    destination=self._resolve_destination(
                        source=source,
                        source_root=source.parent,
                        target_root=target,
                        matched_target=matched,
                        out=out,
                        inplace=inplace,
                    ),
                )
            ]

        if out is not None and out.suffix:
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        if out is not None and out.exists() and not out.is_dir():
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        source_files = self._iter_supported_files(source)
        target_index = self._build_directory_index(
            target,
            ignore_filetype,
            relative_to=target,
        )
        operations: list[_PlannedOperation] = []
        matched_targets: set[Path] = set()
        for file_path in source_files:
            matched = self._match_indexed_file(
                source=file_path,
                target_index=target_index,
                ignore_filetype=ignore_filetype,
                relative_to=source,
            )
            if matched is not None:
                matched_targets.add(matched)

            operations.append(
                _PlannedOperation(
                    source=file_path,
                    target=matched,
                    destination=self._resolve_destination(
                        source=file_path,
                        source_root=source,
                        target_root=target,
                        matched_target=matched,
                        out=out,
                        inplace=inplace,
                    ),
                )
            )

        if not inplace:
            assert out is not None
            for target_file in sorted(set(target_index.values()) - matched_targets):
                operations.append(
                    _PlannedOperation(
                        source=target_file,
                        target=None,
                        destination=out / target_file.relative_to(target),
                    )
                )
        return operations

    def _resolve_destination(
        self,
        source: Path,
        source_root: Path,
        target_root: Path,
        matched_target: Path | None,
        out: Path | None,
        inplace: bool,
    ) -> Path:
        if matched_target is not None:
            if inplace:
                return matched_target
            if out is None:
                raise ValueError("Provide --out unless --inplace is selected.")
            if source_root == source.parent and out.suffix:
                return out
            return out / matched_target.relative_to(target_root)

        relative_path = source.relative_to(source_root)
        if inplace:
            return target_root / relative_path

        if out is None:
            raise ValueError("Provide --out unless --inplace is selected.")

        if out.exists() and out.is_dir():
            return out / relative_path
        if not out.exists() and not out.suffix:
            return out / relative_path
        if relative_path.parent == Path("."):
            return out
        raise ValueError(
            "Folder-to-folder merge requires --out to point to a directory."
        )

    def _prepare_operations(
        self,
        operations: list[_PlannedOperation],
        out: Path | None,
        inplace: bool,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> list[_PreparedOperation]:
        if not operations:
            raise ValueError("No supported source files found to merge.")

        if out is not None and len(operations) > 1 and out.suffix:
            raise ValueError(
                "Folder-to-folder merge requires --out to point to a directory."
            )

        destinations: set[Path] = set()
        prepared_operations: list[_PreparedOperation] = []
        for operation in operations:
            self._validate_destination(operation.destination, destinations, inplace)
            prepared_operations.append(
                self._prepare_operation(
                    operation=operation,
                    on_columns=on_columns,
                    ignore_filetype=ignore_filetype,
                )
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
        operation: _PlannedOperation,
        on_columns: tuple[str, ...],
        ignore_filetype: bool,
    ) -> _PreparedOperation:
        if operation.target is None:
            return _PreparedOperation(
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

        return _PreparedOperation(
            source=operation.source,
            target=operation.target,
            destination=operation.destination,
            validation=validation,
        )

    def _execute_copy(self, operation: _PreparedOperation, inplace: bool) -> None:
        lazy_frame = self._scan_dataframe(operation.source)
        self._write_lazyframe(
            lazy_frame=lazy_frame,
            destination=operation.destination,
            inplace=inplace,
            format_hint=operation.destination,
        )

    def _execute_merge(
        self,
        operation: _PreparedOperation,
        on_columns: tuple[str, ...],
    ) -> None:
        assert operation.target is not None, "_execute_merge requires target to be set"
        assert operation.validation is not None, "_execute_merge requires validation"

        source_frame = self._scan_dataframe(operation.source)
        if operation.validation.cast_source_to_target_schema:
            source_frame = self._cast_lazyframe(
                source_frame,
                operation.validation.target_schema,
            )

        target_frame = self._scan_dataframe(operation.target)

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
            inplace=operation.destination == operation.target,
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

        payload_struct_expr = pl.struct(payload_columns)
        conflict = (
            frame.group_by(key_columns)
            .agg(payload_struct_expr.n_unique().alias("_payload_variants"))
            .filter(pl.col("_payload_variants") > 1)
            .limit(1)
            .collect(engine="streaming")
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

    def _build_directory_index(
        self,
        target_folder: Path,
        ignore_filetype: bool,
        relative_to: Path | None = None,
    ) -> dict[_MatchKey, Path]:
        duplicates: defaultdict[_MatchKey, list[Path]] = defaultdict(list)
        for path in self._iter_supported_files(target_folder, allow_empty=True):
            key = self._match_key(path, ignore_filetype, relative_to=relative_to)
            duplicates[key].append(path)

        # Check for ambiguous matches (multiple files with same key)
        for paths in duplicates.values():
            if len(paths) > 1:
                sample = ", ".join(str(p) for p in paths)
                raise ValueError(f"Ambiguous target match detected: {sample}")

        return {key: paths[0] for key, paths in duplicates.items()}

    def _match_indexed_file(
        self,
        source: Path,
        target_index: dict[_MatchKey, Path],
        ignore_filetype: bool,
        relative_to: Path | None = None,
    ) -> Path | None:
        return target_index.get(
            self._match_key(source, ignore_filetype, relative_to=relative_to)
        )

    def _match_key(
        self,
        path: Path,
        ignore_filetype: bool,
        relative_to: Path | None = None,
    ) -> _MatchKey:
        relative_parent: str | None = None
        if relative_to is not None:
            relative_parent = path.relative_to(relative_to).parent.as_posix()
        if ignore_filetype:
            return _MatchKey(
                relative_parent=relative_parent, stem=path.stem, suffix=None
            )
        return _MatchKey(
            relative_parent=relative_parent,
            stem=path.stem,
            suffix=path.suffix.lower(),
        )

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
