from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from tabcaddy.analysis.sources import SUPPORTED_FILE_SUFFIXES, is_compiled_dataset


_BINARY_FILE_SUFFIXES = {".parquet", ".feather", ".arrow"}
_CSV_SUFFIX = ".csv"
_FEATHER_ARROW_SUFFIXES = {".feather", ".arrow"}
MergeOperationKind = Literal["merge", "source_only", "target_passthrough"]
SchemaEvolution = Literal["strict", "allow-additive"]


@dataclass(frozen=True)
class PlannedOperation:
    source: Path
    target: Path | None
    destination: Path
    output_directory: bool
    kind: MergeOperationKind = "merge"


@dataclass(frozen=True)
class MatchKey:
    relative_parent: str | None
    stem: str
    suffix: str | None


@dataclass(frozen=True)
class ValidationResult:
    target_schema: pl.Schema
    effective_schema: pl.Schema
    cast_source_to_target_schema: bool
    conflicting_columns: list[str]


@dataclass(frozen=True)
class PreparedOperation:
    source: Path
    target: Path | None
    destination: Path
    validation: ValidationResult | None = None


def resolve_merge_path(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{role.capitalize()} does not exist: {resolved}")
    if resolved.is_file() and resolved.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(f"Unsupported file type: {resolved.suffix}")
    if is_compiled_dataset(resolved):
        raise ValueError(
            f"Merge does not support compiled datasets for {role}: {resolved}. "
            "Use files or folders, or merge the raw inputs and compile again."
        )
    return resolved


def iter_supported_files(folder: Path, allow_empty: bool = False) -> list[Path]:
    files = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_SUFFIXES
    )
    if not files and not allow_empty:
        raise ValueError(f"No supported files found under: {folder}")
    return files


def build_directory_index(
    target_folder: Path,
    ignore_filetype: bool,
    relative_to: Path | None = None,
) -> dict[MatchKey, Path]:
    duplicates: defaultdict[MatchKey, list[Path]] = defaultdict(list)
    for path in iter_supported_files(target_folder, allow_empty=True):
        key = match_key(path, ignore_filetype, relative_to=relative_to)
        duplicates[key].append(path)

    for paths in duplicates.values():
        if len(paths) > 1:
            sample = ", ".join(str(path) for path in paths)
            raise ValueError(f"Ambiguous target match detected: {sample}")

    return {key: paths[0] for key, paths in duplicates.items()}


def match_indexed_file(
    source: Path,
    target_index: dict[MatchKey, Path],
    ignore_filetype: bool,
    relative_to: Path | None = None,
) -> Path | None:
    return target_index.get(match_key(source, ignore_filetype, relative_to=relative_to))


def match_key(
    path: Path,
    ignore_filetype: bool,
    relative_to: Path | None = None,
) -> MatchKey:
    relative_parent: str | None = None
    if relative_to is not None:
        relative_parent = path.relative_to(relative_to).parent.as_posix()
    if ignore_filetype:
        return MatchKey(relative_parent=relative_parent, stem=path.stem, suffix=None)
    return MatchKey(
        relative_parent=relative_parent,
        stem=path.stem,
        suffix=path.suffix.lower(),
    )


def supports_merge_pair(source: Path, target: Path, ignore_filetype: bool) -> bool:
    return ignore_filetype or source.suffix.lower() == target.suffix.lower()


def scan_dataframe(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    if suffix == _CSV_SUFFIX:
        return pl.scan_csv(str(path), infer_schema_length=1000, try_parse_dates=True)
    if suffix == ".parquet":
        return pl.scan_parquet(str(path))
    if suffix in _FEATHER_ARROW_SUFFIXES:
        return pl.scan_ipc(str(path), memory_map=False)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def cast_lazyframe(frame: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
    return frame.with_columns(
        pl.col(column).cast(dtype, strict=True) for column, dtype in schema.items()
    )


def align_lazyframe_to_schema(frame: pl.LazyFrame, schema: pl.Schema) -> pl.LazyFrame:
    frame_schema = frame.collect_schema()
    return frame.select(
        (
            pl.col(column).cast(dtype, strict=True)
            if column in frame_schema
            else pl.lit(None, dtype=dtype)
        ).alias(column)
        for column, dtype in schema.items()
    )


def stage_lazyframe(
    lazy_frame: pl.LazyFrame,
    destination: Path,
    format_hint: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    try:
        sink_lazyframe(lazy_frame, destination, format_hint)
    except pl.exceptions.PolarsError as error:
        cleanup_temp_file(destination)
        raise ValueError(str(error)) from error
    except Exception:
        cleanup_temp_file(destination)
        raise


def sink_lazyframe(
    lazy_frame: pl.LazyFrame,
    destination: Path,
    format_hint: Path,
) -> None:
    suffix = format_hint.suffix.lower()
    if suffix == _CSV_SUFFIX:
        lazy_frame.sink_csv(str(destination))
        return
    if suffix == ".parquet":
        lazy_frame.sink_parquet(str(destination))
        return
    if suffix in _FEATHER_ARROW_SUFFIXES:
        lazy_frame.sink_ipc(str(destination))
        return
    raise ValueError(f"Unsupported file type: {format_hint.suffix}")


def cleanup_temp_file(path: Path) -> None:
    if path.exists():
        path.unlink()


def is_binary(path: Path) -> bool:
    return path.suffix.lower() in _BINARY_FILE_SUFFIXES


def is_csv(path: Path) -> bool:
    return path.suffix.lower() == _CSV_SUFFIX


def temp_path(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp")
