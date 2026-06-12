from __future__ import annotations

from pathlib import Path

from tabcaddy.domain.models import DatasetSource, SourceType
from tabcaddy.shared.dataset_io import SUPPORTED_FILE_SUFFIXES


def is_compiled_dataset(path: Path) -> bool:
    return (
        path.is_dir() and (path / "metadata.json").exists() and (path / "data").is_dir()
    )


def iter_dataset_files(source: DatasetSource) -> list[Path]:
    if source.source_type == SourceType.FILE:
        return [source.path]
    if source.source_type == SourceType.COMPILED_DATASET:
        return sorted((source.path / "data").glob("*.parquet"))
    return sorted(
        path
        for path in source.path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_SUFFIXES
    )


def resolve_source(path: Path) -> DatasetSource:
    target = path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Source does not exist: {target}")
    if target.is_file():
        if target.suffix.lower() not in SUPPORTED_FILE_SUFFIXES:
            raise ValueError(f"Unsupported file type: {target.suffix}")
        return DatasetSource(path=target, source_type=SourceType.FILE)
    if is_compiled_dataset(target):
        return DatasetSource(path=target, source_type=SourceType.COMPILED_DATASET)
    files = [
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_FILE_SUFFIXES
    ]
    if not files:
        raise ValueError(f"No supported files found under: {target}")
    return DatasetSource(path=target, source_type=SourceType.FOLDER)