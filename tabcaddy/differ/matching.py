from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from tabcaddy.differ.hash_utils import (
    content_hash,
    content_matches,
    relative_dataset_path,
)
from tabcaddy.domain.models import DatasetSource
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


class MatchStatus(str, Enum):
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNMODIFIED = "unmodified"
    MODIFIED = "modified"


@dataclass(frozen=True)
class FileFolderMatch:
    status: MatchStatus
    matched_path: str | None = None
    candidate_paths: list[str] = field(default_factory=list)
    matched_file: Path | None = None


def resolve_file_folder_match(
    file_source: DatasetSource,
    folder_source: DatasetSource,
) -> FileFolderMatch:
    source_hash: str | None = None
    hash_cache: dict[Path, str] = {}
    matches = [
        path
        for path in iter_dataset_files(folder_source)
        if path.name == file_source.path.name
    ]
    if not matches:
        return FileFolderMatch(status=MatchStatus.MISSING)

    if len(matches) > 1:
        source_hash = content_hash(file_source.path)
        hash_cache[file_source.path] = source_hash
        exact_matches = [
            path
            for path in matches
            if content_matches(file_source.path, path, hash_cache=hash_cache)
        ]
        if len(exact_matches) == 1:
            matches = exact_matches
        else:
            return FileFolderMatch(
                status=MatchStatus.AMBIGUOUS,
                candidate_paths=[
                    relative_dataset_path(folder_source.path, path) for path in matches
                ],
            )

    matched_file = matches[0]
    matched_path = relative_dataset_path(folder_source.path, matched_file)
    if source_hash is None:
        source_hash = content_hash(file_source.path)
        hash_cache[file_source.path] = source_hash
    if content_matches(file_source.path, matched_file, hash_cache=hash_cache):
        return FileFolderMatch(
            status=MatchStatus.UNMODIFIED,
            matched_path=matched_path,
            matched_file=matched_file,
        )
    return FileFolderMatch(
        status=MatchStatus.MODIFIED,
        matched_path=matched_path,
        matched_file=matched_file,
    )
