from __future__ import annotations

from hashlib import file_digest
from pathlib import Path


def content_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def content_matches(
    left: Path,
    right: Path,
    *,
    hash_cache: dict[Path, str] | None = None,
) -> bool:
    return _cached_content_hash(left, hash_cache) == _cached_content_hash(
        right,
        hash_cache,
    )


def relative_dataset_path(base_path: Path, path: Path) -> str:
    return str(path.relative_to(base_path)).replace("\\", "/")


def _cached_content_hash(path: Path, hash_cache: dict[Path, str] | None) -> str:
    if hash_cache is None:
        return content_hash(path)
    if path not in hash_cache:
        hash_cache[path] = content_hash(path)
    return hash_cache[path]
