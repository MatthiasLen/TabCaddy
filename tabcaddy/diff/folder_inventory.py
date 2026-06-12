from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.diff.hash_utils import content_hash, relative_dataset_path
from tabcaddy.domain.models import DatasetSource


@dataclass(frozen=True)
class FolderInventoryDiff:
    added_files: list[str]
    removed_files: list[str]
    modified_files: list[str]
    matching_files: int
    only_in_left: int
    only_in_right: int


def build_relative_file_index(source: DatasetSource) -> dict[str, Path]:
    return {
        relative_dataset_path(source.path, path): path
        for path in iter_dataset_files(source)
    }


def diff_folder_inventory(
    left: DatasetSource,
    right: DatasetSource,
) -> FolderInventoryDiff:
    left_files = build_relative_file_index(left)
    right_files = build_relative_file_index(right)
    added_files = sorted(right_files.keys() - left_files.keys())
    removed_files = sorted(left_files.keys() - right_files.keys())
    modified_files = _find_modified_files(
        left_files,
        right_files,
        sorted(left_files.keys() & right_files.keys()),
    )
    return FolderInventoryDiff(
        added_files=added_files,
        removed_files=removed_files,
        modified_files=modified_files,
        matching_files=len(left_files.keys() & right_files.keys())
        - len(modified_files),
        only_in_left=len(removed_files),
        only_in_right=len(added_files),
    )


def _find_modified_files(
    left_files: dict[str, Path],
    right_files: dict[str, Path],
    common_files: list[str],
) -> list[str]:
    modified: list[str] = []
    if not common_files:
        return modified

    hash_needed: list[tuple[str, Path, Path]] = []
    for file_name in common_files:
        left_path = left_files[file_name]
        right_path = right_files[file_name]
        left_stat = left_path.stat()
        right_stat = right_path.stat()

        if left_stat.st_size != right_stat.st_size:
            modified.append(file_name)
        else:
            hash_needed.append((file_name, left_path, right_path))

    if not hash_needed:
        return modified

    modified.extend(_check_file_hashes(hash_needed))
    return sorted(modified)


def _check_file_hashes(file_tuples: list[tuple[str, Path, Path]]) -> list[str]:
    modified: list[str] = []
    hash_cache: dict[Path, str] = {}

    max_workers = min(8, len(file_tuples) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(content_hash, path): path
            for _, left_path, right_path in file_tuples
            for path in (left_path, right_path)
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            hash_cache[path] = future.result()

    for file_name, left_path, right_path in file_tuples:
        if hash_cache.get(left_path) != hash_cache.get(right_path):
            modified.append(file_name)

    return modified
