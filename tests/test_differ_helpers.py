from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from tabcaddy.analysis import resolve_source
from tabcaddy.diff import MatchStatus, diff_folder_inventory, resolve_file_folder_match


def test_folder_inventory_reports_added_removed_and_modified_files(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    pl.DataFrame({"id": [1], "value": [10]}).write_csv(left / "same.csv")
    pl.DataFrame({"id": [1], "value": [10]}).write_csv(right / "same.csv")
    pl.DataFrame({"id": [2], "value": [20]}).write_csv(left / "removed.csv")
    pl.DataFrame({"id": [3], "value": [30]}).write_csv(right / "added.csv")
    pl.DataFrame({"id": [4], "value": [40]}).write_csv(left / "changed.csv")
    pl.DataFrame({"id": [4], "value": [99]}).write_csv(right / "changed.csv")

    inventory = diff_folder_inventory(resolve_source(left), resolve_source(right))

    assert inventory.added_files == ["added.csv"]
    assert inventory.removed_files == ["removed.csv"]
    assert inventory.modified_files == ["changed.csv"]
    assert inventory.matching_files == 1
    assert inventory.only_in_left == 1
    assert inventory.only_in_right == 1


def test_folder_inventory_uses_hash_check_when_only_mtime_differs(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    frame = pl.DataFrame({"id": [1], "value": [10]})
    left_file = left / "data.csv"
    right_file = right / "data.csv"
    frame.write_csv(left_file)
    frame.write_csv(right_file)

    left_stat = left_file.stat()
    right_file.touch()
    right_file.touch()

    inventory = diff_folder_inventory(resolve_source(left), resolve_source(right))

    assert right_file.stat().st_mtime >= left_stat.st_mtime
    assert inventory.modified_files == []
    assert inventory.matching_files == 1


def test_folder_inventory_hashes_same_size_files_even_when_mtime_matches(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    left_file = left / "changed.csv"
    right_file = right / "changed.csv"
    pl.DataFrame({"id": [4], "value": [40]}).write_csv(left_file)
    pl.DataFrame({"id": [4], "value": [99]}).write_csv(right_file)

    stat = left_file.stat()
    os.utime(right_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    inventory = diff_folder_inventory(resolve_source(left), resolve_source(right))

    assert left_file.stat().st_size == right_file.stat().st_size
    assert left_file.stat().st_mtime_ns == right_file.stat().st_mtime_ns
    assert inventory.modified_files == ["changed.csv"]
    assert inventory.matching_files == 0


def test_file_folder_match_returns_ambiguous_candidates_in_relative_order(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    nested = folder / "nested"
    folder.mkdir()
    nested.mkdir()

    frame = pl.DataFrame({"id": [1], "value": [10]})
    frame.write_csv(source_file)
    frame.write_csv(folder / "data.csv")
    frame.write_csv(nested / "data.csv")

    match = resolve_file_folder_match(
        resolve_source(source_file), resolve_source(folder)
    )

    assert match.status is MatchStatus.AMBIGUOUS
    assert match.candidate_paths == ["data.csv", "nested/data.csv"]
    assert match.matched_file is None


def test_file_folder_match_prefers_exact_content_match_when_names_collide(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    nested = folder / "nested"
    folder.mkdir()
    nested.mkdir()

    source_frame = pl.DataFrame({"id": [1], "value": [10]})
    source_frame.write_csv(source_file)
    pl.DataFrame({"id": [1], "value": [99]}).write_csv(folder / "data.csv")
    source_frame.write_csv(nested / "data.csv")

    match = resolve_file_folder_match(
        resolve_source(source_file), resolve_source(folder)
    )

    assert match.status is MatchStatus.UNMODIFIED
    assert match.matched_path == "nested/data.csv"
    assert match.matched_file == nested / "data.csv"
