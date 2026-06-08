from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from tabcaddy.cli.app import app


runner = CliRunner()


def _write_frame(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(rows)
    if path.suffix == ".csv":
        frame.write_csv(path)
        return
    if path.suffix == ".parquet":
        frame.write_parquet(path)
        return
    frame.write_ipc(path)


def _read_frame(path: Path) -> pl.DataFrame:
    if path.suffix == ".csv":
        return pl.read_csv(path, try_parse_dates=True)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    return pl.read_ipc(path, memory_map=False)


def test_merge_file_to_file_row_deduplicates_exact_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    output = tmp_path / "merged.csv"

    _write_frame(source, [{"id": 2, "value": 20}, {"id": 3, "value": 30}])
    _write_frame(target, [{"id": 1, "value": 10}, {"id": 2, "value": 20}])

    result = runner.invoke(
        app, ["merge", str(source), str(target), "--out", str(output)]
    )

    assert result.exit_code == 0
    assert output.exists()
    assert _read_frame(output).to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]
    assert _read_frame(target).to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]


def test_merge_key_conflict_fails_without_writing_output(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"
    output = tmp_path / "merged.csv"

    _write_frame(source, [{"id": 1, "value": 99}])
    _write_frame(target, [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        ["merge", str(source), str(target), "--on", "id", "--out", str(output)],
    )

    assert result.exit_code == 1
    assert not output.exists()
    assert "Conflicting duplicate key" in result.stdout


def test_merge_file_into_folder_inplace_copies_when_no_match_exists(
    tmp_path: Path,
) -> None:
    source = tmp_path / "incoming.csv"
    archive = tmp_path / "archive"
    archive.mkdir()

    _write_frame(source, [{"id": 1, "value": 10}])

    result = runner.invoke(app, ["merge", str(source), str(archive), "--inplace"])

    assert result.exit_code == 0
    copied = archive / "incoming.csv"
    assert copied.exists()
    assert _read_frame(copied).to_dicts() == [{"id": 1, "value": 10}]


def test_merge_file_into_folder_out_file_merges_into_matched_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sales.csv"
    archive = tmp_path / "archive"
    output = tmp_path / "merged.csv"
    archive.mkdir()

    _write_frame(source, [{"id": 2, "value": 20}, {"id": 3, "value": 30}])
    _write_frame(
        archive / "sales.csv", [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
    )

    result = runner.invoke(
        app,
        ["merge", str(source), str(archive), "--out", str(output)],
    )

    assert result.exit_code == 0
    assert _read_frame(output).to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]
    assert _read_frame(archive / "sales.csv").to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]


def test_merge_folder_to_folder_out_dir_merges_matches_and_copies_missing(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_dir = tmp_path / "combined"
    source_dir.mkdir()
    target_dir.mkdir()

    _write_frame(
        source_dir / "sales.csv", [{"id": 2, "value": 20}, {"id": 3, "value": 30}]
    )
    _write_frame(source_dir / "new.csv", [{"id": 9, "value": 90}])
    _write_frame(
        target_dir / "sales.csv", [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
    )
    _write_frame(target_dir / "target_only.csv", [{"id": 8, "value": 80}])

    result = runner.invoke(
        app,
        ["merge", str(source_dir), str(target_dir), "--out", str(output_dir)],
    )

    assert result.exit_code == 0
    assert _read_frame(output_dir / "sales.csv").to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]
    assert _read_frame(output_dir / "new.csv").to_dicts() == [{"id": 9, "value": 90}]
    assert _read_frame(output_dir / "target_only.csv").to_dicts() == [
        {"id": 8, "value": 80}
    ]


def test_merge_folder_to_folder_out_rejects_file_path_even_for_single_source_file(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_file = tmp_path / "combined.csv"

    _write_frame(source_dir / "sales.csv", [{"id": 2, "value": 20}])
    _write_frame(target_dir / "sales.csv", [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        ["merge", str(source_dir), str(target_dir), "--out", str(output_file)],
    )

    assert result.exit_code == 1
    assert (
        "Folder-to-folder merge requires --out to point to a directory" in result.stdout
    )
    assert not output_file.exists()


def test_merge_folder_to_folder_matches_by_relative_path(tmp_path: Path) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_dir = tmp_path / "combined"

    _write_frame(source_dir / "eu" / "sales.csv", [{"id": 2, "value": 20}])
    _write_frame(source_dir / "us" / "sales.csv", [{"id": 4, "value": 40}])
    _write_frame(target_dir / "eu" / "sales.csv", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "us" / "sales.csv", [{"id": 3, "value": 30}])

    result = runner.invoke(
        app,
        ["merge", str(source_dir), str(target_dir), "--out", str(output_dir)],
    )

    assert result.exit_code == 0
    assert _read_frame(output_dir / "eu" / "sales.csv").to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]
    assert _read_frame(output_dir / "us" / "sales.csv").to_dicts() == [
        {"id": 3, "value": 30},
        {"id": 4, "value": 40},
    ]


def test_merge_folder_to_folder_accepts_dotted_out_directory_name(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_dir = tmp_path / "combined.v2"

    _write_frame(source_dir / "sales.csv", [{"id": 2, "value": 20}])
    _write_frame(target_dir / "sales.csv", [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        ["merge", str(source_dir), str(target_dir), "--out", str(output_dir)],
    )

    assert result.exit_code == 0
    assert _read_frame(output_dir / "sales.csv").to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]


def test_merge_file_into_folder_uses_existing_dotted_output_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sales.csv"
    archive = tmp_path / "archive"
    output_dir = tmp_path / "combined.v2"

    archive.mkdir()
    output_dir.mkdir()
    _write_frame(source, [{"id": 2, "value": 20}, {"id": 3, "value": 30}])
    _write_frame(
        archive / "sales.csv", [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
    )

    result = runner.invoke(
        app,
        ["merge", str(source), str(archive), "--out", str(output_dir)],
    )

    assert result.exit_code == 0
    assert _read_frame(output_dir / "sales.csv").to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]


def test_merge_ignore_filetype_casts_csv_into_binary_target(tmp_path: Path) -> None:
    source = tmp_path / "sales.csv"
    archive = tmp_path / "archive"
    output = tmp_path / "merged.parquet"
    archive.mkdir()

    _write_frame(source, [{"id": "2", "value": "20"}, {"id": "3", "value": "30"}])
    _write_frame(
        archive / "sales.parquet", [{"id": 1, "value": 10}, {"id": 2, "value": 20}]
    )

    result = runner.invoke(
        app,
        [
            "merge",
            str(source),
            str(archive),
            "--on",
            "id",
            "--out",
            str(output),
            "--ignore-filetype",
        ],
    )

    assert result.exit_code == 0
    merged = _read_frame(output)
    assert merged.schema == {"id": pl.Int64, "value": pl.Int64}
    assert merged.to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]


def test_merge_ignore_filetype_rejects_uncoercible_csv_values(tmp_path: Path) -> None:
    source = tmp_path / "sales.csv"
    archive = tmp_path / "archive"
    output = tmp_path / "merged.parquet"
    archive.mkdir()

    _write_frame(source, [{"id": "bad", "value": "oops"}])
    _write_frame(archive / "sales.parquet", [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        [
            "merge",
            str(source),
            str(archive),
            "--on",
            "id",
            "--out",
            str(output),
            "--ignore-filetype",
        ],
    )

    assert result.exit_code == 1
    assert not output.exists()


def test_merge_ignore_filetype_matches_nested_files_by_relative_path(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_dir = tmp_path / "combined"

    _write_frame(source_dir / "eu" / "sales.csv", [{"id": "2", "value": "20"}])
    _write_frame(source_dir / "us" / "sales.csv", [{"id": "4", "value": "40"}])
    _write_frame(target_dir / "eu" / "sales.parquet", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "us" / "sales.parquet", [{"id": 3, "value": 30}])

    result = runner.invoke(
        app,
        [
            "merge",
            str(source_dir),
            str(target_dir),
            "--out",
            str(output_dir),
            "--ignore-filetype",
        ],
    )

    assert result.exit_code == 0
    assert _read_frame(output_dir / "eu" / "sales.parquet").to_dicts() == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
    ]
    assert _read_frame(output_dir / "us" / "sales.parquet").to_dicts() == [
        {"id": 3, "value": 30},
        {"id": 4, "value": 40},
    ]
