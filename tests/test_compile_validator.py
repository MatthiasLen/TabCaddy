from __future__ import annotations

from pathlib import Path

import polars as pl

from tabcaddy.compilation import ValidateCompiledDataset


def _write_source_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    pl.DataFrame(rows).write_csv(path)
    return path


def _write_compiled_part(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_validator_passes_and_reports_skipped_as_warning_only(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    selected_a = _write_source_csv(
        source_root / "a.csv",
        [{"id": 1, "value": 10.0}, {"id": 2, "value": 20.0}],
    )
    selected_b = _write_source_csv(
        source_root / "b.csv",
        [{"id": 3, "value": 30.0}],
    )

    compiled_part = _write_compiled_part(
        tmp_path / "compiled" / "data" / "part-001.parquet",
        [
            {"id": 1, "value": 10.0, "_source_file": "a.csv"},
            {"id": 2, "value": 20.0, "_source_file": "a.csv"},
            {"id": 3, "value": 30.0, "_source_file": "b.csv"},
        ],
    )

    result = ValidateCompiledDataset().run(
        source_root=source_root,
        selected_files=[selected_a, selected_b],
        skipped_files=["c.csv"],
        compiled_parts=[compiled_part],
        expected_columns={"id", "value", "_source_file"},
    )

    assert result.passed is True
    assert result.errors == []
    assert any(
        "Excluded files from compilation" in warning for warning in result.warnings
    )


def test_validator_fails_when_selected_file_rows_are_missing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    selected_a = _write_source_csv(
        source_root / "a.csv",
        [{"id": 1, "value": 10.0}],
    )
    selected_b = _write_source_csv(
        source_root / "b.csv",
        [{"id": 2, "value": 20.0}],
    )

    compiled_part = _write_compiled_part(
        tmp_path / "compiled" / "data" / "part-001.parquet",
        [{"id": 1, "value": 10.0, "_source_file": "a.csv"}],
    )

    result = ValidateCompiledDataset().run(
        source_root=source_root,
        selected_files=[selected_a, selected_b],
        skipped_files=[],
        compiled_parts=[compiled_part],
        expected_columns={"id", "value", "_source_file"},
    )

    assert result.passed is False
    assert any("missing rows for selected files" in error for error in result.errors)


def test_validator_fails_on_unexpected_source_file_rows(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    selected_a = _write_source_csv(
        source_root / "a.csv",
        [{"id": 1, "value": 10.0}],
    )

    compiled_part = _write_compiled_part(
        tmp_path / "compiled" / "data" / "part-001.parquet",
        [
            {"id": 1, "value": 10.0, "_source_file": "a.csv"},
            {"id": 9, "value": 90.0, "_source_file": "c.csv"},
        ],
    )

    result = ValidateCompiledDataset().run(
        source_root=source_root,
        selected_files=[selected_a],
        skipped_files=[],
        compiled_parts=[compiled_part],
        expected_columns={"id", "value", "_source_file"},
    )

    assert result.passed is False
    assert any("rows from non-selected files" in error for error in result.errors)


def test_validator_fails_on_row_count_mismatch(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    selected_a = _write_source_csv(
        source_root / "a.csv",
        [{"id": 1, "value": 10.0}, {"id": 2, "value": 20.0}],
    )

    compiled_part = _write_compiled_part(
        tmp_path / "compiled" / "data" / "part-001.parquet",
        [{"id": 1, "value": 10.0, "_source_file": "a.csv"}],
    )

    result = ValidateCompiledDataset().run(
        source_root=source_root,
        selected_files=[selected_a],
        skipped_files=[],
        compiled_parts=[compiled_part],
        expected_columns={"id", "value", "_source_file"},
    )

    assert result.passed is False
    assert any("row count mismatch" in error for error in result.errors)


def test_validator_fails_when_source_file_column_is_missing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    selected_a = _write_source_csv(
        source_root / "a.csv",
        [{"id": 1, "value": 10.0}],
    )

    compiled_part = _write_compiled_part(
        tmp_path / "compiled" / "data" / "part-001.parquet",
        [{"id": 1, "value": 10.0}],
    )

    result = ValidateCompiledDataset().run(
        source_root=source_root,
        selected_files=[selected_a],
        skipped_files=[],
        compiled_parts=[compiled_part],
        expected_columns={"id", "value", "_source_file"},
    )

    assert result.passed is False
    assert any("missing expected columns" in error for error in result.errors)


def test_validator_reports_unreadable_selected_files_without_row_mismatch(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()

    selected_bad = source_root / "bad.feather"
    selected_bad.write_bytes(b"not-a-valid-feather")

    compiled_part = _write_compiled_part(
        tmp_path / "compiled" / "data" / "part-001.parquet",
        [{"id": 1, "_source_file": "bad.feather"}],
    )

    result = ValidateCompiledDataset().run(
        source_root=source_root,
        selected_files=[selected_bad],
        skipped_files=[],
        compiled_parts=[compiled_part],
        expected_columns={"id", "_source_file"},
    )

    assert result.passed is False
    assert any(
        "Unable to read selected source file during validation" in error
        for error in result.errors
    )
    assert not any("row count mismatch" in error for error in result.errors)
