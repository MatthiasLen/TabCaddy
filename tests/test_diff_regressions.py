from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import polars as pl
from rich.console import Console
from typer.testing import CliRunner

from tabcaddy.cli.app import app
from tabcaddy.domain.models import (
    DiffComparisonType,
    DiffReport,
    DiffSummary,
    RowChangeExample,
    RowFieldDelta,
)
from tabcaddy.rendering.console import RenderProfile, SafeConsole
from tabcaddy.rendering.views.diff import build_diff_view


runner = CliRunner()


def _write_compiled_dataset(root: Path, frame: pl.DataFrame) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    part_path = data_dir / "part-001.parquet"
    frame.write_parquet(part_path)

    metadata = {
        "metadata": {
            "version": 1,
            "created_at": "2026-01-01T00:00:00+00:00",
            "row_count": frame.height,
            "column_count": len(frame.columns),
            "source_file_count": 1,
            "schema_hash": "schema-1",
            "column_hashes": {
                column: f"hash-{index}"
                for index, column in enumerate(frame.columns, start=1)
            },
        },
        "schemas": [
            {
                "columns": [
                    {"name": name, "dtype": str(dtype)}
                    for name, dtype in frame.schema.items()
                ],
                "hash": "schema-1",
                "occurrence_count": 1,
            }
        ],
        "statistics": None,
        "warnings": [],
        "compiled": {
            "source": "fixture-source",
            "selected_schema_hash": "schema-1",
            "written_parts": ["data/part-001.parquet"],
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_diff_safe_console_output_falls_back_on_encoding_errors() -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")

    console = SafeConsole(file=stream, force_terminal=False, color_system=None)
    with patch.object(
        Console,
        "print",
        side_effect=UnicodeEncodeError("cp1252", "π", 0, 1, "cannot encode"),
    ):
        console.print("Failed π to render")

    stream.flush()
    text = buffer.getvalue().decode("cp1252")
    assert "Failed ? to render" in text


def test_diff_ascii_render_profile_escapes_unicode_row_values() -> None:
    report = DiffReport(
        summary=DiffSummary(
            comparison_type=DiffComparisonType.FILE,
            content_status="modified",
        ),
        row_change_examples=[
            RowChangeExample(
                key={"id": "α"},
                deltas=[
                    RowFieldDelta(column="txt", left_value="π🙂", right_value="π🙃")
                ],
            )
        ],
    )

    view = build_diff_view(report, render=RenderProfile(ascii_only=True))
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252")
    console = Console(file=stream, force_terminal=False, color_system=None)

    console.print(view)
    stream.flush()
    text = buffer.getvalue().decode("cp1252")

    assert "\\u03b1" in text
    assert "\\U0001f642" in text
    assert "\\U0001f643" in text


def test_diff_file_with_corrupted_parquet_fails_without_row_keys(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    pl.DataFrame({"id": [1, 2], "v": [10, 20]}).write_parquet(left)
    right.write_bytes(b"not-a-parquet-file")

    result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])

    assert result.exit_code == 1
    assert "failed to inspect any dataset files" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_diff_folder_with_only_corrupted_parquet_fails_without_row_keys(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "bad.parquet").write_bytes(b"left-corrupt")
    (right / "bad.parquet").write_bytes(b"right-corrupt")

    result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])

    assert result.exit_code == 1
    assert "failed to inspect any dataset files" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_diff_compiled_with_corrupted_parts_fails_without_row_keys(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left_compiled"
    right = tmp_path / "right_compiled"
    _write_compiled_dataset(left, pl.DataFrame({"id": [1, 2], "v": [10, 20]}))
    _write_compiled_dataset(right, pl.DataFrame({"id": [1, 2], "v": [10, 20]}))
    (left / "data" / "part-001.parquet").write_bytes(b"left-corrupt")
    (right / "data" / "part-001.parquet").write_bytes(b"right-corrupt")

    result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])

    assert result.exit_code == 1
    assert "failed to inspect any dataset files" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()
