from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from tabcaddy import __version__
from tabcaddy.cli.app import app
from tabcaddy.compilation import ValidationResult


runner = CliRunner()


def test_version_option_displays_current_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_nested_parquet(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def _write_compiled_dataset(
    root: Path,
    frame: pl.DataFrame,
    *,
    source: str = "fixture-source",
) -> None:
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
            "source": source,
            "selected_schema_hash": "schema-1",
            "written_parts": ["data/part-001.parquet"],
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_summary_and_schema_commands(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}])
    _write_csv(data / "b.csv", [{"id": 3, "value": 12.0}])

    summary_result = runner.invoke(app, ["summary", str(data)])
    schema_result = runner.invoke(app, ["schema", str(data)])

    assert summary_result.exit_code == 0
    assert "Metadata" in summary_result.stdout
    assert schema_result.exit_code == 0
    assert "Schema Groups" in schema_result.stdout


def test_schema_command_populates_analysis_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}])

    schema_result = runner.invoke(app, ["schema", str(data)])

    assert schema_result.exit_code == 0
    cache_root = tmp_path / ".tabcaddy" / "cache"
    assert cache_root.exists()
    assert any(cache_root.glob("*.json"))


def test_schema_command_shows_warnings_for_unreadable_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame({"id": [1, 2], "value": [10.0, 11.0]}).write_parquet(
        data / "good.parquet"
    )
    (data / "bad.parquet").write_bytes(b"not-a-valid-parquet")

    schema_result = runner.invoke(app, ["schema", str(data)])

    assert schema_result.exit_code == 0
    assert "Warnings" in schema_result.stdout
    assert "Failed to inspect bad.parquet" in schema_result.stdout


def test_scaffold_transform_populates_analysis_cache(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}])

    scaffold_result = runner.invoke(
        app,
        ["scaffold-transform", str(data), "--output", str(tmp_path / "transform.py")],
    )

    assert scaffold_result.exit_code == 0
    cache_root = tmp_path / ".tabcaddy" / "cache"
    assert cache_root.exists()
    assert any(cache_root.glob("*.json"))


def test_scaffold_transform_fails_when_output_file_exists(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}])

    output_file = tmp_path / "transform.py"
    output_file.write_text("# existing", encoding="utf-8")

    scaffold_result = runner.invoke(
        app,
        ["scaffold-transform", str(data), "--output", str(output_file)],
    )

    assert scaffold_result.exit_code == 2


def test_transform_fails_cleanly_when_output_folder_exists(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}])

    transform_script = tmp_path / "transform.py"
    transform_script.write_text(
        "import polars as pl\n\ndef transform(df, context=None):\n    return df\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "transformed"
    output_dir.mkdir()

    transform_result = runner.invoke(
        app,
        ["transform", str(data), str(transform_script), str(output_dir)],
    )

    assert transform_result.exit_code == 1
    assert "Output folder already exists" in transform_result.stdout
    assert "Traceback" not in transform_result.stdout


def test_transform_single_file_output_path_like_file_writes_file(
    tmp_path: Path,
) -> None:
    input_file = tmp_path / "input.csv"
    _write_csv(input_file, [{"id": 1, "value": 10.0}])

    transform_script = tmp_path / "transform.py"
    transform_script.write_text(
        "import polars as pl\n\ndef transform(df, context=None):\n    return df\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "output.csv"
    transform_result = runner.invoke(
        app,
        ["transform", str(input_file), str(transform_script), str(output_file)],
    )

    assert transform_result.exit_code == 0
    assert output_file.is_file()
    assert not output_file.is_dir()
    assert not (tmp_path / "output.csv" / "input.csv").exists()

    transformed = pl.read_csv(output_file)
    assert transformed.height == 1
    assert transformed["value"][0] == 10.0


def test_transform_shows_warnings_for_unreadable_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "good.csv", [{"id": 1, "value": 10.0}])
    (data / "bad.parquet").write_bytes(b"not-a-valid-parquet")

    transform_script = tmp_path / "transform.py"
    transform_script.write_text(
        "import polars as pl\n\ndef transform(df, context=None):\n    return df\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "transformed"
    transform_result = runner.invoke(
        app,
        ["transform", str(data), str(transform_script), str(output_dir)],
    )

    assert transform_result.exit_code == 0
    assert "Warnings" in transform_result.stdout
    assert "Failed to inspect bad.parquet" in transform_result.stdout
    assert (output_dir / "good.csv").exists()
    assert not (output_dir / "bad.parquet").exists()


def test_compile_transform_scaffold_and_diff_commands(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_csv(left / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}])
    _write_csv(left / "b.csv", [{"id": 3, "value": 12.0}])
    _write_csv(right / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 20.0}])
    _write_csv(right / "b.csv", [{"id": 3, "value": 12.0}])

    compile_result = runner.invoke(
        app, ["compile", str(left), "--output", str(tmp_path / "compiled")]
    )
    assert compile_result.exit_code == 0
    assert (tmp_path / "compiled" / "metadata.json").exists()
    assert any((tmp_path / "compiled" / "data").iterdir())

    scaffold_result = runner.invoke(
        app,
        ["scaffold-transform", str(left), "--output", str(tmp_path / "transform.py")],
    )
    assert scaffold_result.exit_code == 0
    transform_path = tmp_path / "transform.py"
    transform_path.write_text(
        "import polars as pl\n\n"
        "def transform(df, context=None):\n"
        "    return df.filter(pl.col('value') >= 11)\n",
        encoding="utf-8",
    )
    transform_result = runner.invoke(
        app,
        ["transform", str(left), str(transform_path), str(tmp_path / "transformed")],
    )
    assert transform_result.exit_code == 0
    transformed = pl.read_csv(tmp_path / "transformed" / "a.csv")
    assert transformed.height == 1

    diff_result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])
    assert diff_result.exit_code == 0
    assert "Summary" in diff_result.stdout
    assert "Statistics Changes" in diff_result.stdout
    assert "File Changes" in diff_result.stdout
    assert "Dataset Metadata" in diff_result.stdout


def test_compile_accepts_parquet_inputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(
        [
            {"id": 1, "value": 10.0},
            {"id": 2, "value": 11.0},
        ]
    ).write_parquet(data / "a.parquet")

    compile_result = runner.invoke(
        app, ["compile", str(data), "--output", str(tmp_path / "compiled")]
    )

    assert compile_result.exit_code == 0
    assert (tmp_path / "compiled" / "metadata.json").exists()
    assert any((tmp_path / "compiled" / "data").glob("*.parquet"))


def test_compile_shows_warnings_for_unreadable_files(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pl.DataFrame(
        [
            {"id": 1, "value": 10.0},
            {"id": 2, "value": 11.0},
        ]
    ).write_parquet(data / "good.parquet")
    (data / "bad.parquet").write_bytes(b"not-a-valid-parquet")

    compile_result = runner.invoke(
        app, ["compile", str(data), "--output", str(tmp_path / "compiled")]
    )

    assert compile_result.exit_code == 0
    assert "Warnings" in compile_result.stdout
    assert "Failed to inspect bad.parquet" in compile_result.stdout
    assert (tmp_path / "compiled" / "metadata.json").exists()


def test_head_accepts_parquet_file_and_folder_inputs(tmp_path: Path) -> None:
    parquet_file = tmp_path / "single.parquet"
    pl.DataFrame([{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}]).write_parquet(
        parquet_file
    )

    file_result = runner.invoke(app, ["head", str(parquet_file), "--n", "1"])
    assert file_result.exit_code == 0
    assert "id" in file_result.stdout
    assert "value" in file_result.stdout

    folder = tmp_path / "parquet_folder"
    folder.mkdir()
    pl.DataFrame([{"id": 10, "value": 100.0}]).write_parquet(folder / "a.parquet")
    pl.DataFrame([{"id": 20, "value": 200.0}]).write_parquet(folder / "b.parquet")

    folder_result = runner.invoke(app, ["head", str(folder), "--n", "2"])
    assert folder_result.exit_code == 0
    assert "a.parquet" in folder_result.stdout
    assert "b.parquet" in folder_result.stdout


def test_head_rejects_negative_n_for_file_and_folder(tmp_path: Path) -> None:
    csv_file = tmp_path / "single.csv"
    _write_csv(csv_file, [{"id": 1, "value": 10.0}])

    file_result = runner.invoke(app, ["head", str(csv_file), "--n", "-1"])
    assert file_result.exit_code == 1
    assert "--n must be greater than or equal to 0" in file_result.stdout
    assert "Traceback" not in file_result.stdout

    folder = tmp_path / "csv_folder"
    folder.mkdir()
    _write_csv(folder / "a.csv", [{"id": 1, "value": 10.0}])
    _write_csv(folder / "b.csv", [{"id": 2, "value": 20.0}])

    folder_result = runner.invoke(app, ["head", str(folder), "--n", "-1"])
    assert folder_result.exit_code == 1
    assert "--n must be greater than or equal to 0" in folder_result.stdout
    assert "Traceback" not in folder_result.stdout


def test_diff_metadata_level_hides_schema_and_statistics_sections(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_csv(left / "a.csv", [{"id": 1, "value": 10.0}])
    _write_csv(right / "a.csv", [{"id": 1, "value": 11.0}])

    diff_result = runner.invoke(
        app, ["diff", str(left), str(right), "--level", "metadata"]
    )

    assert diff_result.exit_code == 0
    assert "File Changes" in diff_result.stdout
    assert "Dataset Metadata" not in diff_result.stdout
    assert "Schema Changes" not in diff_result.stdout
    assert "Statistics Changes" not in diff_result.stdout


def test_diff_command_returns_friendly_error_for_unsupported_combination(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "folder"
    compiled = tmp_path / "compiled"
    folder.mkdir()
    compiled.mkdir()
    (compiled / "data").mkdir()
    _write_csv(folder / "a.csv", [{"id": 1, "value": 10.0}])
    (compiled / "metadata.json").write_text(
        '{"metadata": {"version": 1, "created_at": "2024-01-01T00:00:00+00:00", "row_count": 0, "column_count": 0, "source_file_count": 0, "schema_hash": null, "column_hashes": null}, "schemas": [], "statistics": null, "warnings": [], "compiled": {"source": "x", "written_parts": []}}',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["diff", str(folder), str(compiled)])

    assert result.exit_code == 1
    assert "Unsupported diff source combination" in result.stdout


def test_compile_with_validate_returns_nonzero_when_validation_fails(
    tmp_path: Path, homogeneous_folder, monkeypatch
) -> None:
    class _FailingValidator:
        def run(self, **_kwargs) -> ValidationResult:
            _ = _kwargs
            return ValidationResult(
                passed=False,
                errors=["synthetic validation failure"],
            )

    monkeypatch.setattr(
        "tabcaddy.compilation.service.ValidateCompiledDataset",
        _FailingValidator,
    )

    compiled = tmp_path / "compiled_invalid"
    result = runner.invoke(
        app,
        [
            "compile",
            str(homogeneous_folder),
            "--output",
            str(compiled),
            "--validate",
        ],
    )

    assert result.exit_code == 1
    assert "synthetic validation failure" in result.stdout


def test_diff_with_keys_shows_row_level_sections(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(
        left,
        [
            {"customer_id": 1, "status": "active"},
            {"customer_id": 2, "status": "active"},
        ],
    )
    _write_csv(
        right,
        [
            {"customer_id": 1, "status": "active"},
            {"customer_id": 2, "status": "inactive"},
            {"customer_id": 3, "status": "active"},
        ],
    )

    result = runner.invoke(
        app,
        [
            "diff",
            str(left),
            str(right),
            "--level",
            "full",
            "--on",
            "customer_id",
        ],
    )

    assert result.exit_code == 0
    assert "Row Diff Summary" in result.stdout
    assert "Updated Rows" in result.stdout


def test_diff_with_keys_requires_full_level(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"customer_id": 1, "status": "active"}])
    _write_csv(right, [{"customer_id": 1, "status": "inactive"}])

    result = runner.invoke(
        app,
        [
            "diff",
            str(left),
            str(right),
            "--level",
            "statistics",
            "--on",
            "customer_id",
        ],
    )

    assert result.exit_code == 1
    assert "Row-level key diff requires --level full" in result.stdout


def test_diff_row_key_dtype_mismatch_int_vs_str_returns_friendly_error(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    pl.DataFrame({"id": [1, 2], "v": [10, 20]}).write_parquet(left)
    pl.DataFrame({"id": ["1", "2"], "v": [10, 20]}).write_parquet(right)

    result = runner.invoke(
        app,
        ["diff", str(left), str(right), "--level", "full", "--on", "id"],
    )

    assert result.exit_code == 1
    assert "incompatible key column types" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_diff_row_key_dtype_mismatch_date_vs_str_returns_friendly_error(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.parquet"
    right = tmp_path / "right.parquet"
    pl.DataFrame({"id": ["2024-01-01", "2024-01-02"], "v": [10, 20]}).with_columns(
        pl.col("id").str.to_date()
    ).write_parquet(left)
    pl.DataFrame({"id": ["2024-01-01", "2024-01-02"], "v": [10, 20]}).write_parquet(
        right
    )

    result = runner.invoke(
        app,
        ["diff", str(left), str(right), "--level", "full", "--on", "id"],
    )

    assert result.exit_code == 1
    assert "incompatible key column types" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_diff_duplicate_on_arguments_returns_friendly_error(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"id": 1, "v": 10}, {"id": 2, "v": 20}])
    _write_csv(right, [{"id": 1, "v": 11}, {"id": 2, "v": 21}])

    result = runner.invoke(
        app,
        [
            "diff",
            str(left),
            str(right),
            "--level",
            "full",
            "--on",
            "id",
            "--on",
            "id",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate key columns" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_diff_malformed_csv_returns_friendly_error(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    _write_csv(left, [{"id": 1, "v": 10}, {"id": 2, "v": 20}])
    right.write_bytes(b"id,v\n1,10\n2,\xff\n")

    result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])

    assert result.exit_code == 1
    assert "failed to read input data" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_diff_compiled_same_metadata_but_different_data_detects_changes(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left_compiled"
    right = tmp_path / "right_compiled"

    _write_compiled_dataset(left, pl.DataFrame({"id": [1, 2], "v": [10, 20]}))
    _write_compiled_dataset(right, pl.DataFrame({"id": [1, 2], "v": [10, 99]}))

    result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])

    assert result.exit_code == 0
    assert "No differences detected." not in result.stdout
    assert "Modified file:" in result.stdout


def test_diff_compiled_corrupted_part_is_not_reported_as_clean(tmp_path: Path) -> None:
    left = tmp_path / "left_compiled"
    right = tmp_path / "right_compiled"

    _write_compiled_dataset(left, pl.DataFrame({"id": [1, 2], "v": [10, 20]}))
    _write_compiled_dataset(right, pl.DataFrame({"id": [1, 2], "v": [10, 20]}))
    (right / "data" / "part-001.parquet").write_bytes(b"not-a-valid-parquet")

    result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])

    assert result.exit_code == 1
    assert "failed to inspect any dataset files" in result.stdout.lower()
    assert "traceback" not in result.stdout.lower()


def test_nested_parquet_columns_do_not_crash_summary_diff_or_compile(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    _write_nested_parquet(
        left / "nested.parquet",
        [
            {"id": 1, "tags": ["a"], "meta": {"k": "x", "n": 1}},
            {"id": 2, "tags": ["b"], "meta": {"k": "y", "n": 2}},
        ],
    )
    _write_nested_parquet(
        right / "nested.parquet",
        [
            {"id": 1, "tags": ["a"], "meta": {"k": "x", "n": 1}},
            {"id": 3, "tags": ["c"], "meta": {"k": "z", "n": 3}},
        ],
    )

    summary_result = runner.invoke(app, ["summary", str(left)])
    assert summary_result.exit_code == 0
    assert "Traceback" not in summary_result.stdout

    diff_result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])
    assert diff_result.exit_code == 0
    assert "Traceback" not in diff_result.stdout

    compile_result = runner.invoke(
        app,
        ["compile", str(left), "--output", str(tmp_path / "compiled_nested")],
    )
    assert compile_result.exit_code == 0
    assert "Traceback" not in compile_result.stdout
    assert (tmp_path / "compiled_nested" / "metadata.json").exists()


def test_binary_parquet_columns_do_not_crash_summary_diff_or_compile(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    pl.DataFrame(
        [
            {"id": 1, "blob": b"abc"},
            {"id": 2, "blob": b"\xff\xfe"},
        ]
    ).write_parquet(left / "binary.parquet")
    pl.DataFrame(
        [
            {"id": 1, "blob": b"abc"},
            {"id": 3, "blob": b"xyz"},
        ]
    ).write_parquet(right / "binary.parquet")

    summary_result = runner.invoke(app, ["summary", str(left), "--profile", "deep"])
    assert summary_result.exit_code == 0
    assert "Traceback" not in summary_result.stdout

    diff_result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])
    assert diff_result.exit_code == 0
    assert "Traceback" not in diff_result.stdout

    compile_result = runner.invoke(
        app,
        ["compile", str(left), "--output", str(tmp_path / "compiled_binary")],
    )
    assert compile_result.exit_code == 0
    assert "Traceback" not in compile_result.stdout
    assert (tmp_path / "compiled_binary" / "metadata.json").exists()


def test_duration_parquet_columns_do_not_crash_summary_diff_or_compile(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    pl.DataFrame({"id": [1, 2], "dur": [1000, 2000]}).with_columns(
        pl.col("dur").cast(pl.Duration("ms"))
    ).write_parquet(left / "duration.parquet")
    pl.DataFrame({"id": [1, 3], "dur": [1000, 3000]}).with_columns(
        pl.col("dur").cast(pl.Duration("ms"))
    ).write_parquet(right / "duration.parquet")

    summary_result = runner.invoke(app, ["summary", str(left), "--profile", "deep"])
    assert summary_result.exit_code == 0
    assert "Traceback" not in summary_result.stdout

    diff_result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])
    assert diff_result.exit_code == 0
    assert "Traceback" not in diff_result.stdout

    compile_result = runner.invoke(
        app,
        ["compile", str(left), "--output", str(tmp_path / "compiled_duration")],
    )
    assert compile_result.exit_code == 0
    assert "Traceback" not in compile_result.stdout
    assert (tmp_path / "compiled_duration" / "metadata.json").exists()


def test_nan_inf_float_columns_do_not_crash_summary_diff_or_compile(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "x": [float("-inf"), -1.0, 0.0, float("inf")],
            "y": [float("nan"), 1.0, 2.0, 3.0],
        }
    ).write_parquet(left / "edges.parquet")
    pl.DataFrame(
        {
            "id": [1, 2, 3, 5],
            "x": [float("-inf"), -1.0, 1.0, float("inf")],
            "y": [float("nan"), 1.5, 2.5, 3.5],
        }
    ).write_parquet(right / "edges.parquet")

    summary_result = runner.invoke(app, ["summary", str(left), "--profile", "deep"])
    assert summary_result.exit_code == 0
    assert "Traceback" not in summary_result.stdout

    diff_result = runner.invoke(app, ["diff", str(left), str(right), "--level", "full"])
    assert diff_result.exit_code == 0
    assert "Traceback" not in diff_result.stdout

    compile_result = runner.invoke(
        app,
        ["compile", str(left), "--output", str(tmp_path / "compiled_edges")],
    )
    assert compile_result.exit_code == 0
    assert "Traceback" not in compile_result.stdout
    assert (tmp_path / "compiled_edges" / "metadata.json").exists()


def test_summary_and_schema_missing_source_fail_without_traceback(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing_source"

    summary_result = runner.invoke(app, ["summary", str(missing)])
    schema_result = runner.invoke(app, ["schema", str(missing)])

    assert summary_result.exit_code == 1
    assert "Source does not exist" in summary_result.stdout
    assert "Traceback" not in summary_result.stdout

    assert schema_result.exit_code == 1
    assert "Source does not exist" in schema_result.stdout
    assert "Traceback" not in schema_result.stdout


def test_plot_line_renders_for_numeric_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "series.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y": 10.0},
            {"x": 2, "y": 12.5},
            {"x": 3, "y": 15.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y", "--kind", "line"])

    assert result.exit_code == 0
    assert "Plot" in result.stdout
    assert "y" in result.stdout
    assert "Kind" in result.stdout
    assert "line" in result.stdout


def test_plot_multiple_y_columns_renders_stacked_sections(tmp_path: Path) -> None:
    csv_file = tmp_path / "multi_series.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y_a": 10.0, "y_b": 100.0},
            {"x": 2, "y_a": 12.5, "y_b": 110.0},
            {"x": 3, "y_a": 15.0, "y_b": 120.0},
        ],
    )

    result = runner.invoke(
        app,
        ["plot", str(csv_file), "x", "y_a", "y_b", "--kind", "line"],
    )

    assert result.exit_code == 0
    assert "Plot" in result.stdout
    assert "Field" in result.stdout
    assert "Plotted rows" in result.stdout
    assert "y_a" in result.stdout
    assert "y_b" in result.stdout


def test_plot_line_uses_nearest_interpolation_when_requested(tmp_path: Path) -> None:
    csv_file = tmp_path / "series.csv"
    _write_csv(
        csv_file,
        [
            {"x": 0, "y": 0.0},
            {"x": 10, "y": 10.0},
            {"x": 11, "y": 30.0},
        ],
    )

    result = runner.invoke(
        app,
        [
            "plot",
            str(csv_file),
            "x",
            "y",
            "--kind",
            "line",
            "--interpolation",
            "nearest",
        ],
    )

    assert result.exit_code == 0
    assert "Interpolation" in result.stdout
    assert "nearest" in result.stdout


def test_plot_line_fails_on_duplicate_x_without_aggregation(tmp_path: Path) -> None:
    csv_file = tmp_path / "duplicates.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y": 10.0},
            {"x": 1, "y": 12.0},
            {"x": 2, "y": 15.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y", "--kind", "line"])

    assert result.exit_code == 1
    assert "Duplicate x-values detected" in result.stdout
    assert "Traceback" not in result.stdout


def test_plot_line_aggregates_duplicate_x_with_requested_strategy(
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / "duplicates.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y": 10.0},
            {"x": 1, "y": 12.0},
            {"x": 2, "y": 15.0},
        ],
    )

    result = runner.invoke(
        app,
        [
            "plot",
            str(csv_file),
            "x",
            "y",
            "--kind",
            "line",
            "--aggregate-x",
            "mean",
        ],
    )

    assert result.exit_code == 0
    assert "Aggregated" in result.stdout
    assert "yes" in result.stdout


def test_plot_line_auto_sorts_unsorted_x_by_default(tmp_path: Path) -> None:
    csv_file = tmp_path / "unsorted.csv"
    _write_csv(
        csv_file,
        [
            {"x": 3, "y": 30.0},
            {"x": 1, "y": 10.0},
            {"x": 2, "y": 20.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y", "--kind", "line"])

    assert result.exit_code == 0
    assert "Auto-sorted" in result.stdout
    assert "x-values were auto-sorted" in result.stdout


def test_plot_line_fails_for_unsorted_x_when_strict_option_is_set(
    tmp_path: Path,
) -> None:
    csv_file = tmp_path / "unsorted.csv"
    _write_csv(
        csv_file,
        [
            {"x": 3, "y": 30.0},
            {"x": 1, "y": 10.0},
            {"x": 2, "y": 20.0},
        ],
    )

    result = runner.invoke(
        app,
        [
            "plot",
            str(csv_file),
            "x",
            "y",
            "--kind",
            "line",
            "--fail-on-x-unsorted",
        ],
    )

    assert result.exit_code == 1
    assert "x-values are not sorted" in result.stdout
    assert "Traceback" not in result.stdout


def test_plot_scatter_encodes_categorical_x_values(tmp_path: Path) -> None:
    csv_file = tmp_path / "categories.csv"
    _write_csv(
        csv_file,
        [
            {"category": "A", "y": 10.0},
            {"category": "B", "y": 12.0},
            {"category": "A", "y": 8.0},
        ],
    )

    result = runner.invoke(
        app,
        ["plot", str(csv_file), "category", "y", "--kind", "scatter"],
    )

    assert result.exit_code == 0
    assert "scatter" in result.stdout
    assert "encoded categorical x-values" in result.stdout


def test_plot_fails_when_column_is_missing(tmp_path: Path) -> None:
    csv_file = tmp_path / "series.csv"
    _write_csv(csv_file, [{"x": 1, "y": 10.0}])

    result = runner.invoke(app, ["plot", str(csv_file), "missing", "y"])

    assert result.exit_code == 1
    assert "Column not found" in result.stdout


def test_plot_auto_selects_line_for_sorted_numeric_x(tmp_path: Path) -> None:
    csv_file = tmp_path / "auto_line.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y": 10.0},
            {"x": 2, "y": 12.0},
            {"x": 3, "y": 14.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y"])

    assert result.exit_code == 0
    assert "Kind" in result.stdout
    assert "line" in result.stdout


def test_plot_auto_selects_scatter_for_unsorted_numeric_x(tmp_path: Path) -> None:
    csv_file = tmp_path / "auto_scatter_unsorted.csv"
    _write_csv(
        csv_file,
        [
            {"x": 3, "y": 30.0},
            {"x": 1, "y": 10.0},
            {"x": 2, "y": 20.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y"])

    assert result.exit_code == 0
    assert "scatter" in result.stdout
    assert "not monotonic" in result.stdout


def test_plot_auto_selects_scatter_for_duplicate_numeric_x(tmp_path: Path) -> None:
    csv_file = tmp_path / "auto_scatter_duplicates.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y": 10.0},
            {"x": 1, "y": 12.0},
            {"x": 2, "y": 20.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y"])

    assert result.exit_code == 0
    assert "scatter" in result.stdout
    assert "contain duplicates" in result.stdout


def test_plot_auto_selects_scatter_for_duplicate_temporal_x(tmp_path: Path) -> None:
    parquet_file = tmp_path / "auto_scatter_temporal_duplicates.parquet"
    pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 1, 0, 1, 0),
            ],
            "y": [10.0, 12.0, 20.0],
        }
    ).write_parquet(parquet_file)

    result = runner.invoke(app, ["plot", str(parquet_file), "ts", "y"])

    assert result.exit_code == 0
    assert "scatter" in result.stdout
    assert "temporal x-values contain duplicates" in result.stdout


def test_plot_filter_prefilters_rows_for_line_plot(tmp_path: Path) -> None:
    csv_file = tmp_path / "filtered_series.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y": 10.0, "current": 0.2},
            {"x": 2, "y": 20.0, "current": 0.6},
            {"x": 3, "y": 30.0, "current": 0.8},
        ],
    )

    result = runner.invoke(
        app,
        [
            "plot",
            str(csv_file),
            "x",
            "y",
            "--kind",
            "line",
            "--filter",
            "current>0.5",
        ],
    )

    assert result.exit_code == 0
    assert "Plot" in result.stdout
    assert "Plotted rows" in result.stdout
    plotted_rows_line = next(
        line for line in result.stdout.splitlines() if "Plotted rows" in line
    )
    assert "2" in plotted_rows_line


def test_plot_filter_applies_to_multiple_y_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "filtered_multi_series.csv"
    _write_csv(
        csv_file,
        [
            {"x": 1, "y_a": 10.0, "y_b": 100.0, "keep": 0},
            {"x": 1, "y_a": 11.0, "y_b": 101.0, "keep": 1},
            {"x": 2, "y_a": 12.0, "y_b": 102.0, "keep": 1},
        ],
    )

    result = runner.invoke(
        app,
        [
            "plot",
            str(csv_file),
            "x",
            "y_a",
            "y_b",
            "--kind",
            "line",
            "--filter",
            "keep==1",
        ],
    )

    assert result.exit_code == 0
    assert "y_a" in result.stdout
    assert "y_b" in result.stdout
    assert "Duplicate x-values detected" not in result.stdout


def test_plot_filter_rejects_invalid_syntax(tmp_path: Path) -> None:
    csv_file = tmp_path / "series.csv"
    _write_csv(csv_file, [{"x": 1, "y": 10.0, "current": 0.1}])

    result = runner.invoke(
        app,
        [
            "plot",
            str(csv_file),
            "x",
            "y",
            "--filter",
            "current=>0.5",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid --filter expression" in result.stdout


def test_plot_line_rejects_categorical_x_values(tmp_path: Path) -> None:
    csv_file = tmp_path / "categorical_line.csv"
    _write_csv(
        csv_file,
        [
            {"x": "A", "y": 10.0},
            {"x": "B", "y": 12.0},
            {"x": "C", "y": 14.0},
        ],
    )

    result = runner.invoke(app, ["plot", str(csv_file), "x", "y", "--kind", "line"])

    assert result.exit_code == 1
    assert "Line plots require numeric or temporal x-values" in result.stdout
    assert "Traceback" not in result.stdout


def test_plot_line_drops_non_plottable_numeric_x_rows_with_warning(
    tmp_path: Path,
) -> None:
    parquet_file = tmp_path / "numeric_x_edges.parquet"
    pl.DataFrame(
        {
            "x": [1.0, float("nan"), 2.0, float("inf")],
            "y": [10.0, 11.0, 12.0, 13.0],
        }
    ).write_parquet(parquet_file)

    result = runner.invoke(
        app,
        ["plot", str(parquet_file), "x", "y", "--kind", "line"],
    )

    assert result.exit_code == 0
    assert "Line plot dropped 2 rows with non-plottable x-values" in result.stdout
    assert "Line plots require numeric or temporal x-values" not in result.stdout


def test_plot_folder_renders_stacked_per_file_plots(tmp_path: Path) -> None:
    folder = tmp_path / "plots"
    folder.mkdir()
    _write_csv(folder / "a.csv", [{"x": 1, "y": 10.0}, {"x": 2, "y": 12.0}])
    _write_csv(folder / "b.csv", [{"x": 1, "y": 20.0}, {"x": 2, "y": 22.0}])

    result = runner.invoke(app, ["plot", str(folder), "x", "y", "--kind", "line"])

    assert result.exit_code == 0
    assert "Folder Plot Summary" in result.stdout
    assert "File 1: a.csv" in result.stdout
    assert "File 2: b.csv" in result.stdout


def test_plot_folder_limits_default_number_of_files(tmp_path: Path) -> None:
    folder = tmp_path / "many_plots"
    folder.mkdir()
    for index in range(7):
        _write_csv(
            folder / f"{index:02d}.csv",
            [{"x": 1, "y": float(index)}, {"x": 2, "y": float(index + 1)}],
        )

    result = runner.invoke(app, ["plot", str(folder), "x", "y", "--kind", "line"])

    assert result.exit_code == 0
    assert "File 5:" in result.stdout
    assert "File 6:" not in result.stdout
    assert "Folder contains 7 files; plotted first 5." in result.stdout


def test_plot_folder_respects_custom_file_limit(tmp_path: Path) -> None:
    folder = tmp_path / "many_plots"
    folder.mkdir()
    for index in range(7):
        _write_csv(
            folder / f"{index:02d}.csv",
            [{"x": 1, "y": float(index)}, {"x": 2, "y": float(index + 1)}],
        )

    result = runner.invoke(
        app,
        [
            "plot",
            str(folder),
            "x",
            "y",
            "--kind",
            "line",
            "-n",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "File 7:" in result.stdout
    assert "plotted first 5" not in result.stdout


def test_plot_folder_limit_rejects_zero(tmp_path: Path) -> None:
    folder = tmp_path / "plots"
    folder.mkdir()
    _write_csv(folder / "a.csv", [{"x": 1, "y": 10.0}])

    result = runner.invoke(
        app,
        [
            "plot",
            str(folder),
            "x",
            "y",
            "-n",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "-n must be greater than or equal to 1." in result.stdout
    assert "Traceback" not in result.stdout


def test_transform_user_script_failures_fail_without_traceback(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}])

    missing_dep_script = tmp_path / "missing_dep_transform.py"
    missing_dep_script.write_text(
        "import definitely_not_installed_pkg\n\ndef transform(df):\n    return df\n",
        encoding="utf-8",
    )

    runtime_script = tmp_path / "runtime_transform.py"
    runtime_script.write_text(
        "def transform(df):\n    return 1 / 0\n",
        encoding="utf-8",
    )

    missing_dep_result = runner.invoke(
        app,
        [
            "transform",
            str(data),
            str(missing_dep_script),
            str(tmp_path / "out_missing_dep"),
        ],
    )
    runtime_result = runner.invoke(
        app,
        ["transform", str(data), str(runtime_script), str(tmp_path / "out_runtime")],
    )

    assert missing_dep_result.exit_code == 1
    assert "definitely_not_installed_pkg" in missing_dep_result.stdout
    assert "Traceback" not in missing_dep_result.stdout

    assert runtime_result.exit_code == 1
    assert "division by zero" in runtime_result.stdout
    assert "Traceback" not in runtime_result.stdout
