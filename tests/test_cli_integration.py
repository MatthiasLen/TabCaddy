from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tabcaddy.cli.app import app


runner = CliRunner()


def test_summary_command_runs(homogeneous_folder) -> None:
    result = runner.invoke(app, ["summary", str(homogeneous_folder)])
    assert result.exit_code == 0
    assert "Metadata" in result.stdout
    assert "Statistics" in result.stdout


def test_compile_transform_scaffold_and_diff_commands(
    tmp_path: Path, drift_folder, homogeneous_folder
) -> None:
    compiled = tmp_path / "compiled_dataset"
    compile_result = runner.invoke(
        app, ["compile", str(drift_folder), "--schema", "1", "--output", str(compiled)]
    )
    assert compile_result.exit_code == 0
    assert (compiled / "metadata.json").exists()
    assert (compiled / "data").is_dir()

    transform_script = tmp_path / "transform.py"
    transform_script.write_text(
        "import polars as pl\n\n"
        "def transform(df, context):\n"
        "    return df.with_columns((pl.col('value') * 2).alias('value'))\n",
        encoding="utf-8",
    )
    transformed = tmp_path / "transformed"
    transform_result = runner.invoke(
        app,
        ["transform", str(homogeneous_folder), str(transform_script), str(transformed)],
    )
    assert transform_result.exit_code == 0
    assert transformed.exists()

    scaffold_target = tmp_path / "template.py"
    scaffold_result = runner.invoke(
        app,
        [
            "scaffold-transform",
            str(homogeneous_folder),
            "--output",
            str(scaffold_target),
        ],
    )
    assert scaffold_result.exit_code == 0
    assert "def transform" in scaffold_target.read_text(encoding="utf-8")

    diff_result = runner.invoke(app, ["diff", str(compiled), str(compiled)])
    assert diff_result.exit_code == 0
    assert "No changes." in diff_result.stdout
