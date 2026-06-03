from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from tabcaddy.application.generate_synthetic_test_assets import (
    generate_synthetic_test_assets,
)
from tabcaddy.cli.app import app


runner = CliRunner()


def test_generated_assets_cover_all_cli_commands(tmp_path: Path) -> None:
    layout = generate_synthetic_test_assets(tmp_path / "synthetic_assets", n=12)
    baseline = layout.baseline_root
    variant = layout.variant_root

    summary_result = runner.invoke(app, ["summary", str(baseline), "--profile", "deep"])
    assert summary_result.exit_code == 0
    assert "Metadata" in summary_result.stdout
    assert "Statistics" in summary_result.stdout
    assert "Schema Distribution" in summary_result.stdout
    assert "Warnings" in summary_result.stdout

    schema_result = runner.invoke(app, ["schema", str(baseline)])
    assert schema_result.exit_code == 0
    assert "Schema Groups" in schema_result.stdout
    assert "Files Violating Dominant Schema" in schema_result.stdout

    compiled_left = tmp_path / "compiled_left"
    compiled_right = tmp_path / "compiled_right"
    compile_left_result = runner.invoke(
        app,
        ["compile", str(baseline), "--schema", "1", "--output", str(compiled_left)],
    )
    compile_right_result = runner.invoke(
        app,
        ["compile", str(variant), "--schema", "1", "--output", str(compiled_right)],
    )
    assert compile_left_result.exit_code == 0
    assert compile_right_result.exit_code == 0
    assert (compiled_left / "metadata.json").exists()
    assert (compiled_right / "metadata.json").exists()

    compiled_summary_result = runner.invoke(app, ["summary", str(compiled_left)])
    assert compiled_summary_result.exit_code == 0
    assert "Statistics" in compiled_summary_result.stdout

    scaffold_target = tmp_path / "transform_template.py"
    scaffold_result = runner.invoke(
        app,
        ["scaffold-transform", str(baseline), "--output", str(scaffold_target)],
    )
    assert scaffold_result.exit_code == 0
    scaffold_text = scaffold_target.read_text(encoding="utf-8")
    assert "# Schema 1:" in scaffold_text
    assert "PART_DESCRIPTION" in scaffold_text
    assert "PRODUCT_DESCRIPTION" in scaffold_text

    transform_path = tmp_path / "transform.py"
    transform_path.write_text(
        "import polars as pl\n\n"
        "def transform(df, context):\n"
        "    df = df.with_columns(pl.lit(context.file_name).alias('SOURCE_FILE'))\n"
        "    if 'VOLTAGE' in df.columns and 'CURRENT' in df.columns:\n"
        "        return df.with_columns((pl.col('VOLTAGE') - pl.col('CURRENT')).alias('POWER_GAP'))\n"
        "    if 'PART_DESCRIPTION' in df.columns:\n"
        "        return df.with_columns(pl.col('PART_DESCRIPTION').str.len_chars().alias('DESCRIPTION_LENGTH'))\n"
        "    if 'PRODUCT_DESCRIPTION' in df.columns:\n"
        "        return df.with_columns(pl.col('PRODUCT_DESCRIPTION').str.len_chars().alias('DESCRIPTION_LENGTH'))\n"
        "    return df\n",
        encoding="utf-8",
    )
    transformed_root = tmp_path / "transformed"
    transform_result = runner.invoke(
        app,
        [
            "transform",
            str(baseline),
            str(transform_path),
            str(transformed_root),
            "--workers",
            "2",
        ],
    )
    assert transform_result.exit_code == 0
    transformed_mcu = pl.read_ipc(
        transformed_root / "telemetry" / "mcu" / "1000001-10000_SUDS.feather"
    )
    transformed_parts = pl.read_csv(
        transformed_root / "service" / "PRED_MAINT_SWCalToolsParts.csv"
    )
    assert transformed_mcu.height == 12
    assert transformed_parts.height == 12
    assert "SOURCE_FILE" in transformed_mcu.columns
    assert "POWER_GAP" in transformed_mcu.columns
    assert "DESCRIPTION_LENGTH" in transformed_parts.columns

    folder_diff_result = runner.invoke(
        app, ["diff", str(baseline), str(variant), "--level", "full"]
    )
    assert folder_diff_result.exit_code == 0
    assert "Modified file:" in folder_diff_result.stdout
    assert (
        "source_file_count" in folder_diff_result.stdout.lower()
        or "source file count" in folder_diff_result.stdout.lower()
    )
    assert (
        ".mean:" in folder_diff_result.stdout
        or ".max_value:" in folder_diff_result.stdout
    )

    compiled_diff_result = runner.invoke(
        app, ["diff", str(compiled_left), str(compiled_right), "--level", "full"]
    )
    assert compiled_diff_result.exit_code == 0
    assert "Compiled dataset provenance changed" in compiled_diff_result.stdout
