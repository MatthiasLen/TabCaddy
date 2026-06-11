from __future__ import annotations

from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from tabcaddy.cli.app import app


runner = CliRunner()


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


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
    assert "Statistics Changes" in diff_result.stdout


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
    assert "Metadata Changes" in diff_result.stdout
    assert "Schema Changes" not in diff_result.stdout
    assert "Statistics Changes" not in diff_result.stdout
