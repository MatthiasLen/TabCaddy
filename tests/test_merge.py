from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from tabcaddy.application.generate_synthetic_test_assets import (
    generate_synthetic_test_assets,
)
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


def _write_compiled_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    data_path = path / "data"
    data_path.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(data_path / "part-001.parquet")
    (path / "metadata.json").write_text("{}", encoding="utf-8")


def _copy_tree(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    return destination


def _update_row(frame: pl.DataFrame, row_index: int, **updates: object) -> pl.DataFrame:
    updated = frame.with_row_index("_row_index")
    for column, value in updates.items():
        updated = updated.with_columns(
            pl.when(pl.col("_row_index") == row_index)
            .then(pl.lit(value, dtype=updated.schema[column]))
            .otherwise(pl.col(column))
            .alias(column)
        )
    return updated.drop("_row_index")


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


def test_merge_rejects_compiled_dataset_source(tmp_path: Path) -> None:
    source = tmp_path / "compiled_source"
    target = tmp_path / "target.csv"
    output = tmp_path / "merged.csv"

    _write_compiled_dataset(source, [{"id": 1, "value": 10}])
    _write_frame(target, [{"id": 2, "value": 20}])

    result = runner.invoke(
        app, ["merge", str(source), str(target), "--out", str(output)]
    )

    assert result.exit_code == 1
    assert "Merge does not support compiled datasets for source" in result.stdout
    assert not output.exists()


def test_merge_rejects_compiled_dataset_target_inplace(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "compiled_target"

    _write_frame(source, [{"id": 1, "value": 10}])
    _write_compiled_dataset(target, [{"id": 2, "value": 20}])

    result = runner.invoke(app, ["merge", str(source), str(target), "--inplace"])

    assert result.exit_code == 1
    assert "Merge does not support compiled datasets for target" in result.stdout


def test_merge_folder_conflict_fails_without_writing_any_output(tmp_path: Path) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_dir = tmp_path / "combined"

    _write_frame(source_dir / "a_ok.csv", [{"id": 2, "value": 20}])
    _write_frame(source_dir / "b_conflict.csv", [{"id": 1, "value": 99}])
    _write_frame(target_dir / "a_ok.csv", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "b_conflict.csv", [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        [
            "merge",
            str(source_dir),
            str(target_dir),
            "--out",
            str(output_dir),
            "--on",
            "id",
        ],
    )

    assert result.exit_code == 1
    assert "Conflicting duplicate key" in result.stdout
    assert not output_dir.exists()


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


def test_merge_folder_inplace_cast_failure_rolls_back_all_changes(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"

    _write_frame(source_dir / "a_ok.csv", [{"id": 1, "value": 11}])
    _write_frame(source_dir / "b_new.csv", [{"id": 9, "value": 90}])
    _write_frame(source_dir / "c_bad.csv", [{"id": 2, "value": "oops"}])
    _write_frame(target_dir / "a_ok.parquet", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "c_bad.parquet", [{"id": 2, "value": 20}])

    result = runner.invoke(
        app,
        [
            "merge",
            str(source_dir),
            str(target_dir),
            "--inplace",
            "--ignore-filetype",
        ],
    )

    assert result.exit_code == 1
    assert _read_frame(target_dir / "a_ok.parquet").to_dicts() == [
        {"id": 1, "value": 10}
    ]
    assert _read_frame(target_dir / "c_bad.parquet").to_dicts() == [
        {"id": 2, "value": 20}
    ]
    assert not (target_dir / "b_new.csv").exists()


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
    assert f"Merged 2 files into {output_dir.resolve()}" in result.stdout


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


def test_merge_file_into_folder_accepts_nonexistent_dotted_output_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sales.csv"
    archive = tmp_path / "archive"
    output_dir = tmp_path / "combined.v2"

    archive.mkdir()
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


def test_merge_dry_run_previews_matches_passthrough_casts_and_conflicts(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    output_dir = tmp_path / "combined"

    _write_frame(source_dir / "sales.csv", [{"id": "2", "value": "20"}])
    _write_frame(source_dir / "new.csv", [{"id": 9, "value": 90}])
    _write_frame(source_dir / "conflict.csv", [{"id": 1, "value": 99}])
    _write_frame(target_dir / "sales.parquet", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "target_only.csv", [{"id": 8, "value": 80}])
    _write_frame(target_dir / "conflict.csv", [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        [
            "merge",
            str(source_dir),
            str(target_dir),
            "--out",
            str(output_dir),
            "--on",
            "id",
            "--ignore-filetype",
            "--dry",
        ],
    )

    assert result.exit_code == 1
    assert "Dry-run merge plan" in result.stdout
    assert "MERGE" in result.stdout
    assert "SOURCE_ONLY" in result.stdout
    assert "TARGET_PASSTHROUGH" in result.stdout
    assert "sales.csv" in result.stdout
    assert "sales.parquet" in result.stdout
    assert "target_only.csv" in result.stdout
    assert f"destination={output_dir / 'sales.parquet'}" in result.stdout
    assert "cast=.csv->.parquet" in result.stdout
    assert "Conflicting duplicate key detected" in result.stdout
    assert not output_dir.exists()
    assert _read_frame(target_dir / "conflict.csv").to_dicts() == [
        {"id": 1, "value": 10}
    ]


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


def test_merge_synthetic_folder_to_folder_out_handles_union_dedup_and_passthrough(
    tmp_path: Path,
) -> None:
    layout = generate_synthetic_test_assets(tmp_path / "synthetic_seed", n=7)
    source_dir = _copy_tree(layout.baseline_root, tmp_path / "incoming")
    target_dir = _copy_tree(layout.baseline_root, tmp_path / "archive")
    output_dir = tmp_path / "merged"

    source_products_path = source_dir / "service" / "PRED_MAINT_ProductsConsumed.csv"
    baseline_products = _read_frame(source_products_path)
    modified_products = _update_row(
        baseline_products,
        2,
        PRODUCT_DESCRIPTION="Synthetic pressure-test replacement",
    )
    extra_product = baseline_products.tail(1).with_columns(
        [
            pl.lit("W-999999999").alias("WORK_ORDER"),
            pl.lit("PC-99999").alias("PRODUCT_CONSUMED_ID"),
            pl.lit("Pressure merge edge case").alias("PRODUCT_DESCRIPTION"),
        ]
    )
    _write_frame(
        source_products_path,
        pl.concat([modified_products, extra_product], how="vertical").to_dicts(),
    )

    source_only_frame = _read_frame(
        source_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.feather"
    )
    _write_frame(
        source_dir / "telemetry" / "mcu" / "200000-200100_SUDS.feather",
        source_only_frame.to_dicts(),
    )
    _write_frame(
        target_dir / "service" / "target_only.csv",
        [{"id": 1, "note": "preserve-existing-target"}],
    )

    result = runner.invoke(
        app,
        ["merge", str(source_dir), str(target_dir), "--out", str(output_dir)],
    )

    assert result.exit_code == 0
    merged_products = _read_frame(
        output_dir / "service" / "PRED_MAINT_ProductsConsumed.csv"
    )
    assert merged_products.height == baseline_products.height + 2
    assert merged_products.filter(pl.col("WORK_ORDER") == "W-999999999").height == 1
    assert (
        merged_products.filter(
            pl.col("PRODUCT_DESCRIPTION") == "Synthetic pressure-test replacement"
        ).height
        == 1
    )

    unchanged_version = _read_frame(
        output_dir / "telemetry" / "version" / "1000001-10000_SUDS.feather"
    )
    assert (
        unchanged_version.height
        == _read_frame(
            target_dir / "telemetry" / "version" / "1000001-10000_SUDS.feather"
        ).height
    )
    assert (
        _read_frame(
            output_dir / "telemetry" / "mcu" / "200000-200100_SUDS.feather"
        ).to_dicts()
        == source_only_frame.to_dicts()
    )
    assert _read_frame(output_dir / "service" / "target_only.csv").to_dicts() == [
        {"id": 1, "note": "preserve-existing-target"}
    ]


def test_merge_synthetic_folder_inplace_rolls_back_everything_on_key_conflict(
    tmp_path: Path,
) -> None:
    layout = generate_synthetic_test_assets(tmp_path / "synthetic_seed", n=6)
    source_dir = _copy_tree(layout.baseline_root, tmp_path / "incoming")
    target_dir = _copy_tree(layout.baseline_root, tmp_path / "archive")

    target_products_path = target_dir / "service" / "PRED_MAINT_ProductsConsumed.csv"
    original_products = _read_frame(target_products_path)
    source_products_path = source_dir / "service" / "PRED_MAINT_ProductsConsumed.csv"
    conflicted_products = _update_row(
        _read_frame(source_products_path),
        1,
        PRODUCT_DESCRIPTION="Conflicting payload for identical date",
    )
    _write_frame(source_products_path, conflicted_products.to_dicts())

    source_only_path = source_dir / "telemetry" / "mcu" / "999999-100500_SUDS.feather"
    _write_frame(
        source_only_path,
        _read_frame(
            source_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.feather"
        ).to_dicts(),
    )

    result = runner.invoke(
        app,
        [
            "merge",
            str(source_dir),
            str(target_dir),
            "--inplace",
            "--on",
            "DATE",
        ],
    )

    assert result.exit_code == 1
    assert "Conflicting duplicate key detected" in result.stdout
    assert _read_frame(target_products_path).to_dicts() == original_products.to_dicts()
    assert not (target_dir / source_only_path.relative_to(source_dir)).exists()
    assert not (tmp_path / ".archive.tmp").exists()


def test_merge_synthetic_ignore_filetype_merges_nested_mixed_formats(
    tmp_path: Path,
) -> None:
    layout = generate_synthetic_test_assets(tmp_path / "synthetic_seed", n=5)
    source_dir = _copy_tree(layout.baseline_root, tmp_path / "incoming")
    target_dir = _copy_tree(layout.baseline_root, tmp_path / "archive")
    output_dir = tmp_path / "mixed_out"

    target_products_csv = target_dir / "service" / "PRED_MAINT_ProductsConsumed.csv"
    target_products_frame = _read_frame(target_products_csv)
    _write_frame(
        target_dir / "service" / "PRED_MAINT_ProductsConsumed.parquet",
        target_products_frame.to_dicts(),
    )
    target_products_csv.unlink()

    target_mcu_feather = target_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.feather"
    target_mcu_frame = _read_frame(target_mcu_feather)
    _write_frame(
        target_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.parquet",
        target_mcu_frame.to_dicts(),
    )
    target_mcu_feather.unlink()

    source_products_path = source_dir / "service" / "PRED_MAINT_ProductsConsumed.csv"
    source_products_frame = _read_frame(source_products_path)
    extra_product = source_products_frame.tail(1).with_columns(
        [
            pl.lit("W-777777777").alias("WORK_ORDER"),
            pl.lit("PC-77777").alias("PRODUCT_CONSUMED_ID"),
        ]
    )
    _write_frame(
        source_products_path,
        pl.concat([source_products_frame, extra_product], how="vertical").to_dicts(),
    )

    source_mcu_path = source_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.feather"
    source_mcu_frame = _read_frame(source_mcu_path)
    extra_mcu = source_mcu_frame.tail(1).with_columns(
        [
            pl.lit(999).cast(source_mcu_frame.schema["index"]).alias("index"),
            pl.lit(999999)
            .cast(source_mcu_frame.schema["DELTA_TIME"])
            .alias("DELTA_TIME"),
        ]
    )
    _write_frame(
        source_mcu_path,
        pl.concat([source_mcu_frame, extra_mcu], how="vertical").to_dicts(),
    )

    _write_frame(
        source_dir / "service" / "source_only.csv",
        [{"id": 9, "value": 90}],
    )

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
    merged_products = _read_frame(
        output_dir / "service" / "PRED_MAINT_ProductsConsumed.parquet"
    )
    assert merged_products.height == target_products_frame.height + 1
    target_products_parquet = _read_frame(
        target_dir / "service" / "PRED_MAINT_ProductsConsumed.parquet"
    )
    assert (
        merged_products.schema["WORK_ORDER"]
        == target_products_parquet.schema["WORK_ORDER"]
    )
    assert (
        merged_products.schema["PRODUCT_CONSUMED_ID"]
        == target_products_parquet.schema["PRODUCT_CONSUMED_ID"]
    )
    assert (
        merged_products.schema["SYSTEM_EQUNR"]
        == target_products_parquet.schema["SYSTEM_EQUNR"]
    )
    assert merged_products.filter(pl.col("WORK_ORDER") == "W-777777777").height == 1

    merged_mcu = _read_frame(
        output_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.parquet"
    )
    assert merged_mcu.height == target_mcu_frame.height + 1
    target_mcu_parquet = _read_frame(
        target_dir / "telemetry" / "mcu" / "1000001-10000_SUDS.parquet"
    )
    assert merged_mcu.schema == target_mcu_parquet.schema
    assert merged_mcu.filter(pl.col("index") == 999).height == 1
    assert _read_frame(output_dir / "service" / "source_only.csv").to_dicts() == [
        {"id": 9, "value": 90}
    ]


def test_merge_ignore_filetype_rejects_ambiguous_target_match_before_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sales.csv"
    target_dir = tmp_path / "archive"
    output = tmp_path / "merged.parquet"
    target_dir.mkdir()

    _write_frame(source, [{"id": 2, "value": 20}])
    _write_frame(target_dir / "sales.csv", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "sales.parquet", [{"id": 1, "value": 10}])

    result = runner.invoke(
        app,
        [
            "merge",
            str(source),
            str(target_dir),
            "--out",
            str(output),
            "--ignore-filetype",
        ],
    )

    assert result.exit_code == 1
    assert "Ambiguous target match detected" in result.stdout
    assert not output.exists()
