from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tabcaddy.merge import (
    MergePlanner,
    MergeValidator,
    PlannedOperation,
    PreparedOperation,
)
from tabcaddy.test_support.synthetic_assets import (
    generate_synthetic_test_assets,
)


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


def _operation_by_source(
    operations: list[PlannedOperation],
    relative_source: str,
    source_root: Path,
) -> PlannedOperation:
    return next(
        operation
        for operation in operations
        if operation.source == source_root / Path(relative_source)
    )


def _prepared_by_source(
    operations: list[PreparedOperation],
    relative_source: str,
    source_root: Path,
) -> PreparedOperation:
    return next(
        operation
        for operation in operations
        if operation.source == source_root / Path(relative_source)
    )


def test_planner_rejects_ambiguous_file_to_folder_match(tmp_path: Path) -> None:
    planner = MergePlanner()
    source = tmp_path / "sales.csv"
    target_dir = tmp_path / "archive"

    _write_frame(source, [{"id": 2, "value": 20}])
    _write_frame(target_dir / "eu" / "sales.csv", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "us" / "sales.csv", [{"id": 3, "value": 30}])

    with pytest.raises(ValueError, match="Ambiguous target match detected"):
        planner.plan(
            source=source,
            target=target_dir,
            out=tmp_path / "merged.csv",
            inplace=False,
            ignore_filetype=False,
        )


def test_planner_file_to_folder_treats_nonexistent_dotted_out_as_directory(
    tmp_path: Path,
) -> None:
    planner = MergePlanner()
    source = tmp_path / "sales.csv"
    target_dir = tmp_path / "archive"
    out_dir = tmp_path / "combined.v2"

    _write_frame(source, [{"id": 2, "value": 20}])
    _write_frame(target_dir / "sales.csv", [{"id": 1, "value": 10}])

    operations = planner.plan(
        source=source,
        target=target_dir,
        out=out_dir,
        inplace=False,
        ignore_filetype=False,
    )

    assert operations == [
        PlannedOperation(
            source=source,
            target=target_dir / "sales.csv",
            destination=out_dir / "sales.csv",
            output_directory=True,
            kind="merge",
        )
    ]


def test_planner_folder_to_folder_inplace_routes_new_files_into_target_tree(
    tmp_path: Path,
) -> None:
    planner = MergePlanner()
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"

    _write_frame(source_dir / "eu" / "new.csv", [{"id": 2, "value": 20}])
    _write_frame(target_dir / "us" / "existing.csv", [{"id": 1, "value": 10}])

    operations = planner.plan(
        source=source_dir,
        target=target_dir,
        out=None,
        inplace=True,
        ignore_filetype=False,
    )

    assert operations == [
        PlannedOperation(
            source=source_dir / "eu" / "new.csv",
            target=None,
            destination=target_dir / "eu" / "new.csv",
            output_directory=True,
            kind="source_only",
        )
    ]


def test_planner_folder_to_folder_non_inplace_includes_target_passthrough(
    tmp_path: Path,
) -> None:
    planner = MergePlanner()
    source_dir = tmp_path / "incoming"
    target_dir = tmp_path / "archive"
    out_dir = tmp_path / "combined"

    _write_frame(source_dir / "sales.csv", [{"id": 2, "value": 20}])
    _write_frame(target_dir / "sales.csv", [{"id": 1, "value": 10}])
    _write_frame(target_dir / "target_only.csv", [{"id": 8, "value": 80}])

    operations = planner.plan(
        source=source_dir,
        target=target_dir,
        out=out_dir,
        inplace=False,
        ignore_filetype=False,
    )

    assert operations == [
        PlannedOperation(
            source=source_dir / "sales.csv",
            target=target_dir / "sales.csv",
            destination=out_dir / "sales.csv",
            output_directory=True,
            kind="merge",
        ),
        PlannedOperation(
            source=target_dir / "target_only.csv",
            target=None,
            destination=out_dir / "target_only.csv",
            output_directory=True,
            kind="target_passthrough",
        ),
    ]


def test_planner_and_validator_prepare_realistic_synthetic_merge_plan(
    tmp_path: Path,
) -> None:
    planner = MergePlanner()
    validator = MergeValidator()
    layout = generate_synthetic_test_assets(tmp_path / "synthetic_seed", n=6)
    out_dir = tmp_path / "merged"

    operations = planner.plan(
        source=layout.variant_root,
        target=layout.baseline_root,
        out=out_dir,
        inplace=False,
        ignore_filetype=False,
    )
    prepared = validator.prepare_operations(
        operations=operations,
        out=out_dir,
        inplace=False,
        on_columns=(),
        ignore_filetype=False,
    )

    source_files = sorted(
        path.relative_to(layout.variant_root).as_posix()
        for path in layout.variant_root.rglob("*")
        if path.is_file()
    )

    assert len(operations) == len(source_files)
    assert len(prepared) == len(source_files)

    extra_variant = _operation_by_source(
        operations,
        "telemetry/mcu/100250-100999_SUDS.feather",
        layout.variant_root,
    )
    assert extra_variant.target is None
    assert extra_variant.destination == (
        out_dir / "telemetry" / "mcu" / "100250-100999_SUDS.feather"
    )
    assert extra_variant.output_directory is True
    assert extra_variant.kind == "source_only"

    service_products = _prepared_by_source(
        prepared,
        "service/PRED_MAINT_ProductsConsumed.csv",
        layout.variant_root,
    )
    assert service_products.target == (
        layout.baseline_root / "service" / "PRED_MAINT_ProductsConsumed.csv"
    )
    assert service_products.validation is not None
    assert service_products.validation.conflicting_columns == []
    assert service_products.validation.cast_source_to_target_schema is False


def test_validator_rejects_duplicate_destinations(tmp_path: Path) -> None:
    validator = MergeValidator()
    source_a = tmp_path / "incoming" / "a.csv"
    source_b = tmp_path / "incoming" / "b.csv"
    destination = tmp_path / "out" / "merged.csv"

    _write_frame(source_a, [{"id": 1, "value": 10}])
    _write_frame(source_b, [{"id": 2, "value": 20}])

    operations = [
        PlannedOperation(
            source=source_a,
            target=None,
            destination=destination,
            output_directory=True,
        ),
        PlannedOperation(
            source=source_b,
            target=None,
            destination=destination,
            output_directory=True,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Multiple source files resolve to the same destination",
    ):
        validator.prepare_operations(
            operations=operations,
            out=tmp_path / "out",
            inplace=False,
            on_columns=(),
            ignore_filetype=False,
        )


def test_validator_rejects_existing_output_when_not_inplace(tmp_path: Path) -> None:
    validator = MergeValidator()
    source = tmp_path / "incoming.csv"
    destination = tmp_path / "merged.csv"

    _write_frame(source, [{"id": 1, "value": 10}])
    _write_frame(destination, [{"id": 0, "value": 0}])

    with pytest.raises(FileExistsError, match="Output path already exists"):
        validator.prepare_operations(
            operations=[
                PlannedOperation(
                    source=source,
                    target=None,
                    destination=destination,
                    output_directory=False,
                )
            ],
            out=destination,
            inplace=False,
            on_columns=(),
            ignore_filetype=False,
        )


def test_validator_passthrough_operation_skips_schema_validation(
    tmp_path: Path,
) -> None:
    validator = MergeValidator()
    source = tmp_path / "incoming.csv"
    destination = tmp_path / "out" / "incoming.csv"

    _write_frame(source, [{"id": 1, "value": 10}])

    prepared = validator.prepare_operations(
        operations=[
            PlannedOperation(
                source=source,
                target=None,
                destination=destination,
                output_directory=True,
            )
        ],
        out=tmp_path / "out",
        inplace=False,
        on_columns=("id",),
        ignore_filetype=False,
    )

    assert prepared == [
        PreparedOperation(source=source, target=None, destination=destination)
    ]


def test_validator_rejects_layout_mismatch_before_column_type_checks(
    tmp_path: Path,
) -> None:
    validator = MergeValidator()
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"

    _write_frame(source, [{"id": 1, "value": 10}])
    _write_frame(target, [{"id": 1, "other": 10}])

    with pytest.raises(ValueError, match="column layouts differ"):
        validator.prepare_operations(
            operations=[
                PlannedOperation(
                    source=source,
                    target=target,
                    destination=tmp_path / "merged.csv",
                    output_directory=False,
                )
            ],
            out=tmp_path / "merged.csv",
            inplace=False,
            on_columns=("id",),
            ignore_filetype=False,
        )


def test_validator_rejects_missing_merge_key_columns(tmp_path: Path) -> None:
    validator = MergeValidator()
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"

    _write_frame(source, [{"id": 1, "value": 10}])
    _write_frame(target, [{"id": 2, "value": 20}])

    with pytest.raises(ValueError, match="Merge key columns not found in both files"):
        validator.prepare_operations(
            operations=[
                PlannedOperation(
                    source=source,
                    target=target,
                    destination=tmp_path / "merged.csv",
                    output_directory=False,
                )
            ],
            out=tmp_path / "merged.csv",
            inplace=False,
            on_columns=("id", "missing"),
            ignore_filetype=False,
        )


def test_validator_rejects_incompatible_column_types_without_casting(
    tmp_path: Path,
) -> None:
    validator = MergeValidator()
    source = tmp_path / "source.csv"
    target = tmp_path / "target.csv"

    _write_frame(source, [{"id": 1, "value": "ten"}])
    _write_frame(target, [{"id": 1, "value": 10}])

    with pytest.raises(ValueError, match="incompatible types for value"):
        validator.prepare_operations(
            operations=[
                PlannedOperation(
                    source=source,
                    target=target,
                    destination=tmp_path / "merged.csv",
                    output_directory=False,
                )
            ],
            out=tmp_path / "merged.csv",
            inplace=False,
            on_columns=("id",),
            ignore_filetype=False,
        )


def test_validator_allows_csv_to_binary_cast_when_ignore_filetype_enabled(
    tmp_path: Path,
) -> None:
    validator = MergeValidator()
    source = tmp_path / "source.csv"
    target = tmp_path / "target.parquet"
    destination = tmp_path / "merged.parquet"

    _write_frame(source, [{"id": "1", "value": "10"}])
    _write_frame(target, [{"id": 2, "value": 20}])

    prepared = validator.prepare_operations(
        operations=[
            PlannedOperation(
                source=source,
                target=target,
                destination=destination,
                output_directory=False,
            )
        ],
        out=destination,
        inplace=False,
        on_columns=("id",),
        ignore_filetype=True,
    )

    assert len(prepared) == 1
    assert prepared[0].validation is not None
    assert prepared[0].validation.cast_source_to_target_schema is True
    assert prepared[0].validation.conflicting_columns == []
    assert prepared[0].validation.target_schema == pl.Schema(
        {"id": pl.Int64, "value": pl.Int64}
    )
