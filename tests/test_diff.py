from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path

import polars as pl

from tabcaddy.analysis import GenerateAnalysis, resolve_source
from tabcaddy.diff import DiffDatasets, compare_analyses
from tabcaddy.domain.models import (
    ColumnStatistics,
    DatasetAnalysis,
    DatasetMetadata,
    DatasetSource,
    DiffComparisonType,
    DatasetStatistics,
    DiffLevel,
    DiffReport,
    DiffSummary,
    SchemaSignature,
    SourceType,
)
from tabcaddy.shared.serialization import analysis_to_dict, diff_report_to_dict


def _write_compiled_dataset(
    root: Path,
    *,
    created_at: datetime,
    compiled: dict[str, object],
) -> DatasetSource:
    root.mkdir()
    (root / "data").mkdir()
    analysis = DatasetAnalysis(
        metadata=DatasetMetadata(
            version=1,
            created_at=created_at,
            row_count=1,
            column_count=1,
            source_file_count=1,
            schema_hash="schema-1",
            column_hashes={"value": "hash-1"},
        ),
        schemas=[SchemaSignature(columns=[], hash="schema-1", occurrence_count=1)],
        statistics=DatasetStatistics(columns={}),
        warnings=[],
    )
    payload = analysis_to_dict(analysis)
    payload["compiled"] = compiled
    (root / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return DatasetSource(path=root, source_type=SourceType.COMPILED_DATASET)


def test_diff_reports_changes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).write_csv(left / "data.csv")
    pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 40.0]}).write_csv(
        right / "data.csv"
    )

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(left), resolve_source(right), DiffLevel.FULL
    )
    assert any(
        "Row Count" in change or "rows" in change.lower()
        for change in report.metadata_changes
    )
    assert any("value.mean" in change for change in report.statistics_changes)
    assert "Modified file: data.csv" in report.file_changes
    assert report.summary is not None
    assert report.summary.comparison_type.value == "folder_vs_folder"
    assert report.summary.modified_files == 1


def test_file_diff_summary_reports_content_status(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).write_csv(left)
    pl.DataFrame({"id": [1, 2], "value": [10.0, 21.0]}).write_csv(right)

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(left), resolve_source(right), DiffLevel.FULL
    )

    assert report.summary is not None
    assert report.summary.comparison_type.value == "file_vs_file"
    assert report.summary.content_status == "modified"


def test_mixed_diff_reports_missing_match(tmp_path: Path) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    folder.mkdir()
    pl.DataFrame({"id": [1], "value": [10.0]}).write_csv(source_file)
    pl.DataFrame({"id": [1], "value": [10.0]}).write_csv(folder / "other.csv")

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(source_file), resolve_source(folder), DiffLevel.FULL
    )

    assert report.summary is not None
    assert report.summary.comparison_type.value == "file_vs_folder"
    assert report.summary.match_status == "missing"


def test_mixed_diff_reports_unmodified_unique_match(tmp_path: Path) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    folder.mkdir()
    frame = pl.DataFrame({"id": [1], "value": [10.0]})
    frame.write_csv(source_file)
    frame.write_csv(folder / "data.csv")

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(source_file), resolve_source(folder), DiffLevel.FULL
    )

    assert report.summary is not None
    assert report.summary.match_status == "unmodified"
    assert report.summary.matched_path == "data.csv"


def test_mixed_diff_reports_ambiguous_match(tmp_path: Path) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    nested = folder / "nested"
    folder.mkdir()
    nested.mkdir()
    frame = pl.DataFrame({"id": [1], "value": [10.0]})
    frame.write_csv(source_file)
    frame.write_csv(folder / "data.csv")
    frame.write_csv(nested / "data.csv")

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(source_file), resolve_source(folder), DiffLevel.FULL
    )

    assert report.summary is not None
    assert report.summary.match_status == "ambiguous"
    assert report.summary.candidate_paths == ["data.csv", "nested/data.csv"]


def test_mixed_diff_reports_modified_unique_match_with_statistics(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    folder.mkdir()
    pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]}).write_csv(source_file)
    pl.DataFrame({"id": [1, 2, 3], "value": [10.0, 21.0, 30.0]}).write_csv(
        folder / "data.csv"
    )

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(source_file), resolve_source(folder), DiffLevel.FULL
    )

    assert report.summary is not None
    assert report.summary.match_status == "modified"
    assert any("Row Count" in change for change in report.metadata_changes)
    assert any("value.mean" in change for change in report.statistics_changes)


def test_mixed_diff_uses_exact_content_match_to_break_basename_ties(
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "data.csv"
    folder = tmp_path / "folder"
    nested = folder / "nested"
    folder.mkdir()
    nested.mkdir()
    source_frame = pl.DataFrame({"id": [1], "value": [10.0]})
    source_frame.write_csv(source_file)
    pl.DataFrame({"id": [1], "value": [99.0]}).write_csv(folder / "data.csv")
    source_frame.write_csv(nested / "data.csv")

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(source_file), resolve_source(folder), DiffLevel.FULL
    )

    assert report.summary is not None
    assert report.summary.match_status == "unmodified"
    assert report.summary.matched_path == "nested/data.csv"


def test_compiled_diff_ignores_regenerated_created_at_when_provenance_matches(
    tmp_path: Path,
) -> None:
    compiled = {
        "source": "C:/dataset",
        "selected_schema_hash": "schema-1",
        "written_parts": ["data/part-000.parquet"],
    }
    left = _write_compiled_dataset(
        tmp_path / "left_compiled",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        compiled=compiled,
    )
    right = _write_compiled_dataset(
        tmp_path / "right_compiled",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        compiled=compiled,
    )

    report = DiffDatasets(GenerateAnalysis()).run(left, right, DiffLevel.FULL)

    assert "Compiled dataset provenance changed" not in report.metadata_changes


def test_compiled_diff_reports_changed_compiled_provenance(tmp_path: Path) -> None:
    left = _write_compiled_dataset(
        tmp_path / "left_compiled",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        compiled={
            "source": "C:/dataset-a",
            "selected_schema_hash": "schema-1",
            "written_parts": ["data/part-000.parquet"],
        },
    )
    right = _write_compiled_dataset(
        tmp_path / "right_compiled",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        compiled={
            "source": "C:/dataset-b",
            "selected_schema_hash": "schema-1",
            "written_parts": ["data/part-000.parquet"],
        },
    )

    report = DiffDatasets(GenerateAnalysis()).run(left, right, DiffLevel.FULL)

    assert "Compiled dataset provenance changed" in report.metadata_changes


def test_compiled_diff_treats_malformed_metadata_as_missing_provenance(
    tmp_path: Path,
) -> None:
    left = _write_compiled_dataset(
        tmp_path / "left_compiled",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        compiled={
            "source": "C:/dataset-a",
            "selected_schema_hash": "schema-1",
            "written_parts": ["data/part-000.parquet"],
        },
    )
    right = _write_compiled_dataset(
        tmp_path / "right_compiled",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        compiled={
            "source": "C:/dataset-a",
            "selected_schema_hash": "schema-1",
            "written_parts": ["data/part-000.parquet"],
        },
    )
    (right.path / "metadata.json").write_text("{not-json", encoding="utf-8")

    report = DiffDatasets(GenerateAnalysis()).run(left, right, DiffLevel.FULL)

    assert "Compiled dataset provenance changed" in report.metadata_changes


def test_compare_analyses_ignores_matching_nan_statistics() -> None:
    left = DatasetAnalysis(
        metadata=DatasetMetadata(
            version=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            row_count=1,
            column_count=1,
            source_file_count=1,
            schema_hash="schema-1",
            column_hashes={"value": "hash-1"},
        ),
        schemas=[SchemaSignature(columns=[], hash="schema-1", occurrence_count=1)],
        statistics=DatasetStatistics(
            columns={
                "value": ColumnStatistics(
                    dtype="Float64",
                    null_rate=math.nan,
                    unique_estimate=None,
                    min_value=None,
                    max_value=None,
                    mean=math.nan,
                    median=math.nan,
                    stddev=math.nan,
                )
            }
        ),
        warnings=[],
    )
    right = DatasetAnalysis(
        metadata=DatasetMetadata(
            version=1,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            row_count=1,
            column_count=1,
            source_file_count=1,
            schema_hash="schema-1",
            column_hashes={"value": "hash-1"},
        ),
        schemas=[SchemaSignature(columns=[], hash="schema-1", occurrence_count=1)],
        statistics=DatasetStatistics(
            columns={
                "value": ColumnStatistics(
                    dtype="Float64",
                    null_rate=math.nan,
                    unique_estimate=None,
                    min_value=None,
                    max_value=None,
                    mean=math.nan,
                    median=math.nan,
                    stddev=math.nan,
                )
            }
        ),
        warnings=[],
    )

    report = compare_analyses(left, right, DiffLevel.FULL)

    assert report.statistics_changes == []


def test_diff_report_to_dict_includes_summary_payload() -> None:
    report = DiffReport(
        file_changes=["Modified file: data.csv"],
        summary=DiffSummary(
            comparison_type=DiffComparisonType.FILE_FOLDER,
            match_status="ambiguous",
            candidate_paths=["data.csv", "nested/data.csv"],
        ),
    )

    payload = diff_report_to_dict(report)

    assert payload["summary"] == {
        "comparison_type": "file_vs_folder",
        "content_status": None,
        "match_status": "ambiguous",
        "matched_path": None,
        "candidate_paths": ["data.csv", "nested/data.csv"],
        "matching_files": None,
        "modified_files": None,
        "only_in_left": None,
        "only_in_right": None,
    }


def test_file_diff_reports_row_level_changes_with_keys(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pl.DataFrame(
        {
            "customer_id": [1, 2],
            "status": ["active", "active"],
            "balance": [10.0, 20.0],
        }
    ).write_csv(left)
    pl.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "status": ["active", "inactive", "active"],
            "balance": [10.0, 25.0, 40.0],
        }
    ).write_csv(right)

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(left),
        resolve_source(right),
        DiffLevel.FULL,
        key_columns=("customer_id",),
        row_examples=10,
    )

    assert report.row_diff_summary is not None
    assert report.row_diff_summary.added_rows == 1
    assert report.row_diff_summary.removed_rows == 0
    assert report.row_diff_summary.updated_rows == 1
    assert report.row_diff_summary.unchanged_rows == 1
    assert report.row_added_key_samples == [{"customer_id": 3}]
    assert report.row_removed_key_samples == []
    assert len(report.row_change_examples) == 1
    assert report.row_change_examples[0].key == {"customer_id": 2}


def test_file_diff_row_level_requires_unique_keys(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pl.DataFrame(
        {
            "customer_id": [1, 1],
            "status": ["active", "inactive"],
        }
    ).write_csv(left)
    pl.DataFrame({"customer_id": [1], "status": ["active"]}).write_csv(right)

    try:
        DiffDatasets(GenerateAnalysis()).run(
            resolve_source(left),
            resolve_source(right),
            DiffLevel.FULL,
            key_columns=("customer_id",),
        )
        assert False, "Expected ValueError for duplicate key rows"
    except ValueError as error:
        assert "Duplicate key rows detected" in str(error)


def test_diff_report_to_dict_includes_row_level_payload() -> None:
    report = DiffReport(
        row_diff_summary=None,
        row_change_examples=[],
        row_added_key_samples=[],
        row_removed_key_samples=[],
    )

    payload = diff_report_to_dict(report)

    assert "row_diff_summary" in payload
    assert "row_change_examples" in payload
    assert "row_added_key_samples" in payload
    assert "row_removed_key_samples" in payload


def test_file_diff_row_level_treats_matching_nan_as_unchanged(tmp_path: Path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pl.DataFrame(
        {
            "customer_id": [1, 2],
            "score": [math.nan, 10.0],
        }
    ).write_csv(left)
    pl.DataFrame(
        {
            "customer_id": [1, 2],
            "score": [math.nan, 10.0],
        }
    ).write_csv(right)

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(left),
        resolve_source(right),
        DiffLevel.FULL,
        key_columns=("customer_id",),
    )

    assert report.row_diff_summary is not None
    assert report.row_diff_summary.updated_rows == 0
    assert report.row_diff_summary.unchanged_rows == 2
    assert report.row_change_examples == []


def test_folder_diff_row_level_aggregates_across_matching_files(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    pl.DataFrame(
        {
            "customer_id": [1, 2],
            "status": ["active", "active"],
        }
    ).write_csv(left / "a.csv")
    pl.DataFrame(
        {
            "customer_id": [1, 2],
            "status": ["active", "inactive"],
        }
    ).write_csv(right / "a.csv")

    pl.DataFrame(
        {
            "customer_id": [10],
            "status": ["active"],
        }
    ).write_csv(left / "b.csv")
    pl.DataFrame(
        {
            "customer_id": [10, 11],
            "status": ["active", "active"],
        }
    ).write_csv(right / "b.csv")

    report = DiffDatasets(GenerateAnalysis()).run(
        resolve_source(left),
        resolve_source(right),
        DiffLevel.FULL,
        key_columns=("customer_id",),
        row_examples=5,
    )

    assert report.row_diff_summary is not None
    assert report.row_diff_summary.compared_files == 2
    assert report.row_diff_summary.added_rows == 1
    assert report.row_diff_summary.updated_rows == 1
    assert report.row_diff_summary.removed_rows == 0
    assert report.row_diff_summary.unchanged_rows == 2
