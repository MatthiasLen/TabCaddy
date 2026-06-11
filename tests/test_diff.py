from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import polars as pl

from tabcaddy.application.diff_datasets import DiffDatasets
from tabcaddy.application.generate_analysis import GenerateAnalysis
from tabcaddy.domain.models import (
    DatasetAnalysis,
    DatasetMetadata,
    DatasetSource,
    DatasetStatistics,
    DiffLevel,
    SchemaSignature,
    SourceType,
)
from tabcaddy.domain.serialization import analysis_to_dict
from tabcaddy.infrastructure.source_resolver import resolve_source


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
