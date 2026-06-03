from __future__ import annotations

from pathlib import Path

import polars as pl

from tabcaddy.application.generate_analysis import GenerateAnalysis
from tabcaddy.domain.models import DiffLevel, ProfileMode
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder
from tabcaddy.infrastructure.cache_manager import CacheManager
from tabcaddy.infrastructure.diff_support import compare_analyses
from tabcaddy.infrastructure.schema_analyzer import SchemaAnalyzer
from tabcaddy.infrastructure.source_resolver import resolve_source


def _write_csv(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def _write_feather(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_ipc(path)


def test_schema_hashing_and_grouping(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}])
    _write_csv(data / "b.csv", [{"id": 3, "value": 12.0}])
    _write_csv(data / "drift.csv", [{"id": 1, "label": "x"}])

    result = SchemaAnalyzer().analyze(resolve_source(data))

    assert len(result.schemas) == 2
    assert result.schemas[0].occurrence_count == 2
    assert result.schemas[0].hash != result.schemas[1].hash


def test_analysis_builder_generates_metadata_and_statistics(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(
        data / "a.csv",
        [
            {"id": 1, "amount": 10.0, "trade_date": "2024-01-01"},
            {"id": 2, "amount": 12.0, "trade_date": "2024-01-02"},
        ],
    )
    _write_feather(
        data / "b.feather",
        [
            {"id": 3, "amount": 9.5, "trade_date": "2024-01-03"},
            {"id": 4, "amount": 15.0, "trade_date": "2024-01-04"},
        ],
    )

    analysis = AnalysisBuilder().build(resolve_source(data), ProfileMode.DEEP).analysis

    assert analysis.metadata.row_count == 4
    assert analysis.metadata.column_count == 3
    assert analysis.statistics is not None
    assert analysis.statistics.columns["amount"].mean == 11.625
    assert analysis.statistics.columns["amount"].histogram is not None
    assert sum(count for _, count in analysis.statistics.columns["amount"].histogram or []) == 4
    assert analysis.metadata.column_hashes is not None
    assert "amount" in analysis.metadata.column_hashes


def test_cache_manager_round_trip(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_csv(data / "a.csv", [{"id": 1, "value": 10.0}])

    source = resolve_source(data)
    analysis = AnalysisBuilder().build(source, ProfileMode.STANDARD).analysis
    cache = CacheManager(tmp_path / ".tabcaddy" / "cache")

    cache.set(source, ProfileMode.STANDARD, analysis)
    cached = cache.get(source, ProfileMode.STANDARD)

    assert cached is not None
    assert cached.metadata.row_count == 1
    assert cached.schemas[0].hash == analysis.schemas[0].hash


def test_diff_logic_reports_changes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    _write_csv(left / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 11.0}])
    _write_csv(right / "a.csv", [{"id": 1, "value": 10.0}, {"id": 2, "value": 20.0}])

    generator = GenerateAnalysis()
    left_analysis = generator.run(resolve_source(left), ProfileMode.DEEP)
    right_analysis = generator.run(resolve_source(right), ProfileMode.DEEP)
    report = compare_analyses(left_analysis, right_analysis, DiffLevel.FULL)

    assert any("value.max_value" in item or "value.mean" in item for item in report.statistics_changes)