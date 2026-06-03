from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import polars as pl

from tabcaddy.domain.models import ProfileMode
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder
from tabcaddy.infrastructure.cache_manager import CacheManager
from tabcaddy.infrastructure.source_resolver import resolve_source


def test_analysis_builder_computes_metadata_and_statistics(homogeneous_folder) -> None:
    analysis = (
        AnalysisBuilder()
        .build(resolve_source(homogeneous_folder), ProfileMode.STANDARD)
        .analysis
    )
    assert analysis.metadata.row_count == 4
    assert analysis.metadata.column_count == 3
    assert analysis.metadata.source_file_count == 2
    assert analysis.statistics is not None
    assert analysis.statistics.columns["id"].null_rate == 0.0
    assert analysis.statistics.columns["value"].mean == pytest.approx(25.0)


def test_cache_manager_round_trips_analysis(tmp_path, homogeneous_folder) -> None:
    source = resolve_source(homogeneous_folder)
    analysis = AnalysisBuilder().build(source, ProfileMode.DEEP).analysis
    cache = CacheManager(tmp_path / ".tabcaddy" / "cache")
    cache.set(source, ProfileMode.DEEP, analysis)
    loaded = cache.get(source, ProfileMode.DEEP)
    assert loaded is not None
    assert loaded.metadata.row_count == analysis.metadata.row_count
    assert loaded.schemas[0].hash == analysis.schemas[0].hash
    assert loaded.statistics is not None
    assert analysis.statistics is not None
    assert (
        loaded.statistics.columns["value"].histogram
        == analysis.statistics.columns["value"].histogram
    )
def test_analysis_builder_handles_timezone_aware_datetimes(tmp_path: Path) -> None:
    source_file = tmp_path / "events.feather"
    pl.DataFrame(
        {
            "event_ts": [
                datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 9, 30, tzinfo=timezone.utc),
            ],
            "value": [1.0, 2.0],
        }
    ).write_ipc(source_file)

    analysis = AnalysisBuilder().build_file_set(
        files=[source_file],
        base_path=tmp_path,
        source_type=resolve_source(source_file).source_type,
        profile_mode=ProfileMode.STANDARD,
    ).analysis

    assert analysis.statistics is not None
    assert (
        analysis.statistics.columns["event_ts"].min_value
        == "2024-01-01T08:00:00+00:00"
    )
    assert (
        analysis.statistics.columns["event_ts"].max_value
        == "2024-01-02T09:30:00+00:00"
    )
