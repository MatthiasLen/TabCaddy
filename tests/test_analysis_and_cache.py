from __future__ import annotations

import json
from pathlib import Path

import pytest
import polars as pl

from tabcaddy.analysis import (
    AnalysisBuilder,
    CacheManager,
    GenerateAnalysis,
    resolve_source,
)
from tabcaddy.domain.models import ProfileMode
from tabcaddy.shared.dataset_io import read_compiled_analysis


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
    result = AnalysisBuilder().build(source, ProfileMode.DEEP)
    cache = CacheManager(tmp_path / ".tabcaddy" / "cache")
    cache.set(source, ProfileMode.DEEP, result)
    loaded = cache.get(source, ProfileMode.DEEP)
    assert loaded is not None
    assert loaded.analysis.metadata.row_count == result.analysis.metadata.row_count
    assert loaded.analysis.schemas[0].hash == result.analysis.schemas[0].hash
    assert loaded.analysis.statistics is not None
    assert result.analysis.statistics is not None
    assert loaded.files == result.files
    assert (
        loaded.analysis.statistics.columns["value"].histogram
        == result.analysis.statistics.columns["value"].histogram
    )


def test_analysis_builder_handles_timezone_aware_datetimes(tmp_path: Path) -> None:
    source_file = tmp_path / "events.feather"
    pl.DataFrame(
        {
            "event_ts": [1704096000000000, 1704187800000000],
            "value": [1.0, 2.0],
        },
        schema={"event_ts": pl.Int64, "value": pl.Float64},
    ).with_columns(
        pl.col("event_ts").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
    ).write_ipc(source_file)

    analysis = (
        AnalysisBuilder()
        .build_file_set(
            files=[source_file],
            base_path=tmp_path,
            source_type=resolve_source(source_file).source_type,
            profile_mode=ProfileMode.STANDARD,
        )
        .analysis
    )

    assert analysis.statistics is not None
    assert (
        analysis.statistics.columns["event_ts"].min_value == "2024-01-01T08:00:00+00:00"
    )
    assert (
        analysis.statistics.columns["event_ts"].max_value == "2024-01-02T09:30:00+00:00"
    )


def test_generate_analysis_uses_cached_build_result(
    tmp_path, homogeneous_folder
) -> None:
    source = resolve_source(homogeneous_folder)
    cached_result = AnalysisBuilder().build(source, ProfileMode.STANDARD)
    cache = CacheManager(tmp_path / ".tabcaddy" / "cache")
    cache.set(source, ProfileMode.STANDARD, cached_result)

    class FailOnBuildAnalysisBuilder:
        def build(self, source, profile_mode):
            raise AssertionError("cache hit should not rebuild analysis")

    result = GenerateAnalysis(
        analysis_builder=FailOnBuildAnalysisBuilder(),
        cache_manager=cache,
    ).run(source, ProfileMode.STANDARD)

    assert result == cached_result


def test_generate_analysis_invalidates_stale_cache_payload(
    tmp_path, homogeneous_folder
) -> None:
    source = resolve_source(homogeneous_folder)
    cache = CacheManager(tmp_path / ".tabcaddy" / "cache")
    cache_file = cache.cache_file(source, ProfileMode.QUICK)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"metadata": {"version": 0}}),
        encoding="utf-8",
    )

    result = GenerateAnalysis(
        analysis_builder=AnalysisBuilder(),
        cache_manager=cache,
    ).run(source, ProfileMode.QUICK)

    assert result.files
    assert result.analysis.schemas
    assert cache_file.exists()
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    assert payload["format_version"] == CacheManager.CACHE_FORMAT_VERSION


def test_read_compiled_analysis_returns_none_for_malformed_metadata(
    tmp_path: Path,
) -> None:
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "metadata.json").write_text("{not-json", encoding="utf-8")

    assert read_compiled_analysis(compiled) is None


def test_read_compiled_analysis_returns_none_for_invalid_payload_shape(
    tmp_path: Path,
) -> None:
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "metadata.json").write_text(
        json.dumps({"warnings": []}),
        encoding="utf-8",
    )

    assert read_compiled_analysis(compiled) is None
