from __future__ import annotations

from pathlib import Path

import polars as pl

from tabcaddy.application.diff_datasets import DiffDatasets
from tabcaddy.application.generate_analysis import GenerateAnalysis
from tabcaddy.domain.models import DiffLevel
from tabcaddy.infrastructure.source_resolver import resolve_source


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
    assert "Modified file: data.csv" in report.metadata_changes
