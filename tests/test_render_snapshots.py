from __future__ import annotations

from datetime import datetime, timezone

from tabcaddy.domain.models import ColumnDefinition
from tabcaddy.domain.models import ColumnStatistics
from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.domain.models import DatasetMetadata
from tabcaddy.domain.models import DatasetStatistics
from tabcaddy.domain.models import SchemaSignature
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.views.summary import build_summary_view


def test_summary_render_includes_histograms_when_present() -> None:
    analysis = DatasetAnalysis(
        metadata=DatasetMetadata(
            version=1,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            row_count=4,
            column_count=1,
            source_file_count=1,
            schema_hash="abc123",
            column_hashes={"value": "xyz"},
        ),
        schemas=[
            SchemaSignature(
                columns=[ColumnDefinition("value", "Float64")],
                hash="abc123",
                occurrence_count=1,
            )
        ],
        statistics=DatasetStatistics(
            columns={
                "value": ColumnStatistics(
                    "Float64",
                    0.0,
                    4,
                    10.0,
                    40.0,
                    25.0,
                    25.0,
                    12.9,
                    [("10..20", 1), ("20..30", 2), ("30..40", 1)],
                )
            }
        ),
        warnings=[],
    )
    console = create_console(record=True, width=100)
    console.print(build_summary_view(analysis))
    output = console.export_text()
    assert "Histogram: value" in output
    assert "10..20" in output
