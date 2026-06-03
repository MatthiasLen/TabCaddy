from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tabcaddy.domain.models import ColumnDefinition
from tabcaddy.domain.models import ColumnStatistics
from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.domain.models import DatasetMetadata
from tabcaddy.domain.models import DatasetStatistics
from tabcaddy.domain.models import SchemaSignature
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.views.summary import build_summary_view


def test_summary_render_snapshot() -> None:
    analysis = DatasetAnalysis(
        metadata=DatasetMetadata(
            version=1,
            created_at=datetime(2024, 1, 5, 12, 0, tzinfo=timezone.utc),
            row_count=4,
            column_count=3,
            source_file_count=2,
            schema_hash="abc123",
            column_hashes={"id": "aaa", "amount": "bbb", "trade_date": "ccc"},
        ),
        schemas=[
            SchemaSignature(
                columns=[
                    ColumnDefinition("id", "Int64"),
                    ColumnDefinition("amount", "Float64"),
                    ColumnDefinition("trade_date", "Date"),
                ],
                hash="abc123",
                occurrence_count=2,
            )
        ],
        statistics=DatasetStatistics(
            columns={
                "id": ColumnStatistics("Int64", 0.0, 4, 1, 4, 2.5, 2.5, 1.29),
                "amount": ColumnStatistics("Float64", 0.0, 4, 9.5, 15.0, 11.625, 11.0, 2.38),
                "trade_date": ColumnStatistics("Date", 0.0, None, "2024-01-01", "2024-01-04", None, None, None),
            }
        ),
        warnings=["Schema drift detected across 1 schema groups."],
    )
    console = create_console(record=True, width=100)
    console.print(build_summary_view(analysis))
    output = console.export_text()

    snapshot = (Path(__file__).parent / "snapshots" / "summary_output.txt").read_text(encoding="utf-8")
    assert output == snapshot