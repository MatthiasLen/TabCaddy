from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tabcaddy.domain.models import ColumnDefinition
from tabcaddy.domain.models import ColumnStatistics
from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.domain.models import DatasetMetadata
from tabcaddy.domain.models import DatasetStatistics
from tabcaddy.domain.models import DiffComparisonType
from tabcaddy.domain.models import DiffLevel
from tabcaddy.domain.models import DiffReport
from tabcaddy.domain.models import RowChangeExample
from tabcaddy.domain.models import RowDiffSummary
from tabcaddy.domain.models import RowFieldDelta
from tabcaddy.domain.models import DiffSummary
from tabcaddy.domain.models import SchemaSignature
from tabcaddy.analysis.schema import FileSchemaRecord
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.console import RenderProfile
from tabcaddy.rendering.charts.line_chart import render_line_chart
from tabcaddy.rendering.views.diff import build_diff_view
from tabcaddy.rendering.views.schema import build_schema_view
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
                "amount": ColumnStatistics(
                    "Float64", 0.0, 4, 9.5, 15.0, 11.625, 11.0, 2.38
                ),
                "trade_date": ColumnStatistics(
                    "Date", 0.0, None, "2024-01-01", "2024-01-04", None, None, None
                ),
            }
        ),
        warnings=["Schema drift detected across 1 schema groups."],
    )
    console = create_console(record=True, width=100, legacy_windows=False)
    console.print(build_summary_view(analysis))
    output = console.export_text()

    snapshot = (Path(__file__).parent / "snapshots" / "summary_output.txt").read_text(
        encoding="utf-8"
    )
    assert output == snapshot


def test_schema_render_ascii_fallback_is_cp1252_safe() -> None:
    analysis = DatasetAnalysis(
        metadata=DatasetMetadata(
            version=1,
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            row_count=3,
            column_count=2,
            source_file_count=2,
            schema_hash="abc123",
            column_hashes=None,
        ),
        schemas=[
            SchemaSignature(
                columns=[
                    ColumnDefinition("id", "Int64"),
                    ColumnDefinition("value", "Float64"),
                ],
                hash="abc123",
                occurrence_count=2,
            )
        ],
        statistics=None,
        warnings=[],
    )
    files = [
        FileSchemaRecord(
            path=Path("a.csv"),
            relative_path=Path("a.csv"),
            schema_hash="abc123",
            columns=analysis.schemas[0].columns,
            row_count=2,
        )
    ]

    console = create_console(record=True, width=100, legacy_windows=False)
    console.print(
        build_schema_view(analysis, files, render=RenderProfile(ascii_only=True))
    )
    output = console.export_text()

    output.encode("cp1252")
    assert "#" in output
    assert "█" not in output


def test_diff_render_uses_compact_empty_state_and_hides_empty_sections() -> None:
    report = DiffReport(
        file_changes=[],
        metadata_changes=[],
        schema_changes=[],
        statistics_changes=[],
        warnings=[],
        summary=DiffSummary(
            comparison_type=DiffComparisonType.COMPILED,
            content_status="identical",
        ),
    )
    console = create_console(record=True, width=100, legacy_windows=False)
    console.print(build_diff_view(report, level=DiffLevel.FULL))
    output = console.export_text()

    assert "No differences detected." in output
    assert "File Changes" not in output
    assert "Dataset Metadata" not in output


def test_diff_render_shows_match_candidates() -> None:
    report = DiffReport(
        file_changes=[],
        metadata_changes=[],
        schema_changes=[],
        statistics_changes=[],
        warnings=[],
        summary=DiffSummary(
            comparison_type=DiffComparisonType.FILE_FOLDER,
            match_status="ambiguous",
            candidate_paths=["data.csv", "nested/data.csv"],
        ),
    )
    console = create_console(record=True, width=100, legacy_windows=False)
    console.print(build_diff_view(report, level=DiffLevel.FULL))
    output = console.export_text()

    assert "Match Candidates" in output
    assert "nested/data.csv" in output


def test_diff_render_shows_row_level_sections() -> None:
    report = DiffReport(
        summary=DiffSummary(
            comparison_type=DiffComparisonType.FILE,
            content_status="modified",
        ),
        row_diff_summary=RowDiffSummary(
            key_columns=["customer_id"],
            added_rows=1,
            removed_rows=0,
            updated_rows=1,
            unchanged_rows=2,
            compared_files=1,
        ),
        row_change_examples=[
            RowChangeExample(
                key={"customer_id": 42},
                deltas=[
                    RowFieldDelta(
                        column="status",
                        left_value="active",
                        right_value="inactive",
                    )
                ],
            )
        ],
        row_added_key_samples=[{"customer_id": 99}],
    )

    console = create_console(record=True, width=120, legacy_windows=False)
    console.print(build_diff_view(report, level=DiffLevel.FULL))
    output = console.export_text()

    assert "Row Diff Summary" in output
    assert "Updated Rows" in output
    assert "Added Key Samples" in output
    assert "customer_id=42" in output


def test_line_chart_respects_x_spacing_with_resampling() -> None:
    dense_x = [0.0, 1.0, 2.0, 3.0]
    dense_y = [0.0, 10.0, 20.0, 30.0]
    sparse_x = [0.0, 100.0, 101.0, 102.0]
    sparse_y = [0.0, 10.0, 20.0, 30.0]

    dense_chart = render_line_chart(dense_y, x_values=dense_x, width=40)
    sparse_chart = render_line_chart(sparse_y, x_values=sparse_x, width=40)

    assert dense_chart
    assert sparse_chart
    assert dense_chart != sparse_chart
