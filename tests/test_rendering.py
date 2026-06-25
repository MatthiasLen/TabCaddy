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
from tabcaddy.rendering.charts.axis_formatters import format_epoch_seconds_utc
from tabcaddy.rendering.console import create_console
from tabcaddy.rendering.console import RenderProfile
from tabcaddy.rendering.charts.line_chart import _resample_by_x
from tabcaddy.rendering.charts.line_chart import render_line_chart
from tabcaddy.rendering.charts.scatter_chart import render_scatter_chart
from tabcaddy.plot.service import PlotFileResult
from tabcaddy.plot.service import PlotResult
from tabcaddy.plot.service import PlotRunResult
from tabcaddy.rendering.views.diff import build_diff_view
from tabcaddy.rendering.views.plot import build_plot_view
from tabcaddy.rendering.views.plot import build_multi_y_plot_view
from tabcaddy.rendering.views.plot import _resolve_chart_width
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


def test_line_resampling_supports_nearest_neighbor_mode() -> None:
    values = [0.0, 10.0]
    x_values = [0.0, 10.0]

    linear = _resample_by_x(
        values,
        x_values,
        target_points=3,
        interpolation="linear",
    )
    nearest = _resample_by_x(
        values,
        x_values,
        target_points=3,
        interpolation="nearest",
    )

    assert linear == [0.0, 5.0, 10.0]
    assert nearest == [0.0, 0.0, 10.0]


def test_scatter_chart_clamps_small_dimensions() -> None:
    points = [(0.0, 0.0), (1.0, 1.0)]

    zero_dim_chart = render_scatter_chart(points, width=0, height=0)
    one_dim_chart = render_scatter_chart(points, width=1, height=1)

    assert zero_dim_chart
    assert one_dim_chart


def test_line_chart_adds_utc_footer_for_temporal_x() -> None:
    x_values = [
        datetime(2026, 2, 7, tzinfo=timezone.utc).timestamp(),
        datetime(2026, 2, 8, tzinfo=timezone.utc).timestamp(),
        datetime(2026, 2, 9, tzinfo=timezone.utc).timestamp(),
    ]
    y_values = [1.0, 2.0, 3.0]

    chart = render_line_chart(
        y_values,
        x_values=x_values,
        x_tick_formatter=format_epoch_seconds_utc,
        width=30,
    )

    assert "2026-02-07" in chart
    assert "2026-02-09" in chart


def test_line_chart_supports_ascii_symbols() -> None:
    chart = render_line_chart([1.0, 2.0, 1.0], ascii_only=True, width=20)

    chart.encode("cp1252")
    assert "┤" not in chart
    assert "┼" not in chart
    assert "─" not in chart


def test_plot_view_line_respects_ascii_render_profile() -> None:
    result = PlotResult(
        chart_kind="line",
        x_column="x",
        y_column="y",
        x_axis_kind="numeric",
        x_axis_time_unit=None,
        x_axis_timezone=None,
        row_count=3,
        plotted_rows=3,
        dropped_rows=0,
        duplicate_x_count=0,
        sorted_x=True,
        auto_sorted=False,
        aggregated=False,
        line_interpolation="linear",
        line_x_values=[1.0, 2.0, 3.0],
        line_values=[1.0, 2.0, 1.5],
    )
    run_result = PlotRunResult(
        plots=[PlotFileResult(path=Path("plot.csv"), result=result)],
        total_files=1,
        plotted_files=1,
        skipped_files=0,
    )

    console = create_console(record=True, width=80, legacy_windows=False)
    console.print(build_plot_view(run_result, render=RenderProfile(ascii_only=True)))
    output = console.export_text()

    output.encode("cp1252")
    assert "┤" not in output
    assert "┼" not in output
    assert "─" not in output


def test_plot_view_line_adds_numeric_x_axis_labels() -> None:
    result = PlotResult(
        chart_kind="line",
        x_column="x",
        y_column="y",
        x_axis_kind="numeric",
        x_axis_time_unit=None,
        x_axis_timezone=None,
        row_count=3,
        plotted_rows=3,
        dropped_rows=0,
        duplicate_x_count=0,
        sorted_x=True,
        auto_sorted=False,
        aggregated=False,
        line_interpolation="linear",
        line_x_values=[10.25, 10.5, 10.75],
        line_values=[1.0, 2.0, 3.0],
    )
    run_result = PlotRunResult(
        plots=[PlotFileResult(path=Path("plot.csv"), result=result)],
        total_files=1,
        plotted_files=1,
        skipped_files=0,
    )

    console = create_console(record=True, width=80, legacy_windows=False)
    console.print(build_plot_view(run_result))
    output = console.export_text()

    footer = output.splitlines()[-1]
    assert "10.25" in footer
    assert "10.75" in footer


def test_scatter_chart_formats_temporal_x_labels_as_utc_dates() -> None:
    points = [
        (datetime(2026, 2, 7, tzinfo=timezone.utc).timestamp(), 1.0),
        (datetime(2026, 2, 9, tzinfo=timezone.utc).timestamp(), 2.0),
    ]

    chart = render_scatter_chart(points, x_tick_formatter=format_epoch_seconds_utc)

    assert "2026-02-07" in chart
    assert "2026-02-09" in chart


def test_scatter_chart_uses_box_drawing_axes_by_default() -> None:
    points = [(0.0, 0.0), (1.0, 1.0)]

    chart = render_scatter_chart(points)

    assert "│" in chart
    assert "└" in chart
    assert "─" in chart


def test_scatter_chart_supports_ascii_axis_chars() -> None:
    points = [(0.0, 0.0), (1.0, 1.0)]

    chart = render_scatter_chart(
        points,
        y_axis_char="|",
        x_axis_char="-",
        axis_corner_char="+",
    )

    assert "|" in chart
    assert "+" in chart
    assert "-" in chart


def test_scatter_chart_renders_outlier_overlay_marker() -> None:
    chart = render_scatter_chart(
        [(0.0, 0.0)],
        outlier_points=[(0.0, 0.0)],
        point=".",
        outlier_point="X",
        width=8,
        height=4,
    )

    assert "X" in chart


def test_plot_view_scatter_shows_outlier_count() -> None:
    result = PlotResult(
        chart_kind="scatter",
        x_column="x",
        y_column="y",
        x_axis_kind="numeric",
        x_axis_time_unit=None,
        x_axis_timezone=None,
        row_count=3,
        plotted_rows=3,
        dropped_rows=0,
        duplicate_x_count=0,
        sorted_x=True,
        auto_sorted=False,
        aggregated=False,
        line_interpolation=None,
        scatter_points=[(0.0, 1.0), (1.0, 1.0), (2.0, 10.0)],
        scatter_inlier_points=[(0.0, 1.0), (1.0, 1.0)],
        scatter_outlier_points=[(2.0, 10.0)],
    )
    run_result = PlotRunResult(
        plots=[PlotFileResult(path=Path("plot.csv"), result=result)],
        total_files=1,
        plotted_files=1,
        skipped_files=0,
    )

    console = create_console(record=True, width=100, legacy_windows=False)
    console.print(build_plot_view(run_result, render=RenderProfile(ascii_only=True)))
    output = console.export_text()

    outliers_line = next(line for line in output.splitlines() if "Outliers" in line)

    assert "Outliers" in outliers_line
    assert "1" in outliers_line
    output.encode("cp1252")
    assert "." in output
    assert "*" in output


def test_scatter_chart_single_x_uses_true_footer_labels() -> None:
    chart = render_scatter_chart([(10.0, 5.0)], width=20, height=6)

    footer = chart.splitlines()[-1]
    assert "10" in footer
    assert "9.5" not in footer
    assert "10.5" not in footer


def test_scatter_chart_single_y_uses_true_axis_labels() -> None:
    chart = render_scatter_chart([(10.0, 5.0)], width=20, height=6)

    y_axis_lines = chart.splitlines()[:-2]
    assert y_axis_lines
    assert all(line.startswith("     5 ") for line in y_axis_lines)
    assert "5.5" not in chart
    assert "4.5" not in chart


def test_plot_view_histogram_renders_bins_and_metadata() -> None:
    result = PlotResult(
        chart_kind="histogram",
        x_column="value",
        y_column="value",
        x_axis_kind="numeric",
        x_axis_time_unit=None,
        x_axis_timezone=None,
        row_count=4,
        plotted_rows=4,
        dropped_rows=0,
        duplicate_x_count=0,
        sorted_x=False,
        auto_sorted=False,
        aggregated=False,
        line_interpolation=None,
        histogram_bins=[("1..2", 2), ("2..3", 2)],
    )
    run_result = PlotRunResult(
        plots=[PlotFileResult(path=Path("plot.csv"), result=result)],
        total_files=1,
        plotted_files=1,
        skipped_files=0,
    )

    console = create_console(record=True, width=100, legacy_windows=False)
    console.print(build_plot_view(run_result, render=RenderProfile(ascii_only=True)))
    output = console.export_text()

    assert "histogram" in output
    assert "Bins" in output
    assert "1..2" in output
    assert "Column" in output


def test_plot_chart_width_uses_full_width_for_narrow_console() -> None:
    width = _resolve_chart_width(console_width=80)

    assert width == 80


def test_plot_chart_width_scales_for_sparse_data_on_wide_console() -> None:
    width = _resolve_chart_width(console_width=140)

    assert width == 132


def test_plot_chart_width_scales_for_dense_data_on_wide_console() -> None:
    width = _resolve_chart_width(console_width=140)

    assert width == 132


def test_plot_chart_width_applies_generous_cap_for_extremely_wide_console() -> None:
    width = _resolve_chart_width(console_width=1000)

    assert width == 480


def test_multi_y_plot_metadata_uses_per_series_values() -> None:
    result_a = PlotResult(
        chart_kind="line",
        x_column="x",
        y_column="y_a",
        x_axis_kind="numeric",
        x_axis_time_unit=None,
        x_axis_timezone=None,
        row_count=5,
        plotted_rows=4,
        dropped_rows=1,
        duplicate_x_count=0,
        sorted_x=True,
        auto_sorted=False,
        aggregated=False,
        line_interpolation="nearest",
        line_x_values=[1.0, 2.0, 3.0, 4.0],
        line_values=[10.0, 11.0, 12.0, 13.0],
    )
    result_b = PlotResult(
        chart_kind="line",
        x_column="x",
        y_column="y_b",
        x_axis_kind="numeric",
        x_axis_time_unit=None,
        x_axis_timezone=None,
        row_count=5,
        plotted_rows=2,
        dropped_rows=3,
        duplicate_x_count=0,
        sorted_x=True,
        auto_sorted=False,
        aggregated=False,
        line_interpolation="nearest",
        line_x_values=[1.0, 4.0],
        line_values=[100.0, 130.0],
    )
    run_a = PlotRunResult(
        plots=[PlotFileResult(path=Path("a.csv"), result=result_a)],
        total_files=1,
        plotted_files=1,
        skipped_files=0,
    )
    run_b = PlotRunResult(
        plots=[PlotFileResult(path=Path("a.csv"), result=result_b)],
        total_files=1,
        plotted_files=1,
        skipped_files=0,
    )

    console = create_console(record=True, width=120, legacy_windows=False)
    console.print(build_multi_y_plot_view([("y_a", run_a), ("y_b", run_b)]))
    output = console.export_text()

    plotted_rows_line = next(
        line for line in output.splitlines() if "Plotted rows" in line
    )
    dropped_rows_line = next(
        line for line in output.splitlines() if "Dropped rows" in line
    )

    assert output.count("Plotted rows") == 1
    assert "y_a" in output
    assert "y_b" in output
    assert "4" in plotted_rows_line
    assert "2" in plotted_rows_line
    assert "1" in dropped_rows_line
    assert "3" in dropped_rows_line
