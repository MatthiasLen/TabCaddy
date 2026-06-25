from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from tabcaddy.plot.service import PlotDataset


def test_to_numeric_x_naive_datetime_uses_epoch_delta() -> None:
    dataset = PlotDataset()
    value = datetime(2026, 4, 22, 15, 4, 5, 123456)

    expected = (value - datetime(1970, 1, 1)).total_seconds()

    assert dataset._to_numeric_x(value) == expected


def test_to_numeric_x_naive_date_uses_epoch_delta() -> None:
    dataset = PlotDataset()
    value = date(2026, 4, 22)

    expected = (datetime(2026, 4, 22) - datetime(1970, 1, 1)).total_seconds()

    assert dataset._to_numeric_x(value) == expected


def test_to_numeric_x_timezone_aware_datetime_keeps_timestamp_behavior() -> None:
    dataset = PlotDataset()
    value = datetime(2026, 4, 22, 15, 4, 5, 123456, tzinfo=timezone.utc)

    assert dataset._to_numeric_x(value) == value.timestamp()


def test_to_numeric_x_timedelta_uses_total_seconds() -> None:
    dataset = PlotDataset()
    value = timedelta(days=1, hours=2, minutes=3, seconds=4, microseconds=500000)

    assert dataset._to_numeric_x(value) == value.total_seconds()


def test_duration_x_line_plot_keeps_points() -> None:
    dataset = PlotDataset()
    lazyframe = pl.DataFrame(
        {
            "x": [timedelta(seconds=1), timedelta(seconds=2), timedelta(seconds=3)],
            "y": [1.0, 2.0, 3.0],
        }
    ).lazy()

    result = dataset._run_for_lazyframe(
        lazyframe,
        "x",
        "y",
        kind="line",
        aggregate_x=None,
        line_interpolation="linear",
        fail_on_x_duplicates=False,
        fail_on_unsorted_x=False,
        filter_expr=None,
    )

    assert result.chart_kind == "line"
    assert result.x_axis_kind == "temporal"
    assert result.x_axis_time_unit is None
    assert result.x_axis_timezone is None
    assert result.plotted_rows == 3
    assert result.dropped_rows == 0
    assert result.line_x_values == [1.0, 2.0, 3.0]


def test_duration_x_scatter_plot_keeps_points() -> None:
    dataset = PlotDataset()
    lazyframe = pl.DataFrame(
        {
            "x": [timedelta(seconds=1), timedelta(seconds=2), timedelta(seconds=3)],
            "y": [1.0, 2.0, 3.0],
        }
    ).lazy()

    result = dataset._run_for_lazyframe(
        lazyframe,
        "x",
        "y",
        kind="scatter",
        aggregate_x=None,
        line_interpolation="linear",
        fail_on_x_duplicates=False,
        fail_on_unsorted_x=False,
        filter_expr=None,
    )

    assert result.chart_kind == "scatter"
    assert result.x_axis_kind == "temporal"
    assert result.x_axis_time_unit is None
    assert result.x_axis_timezone is None
    assert result.plotted_rows == 3
    assert result.dropped_rows == 0
    assert result.scatter_points == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert result.scatter_outlier_points == []


def test_scatter_plot_classifies_local_outlier() -> None:
    dataset = PlotDataset()
    x_values = [float(index) for index in range(24)]
    y_values = [10.0 + (0.2 if index % 2 == 0 else -0.2) for index in range(24)]
    y_values[11] = 80.0

    lazyframe = pl.DataFrame({"x": x_values, "y": y_values}).lazy()

    result = dataset._run_for_lazyframe(
        lazyframe,
        "x",
        "y",
        kind="scatter",
        aggregate_x=None,
        line_interpolation="linear",
        fail_on_x_duplicates=False,
        fail_on_unsorted_x=False,
        filter_expr=None,
    )

    assert result.chart_kind == "scatter"
    assert result.plotted_rows == len(x_values)
    assert (11.0, 80.0) in result.scatter_outlier_points
    assert len(result.scatter_inlier_points) + len(
        result.scatter_outlier_points
    ) == len(result.scatter_points)


def test_parse_filter_value_non_datetime_dtype_ignores_non_callable_base_type() -> None:
    class FakeDType:
        base_type = "not-callable"

        def __str__(self) -> str:
            return "FakeNumeric"

    dataset = PlotDataset()

    parsed = dataset._parse_filter_value("42", dtype=FakeDType())  # type: ignore[arg-type]

    assert parsed == 42


def test_histogram_plot_builds_bins_for_numeric_column() -> None:
    dataset = PlotDataset()
    lazyframe = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0]}).lazy()

    result = dataset._run_histogram_for_lazyframe(
        lazyframe,
        "value",
        filter_expr=None,
    )

    assert result.chart_kind == "histogram"
    assert result.plotted_rows == 4
    assert result.histogram_bins
    assert sum(count for _, count in result.histogram_bins) == 4


def test_histogram_plot_rejects_non_numeric_column_values() -> None:
    dataset = PlotDataset()
    lazyframe = pl.DataFrame({"value": ["a", "b", "c"]}).lazy()

    with pytest.raises(ValueError, match="No plottable numeric values remain"):
        dataset._run_histogram_for_lazyframe(
            lazyframe,
            "value",
            filter_expr=None,
        )


def test_histogram_plot_formats_temporal_bucket_bounds() -> None:
    dataset = PlotDataset()
    lazyframe = pl.DataFrame(
        {
            "ts": [
                datetime(2026, 1, 1, 0, 0, 0),
                datetime(2026, 1, 2, 0, 0, 0),
                datetime(2026, 1, 3, 0, 0, 0),
                datetime(2026, 1, 4, 0, 0, 0),
            ]
        }
    ).lazy()

    result = dataset._run_histogram_for_lazyframe(
        lazyframe,
        "ts",
        filter_expr=None,
    )

    assert result.chart_kind == "histogram"
    assert result.x_axis_kind == "temporal"
    assert result.x_axis_time_unit == "epoch_seconds"
    assert result.x_axis_timezone == "UTC"
    assert result.histogram_bins
    assert any("2026-01" in label for label, _ in result.histogram_bins)
