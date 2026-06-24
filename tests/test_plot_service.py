from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl

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
