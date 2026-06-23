from __future__ import annotations

from datetime import date, datetime, timezone

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