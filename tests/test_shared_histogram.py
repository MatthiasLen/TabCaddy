from __future__ import annotations

import numpy as np

from tabcaddy.shared.histogram import build_numeric_histogram
from tabcaddy.shared.histogram import format_histogram_bound
from tabcaddy.shared.histogram import resolve_histogram_bin_count


def test_format_histogram_bound_matches_expected_labels() -> None:
    assert format_histogram_bound(float("nan")) == "nan"
    assert format_histogram_bound(float("inf")) == "inf"
    assert format_histogram_bound(float("-inf")) == "-inf"
    assert format_histogram_bound(2.0) == "2"
    assert format_histogram_bound(2.3456) == "2.35"


def test_resolve_histogram_bin_count_uses_sqrt_bounds() -> None:
    assert resolve_histogram_bin_count(1) == 2
    assert resolve_histogram_bin_count(4) == 2
    assert resolve_histogram_bin_count(25) == 5
    assert resolve_histogram_bin_count(400) == 8


def test_build_numeric_histogram_handles_empty_finite_values() -> None:
    values = np.asarray([float("nan"), float("inf"), float("-inf")], dtype=float)

    assert build_numeric_histogram(values) == []


def test_build_numeric_histogram_handles_constant_values() -> None:
    values = np.asarray([3.0, 3.0, 3.0], dtype=float)

    assert build_numeric_histogram(values) == [("3", 3)]


def test_build_numeric_histogram_counts_all_finite_values() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    histogram = build_numeric_histogram(values)

    assert histogram
    assert sum(count for _, count in histogram) == 4
