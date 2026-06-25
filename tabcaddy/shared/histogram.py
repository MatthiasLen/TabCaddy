from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np


_DEFAULT_MIN_BINS = 2
_DEFAULT_MAX_BINS = 8


def format_histogram_bound(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3g}"


def resolve_histogram_bin_count(
    non_null_count: int,
    *,
    min_bins: int = _DEFAULT_MIN_BINS,
    max_bins: int = _DEFAULT_MAX_BINS,
) -> int:
    if non_null_count <= 0:
        raise ValueError("non_null_count must be greater than 0")
    if min_bins < 1 or max_bins < min_bins:
        raise ValueError("Invalid histogram bin bounds.")
    return min(max_bins, max(min_bins, math.ceil(math.sqrt(non_null_count))))


def build_histogram_edges(
    lower: float,
    upper: float,
    *,
    non_null_count: int,
    min_bins: int = _DEFAULT_MIN_BINS,
    max_bins: int = _DEFAULT_MAX_BINS,
) -> np.ndarray | None:
    if not math.isfinite(lower) or not math.isfinite(upper):
        return None
    if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-9):
        return None
    bin_count = resolve_histogram_bin_count(
        non_null_count,
        min_bins=min_bins,
        max_bins=max_bins,
    )
    return np.linspace(lower, upper, num=bin_count + 1)


def initialize_histogram_counts(edges: np.ndarray) -> np.ndarray:
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("Histogram edges must contain at least 2 values.")
    return np.zeros(len(edges) - 1, dtype=np.int64)


def update_histogram_counts(
    counts: np.ndarray,
    edges: np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> int:
    numeric_values = np.asarray(values, dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        return 0

    bucket_indexes = np.searchsorted(edges, finite_values, side="right") - 1
    bucket_indexes = np.clip(bucket_indexes, 0, len(counts) - 1)
    counts += np.bincount(bucket_indexes, minlength=len(counts))
    return int(finite_values.size)


def single_value_histogram(value: float, count: int) -> list[tuple[str, int]]:
    return [(format_histogram_bound(value), int(count))]


def serialize_histogram(
    edges: np.ndarray,
    counts: np.ndarray,
) -> list[tuple[str, int]]:
    if len(edges) != len(counts) + 1:
        raise ValueError("Histogram edges/counts shape mismatch.")
    return [
        (
            f"{format_histogram_bound(float(edges[index]))}..{format_histogram_bound(float(edges[index + 1]))}",
            int(count),
        )
        for index, count in enumerate(counts.tolist())
    ]


def build_numeric_histogram(
    values: Sequence[float] | np.ndarray,
    *,
    min_bins: int = _DEFAULT_MIN_BINS,
    max_bins: int = _DEFAULT_MAX_BINS,
) -> list[tuple[str, int]]:
    numeric_values = np.asarray(values, dtype=float)
    finite_values = numeric_values[np.isfinite(numeric_values)]
    if finite_values.size == 0:
        return []

    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if math.isclose(lower, upper, rel_tol=1e-9, abs_tol=1e-9):
        return single_value_histogram(lower, int(finite_values.size))

    edges = build_histogram_edges(
        lower,
        upper,
        non_null_count=int(finite_values.size),
        min_bins=min_bins,
        max_bins=max_bins,
    )
    if edges is None:
        return []
    counts = initialize_histogram_counts(edges)
    update_histogram_counts(counts, edges, finite_values)
    return serialize_histogram(edges, counts)
