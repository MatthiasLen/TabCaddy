from __future__ import annotations

from collections.abc import Sequence
import math

import asciichartpy


def _resample_by_x(
    values: Sequence[float],
    x_values: Sequence[float],
    *,
    target_points: int,
) -> list[float]:
    if len(values) < 2 or len(values) != len(x_values):
        return list(values)

    points = sorted(zip(x_values, values, strict=False), key=lambda item: item[0])
    min_x = points[0][0]
    max_x = points[-1][0]
    if not math.isfinite(min_x) or not math.isfinite(max_x):
        return [value for _, value in points]
    if math.isclose(min_x, max_x, rel_tol=1e-9, abs_tol=1e-9):
        return [value for _, value in points]

    sampled: list[float] = []
    segment = 0
    for index in range(target_points):
        ratio = index / (target_points - 1)
        target_x = min_x + ratio * (max_x - min_x)

        while segment + 1 < len(points) and target_x > points[segment + 1][0]:
            segment += 1

        if segment + 1 >= len(points):
            sampled.append(float(points[-1][1]))
            continue

        x0, y0 = points[segment]
        x1, y1 = points[segment + 1]
        if math.isclose(x0, x1, rel_tol=1e-12, abs_tol=1e-12):
            sampled.append(float(y1))
            continue

        local_ratio = (target_x - x0) / (x1 - x0)
        sampled.append(float(y0 + (y1 - y0) * local_ratio))

    return sampled


def render_line_chart(
    values: Sequence[float],
    *,
    x_values: Sequence[float] | None = None,
    height: int = 10,
    width: int = 60,
) -> str:
    if not values:
        return ""

    series = list(values)
    if x_values is not None and len(series) == len(x_values) and len(series) > 1:
        series = _resample_by_x(series, list(x_values), target_points=max(2, width))

    return asciichartpy.plot(series, {"height": height})
