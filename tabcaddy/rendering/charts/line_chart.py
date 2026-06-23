from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Callable, Literal

import asciichartpy
import numpy as np
from scipy.interpolate import interp1d


def _resample_by_x(
    values: Sequence[float],
    x_values: Sequence[float],
    *,
    target_points: int,
    interpolation: Literal["linear", "nearest"] = "linear",
) -> list[float]:
    if len(values) < 2 or len(values) != len(x_values):
        return list(values)

    points = sorted(zip(x_values, values, strict=False), key=lambda item: item[0])
    xs = np.asarray([x for x, _ in points], dtype=np.float64)
    ys = np.asarray([y for _, y in points], dtype=np.float64)

    min_x = float(xs[0])
    max_x = float(xs[-1])
    if not math.isfinite(min_x) or not math.isfinite(max_x):
        return ys.tolist()
    if math.isclose(min_x, max_x, rel_tol=1e-9, abs_tol=1e-9):
        return ys.tolist()

    target_x = np.linspace(min_x, max_x, num=target_points, dtype=np.float64)
    interpolator = interp1d(
        xs,
        ys,
        kind=interpolation,
        assume_sorted=True,
        bounds_error=False,
        fill_value=(float(ys[0]), float(ys[-1])),
    )
    return np.asarray(interpolator(target_x), dtype=np.float64).tolist()


def render_line_chart(
    values: Sequence[float],
    *,
    x_values: Sequence[float] | None = None,
    interpolation: Literal["linear", "nearest"] = "linear",
    x_tick_formatter: Callable[[float], str] | None = None,
    height: int = 10,
    width: int = 60,
) -> str:
    if not values:
        return ""

    series = list(values)
    numeric_x_values: list[float] | None = None
    if x_values is not None and len(series) == len(x_values):
        numeric_x_values = list(x_values)
        if len(series) > 1:
            series = _resample_by_x(
                series,
                numeric_x_values,
                target_points=max(2, width),
                interpolation=interpolation,
            )

    chart = asciichartpy.plot(series, {"height": height})

    if (
        x_tick_formatter is None
        or numeric_x_values is None
        or len(numeric_x_values) < 2
    ):
        return chart

    min_x = min(numeric_x_values)
    max_x = max(numeric_x_values)
    if not math.isfinite(min_x) or not math.isfinite(max_x):
        return chart

    left_label = x_tick_formatter(min_x)
    right_label = x_tick_formatter(max_x)

    lines = chart.splitlines()
    if not lines:
        return chart

    axis_index = next(
        (index for index, ch in enumerate(lines[0]) if ch in {"┤", "┼"}),
        None,
    )
    if axis_index is None:
        return "\n".join([chart, f"{left_label}  {right_label}"])

    plot_start = axis_index + 1
    plot_width = max(0, max(len(line) - plot_start for line in lines))
    gap = max(1, plot_width - len(left_label) - len(right_label))
    footer = " " * plot_start + left_label + (" " * gap) + right_label
    footer_line = " " * (plot_start - 1) + "└" + "─" * (len(footer) - plot_start)

    return "\n".join([chart, footer_line, footer])
