from __future__ import annotations

from collections.abc import Sequence
import itertools
import math
from typing import Callable


def _format_axis(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    if math.isclose(value, round(value), rel_tol=1e-9, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.3g}"


def render_scatter_chart(
    points: Sequence[tuple[float, float]],
    *,
    outlier_points: Sequence[tuple[float, float]] | None = None,
    width: int = 48,
    height: int = 12,
    point: str = "●",
    outlier_point: str = "◦",
    y_axis_char: str = "│",
    x_axis_char: str = "─",
    axis_corner_char: str = "└",
    x_tick_formatter: Callable[[float], str] | None = None,
) -> str:
    outlier_points = outlier_points or ()
    if not points and not outlier_points:
        return ""

    width = max(2, width)
    height = max(2, height)

    finite_points = [
        (x, y)
        for x, y in itertools.chain(points, outlier_points)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if not finite_points:
        return ""

    x_values = [x for x, _ in finite_points]
    y_values = [y for _, y in finite_points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    label_min_x, label_max_x = min_x, max_x
    label_min_y, label_max_y = min_y, max_y

    if math.isclose(min_x, max_x, rel_tol=1e-9, abs_tol=1e-9):
        min_x -= 0.5
        max_x += 0.5
    if math.isclose(min_y, max_y, rel_tol=1e-9, abs_tol=1e-9):
        min_y -= 0.5
        max_y += 0.5

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def plot_points(values: Sequence[tuple[float, float]], marker: str) -> None:
        for x, y in values:
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            x_pos = int(round((x - min_x) / (max_x - min_x) * (width - 1)))
            y_pos = int(round((y - min_y) / (max_y - min_y) * (height - 1)))
            row = height - 1 - max(0, min(height - 1, y_pos))
            col = max(0, min(width - 1, x_pos))
            grid[row][col] = marker

    plot_points(points, point)
    plot_points(outlier_points, outlier_point)

    label_width = max(len(_format_axis(label_min_y)), len(_format_axis(label_max_y)), 6)
    lines: list[str] = []
    for index, row in enumerate(grid):
        y_value = label_max_y - (index / (height - 1)) * (label_max_y - label_min_y)
        lines.append(
            f"{_format_axis(y_value):>{label_width}} {y_axis_char}{''.join(row)}"
        )

    lines.append(" " * (label_width + 1) + axis_corner_char + x_axis_char * width)
    min_x_label = (
        x_tick_formatter(label_min_x)
        if x_tick_formatter is not None
        else _format_axis(label_min_x)
    )
    max_x_label = (
        x_tick_formatter(label_max_x)
        if x_tick_formatter is not None
        else _format_axis(label_max_x)
    )
    lines.append(
        " " * (label_width + 2)
        + min_x_label
        + " " * max(1, width - len(min_x_label) - len(max_x_label))
        + max_x_label
    )
    return "\n".join(lines)
