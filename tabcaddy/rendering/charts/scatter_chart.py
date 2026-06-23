from __future__ import annotations

from collections.abc import Sequence
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
    width: int = 48,
    height: int = 12,
    point: str = "•",
    x_tick_formatter: Callable[[float], str] | None = None,
) -> str:
    if not points:
        return ""

    width = max(2, width)
    height = max(2, height)

    finite_points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if not finite_points:
        return ""

    x_values = [x for x, _ in finite_points]
    y_values = [y for _, y in finite_points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    if math.isclose(min_x, max_x, rel_tol=1e-9, abs_tol=1e-9):
        min_x -= 0.5
        max_x += 0.5
    if math.isclose(min_y, max_y, rel_tol=1e-9, abs_tol=1e-9):
        min_y -= 0.5
        max_y += 0.5

    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x, y in finite_points:
        x_pos = int(round((x - min_x) / (max_x - min_x) * (width - 1)))
        y_pos = int(round((y - min_y) / (max_y - min_y) * (height - 1)))
        row = height - 1 - max(0, min(height - 1, y_pos))
        col = max(0, min(width - 1, x_pos))
        grid[row][col] = point

    label_width = max(len(_format_axis(min_y)), len(_format_axis(max_y)), 6)
    lines: list[str] = []
    for index, row in enumerate(grid):
        y_value = max_y - (index / (height - 1)) * (max_y - min_y)
        lines.append(f"{_format_axis(y_value):>{label_width}} |{''.join(row)}")

    lines.append(" " * (label_width + 1) + "+" + "-" * width)
    min_x_label = (
        x_tick_formatter(min_x) if x_tick_formatter is not None else _format_axis(min_x)
    )
    max_x_label = (
        x_tick_formatter(max_x) if x_tick_formatter is not None else _format_axis(max_x)
    )
    lines.append(
        " " * (label_width + 2)
        + min_x_label
        + " " * max(1, width - len(min_x_label) - len(max_x_label))
        + max_x_label
    )
    return "\n".join(lines)
