from __future__ import annotations

from collections.abc import Sequence
import math
import re
from typing import Callable, Literal

import asciichartpy
import numpy as np
from scipy.interpolate import interp1d


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Keep ANSI and Rich styles in the same order so chart and metadata share a palette.
LINE_SERIES_ANSI_COLORS = [
    asciichartpy.cyan,
    asciichartpy.magenta,
    asciichartpy.green,
    asciichartpy.yellow,
    asciichartpy.blue,
    asciichartpy.red,
]
LINE_SERIES_RICH_STYLES = ["cyan", "magenta", "green", "yellow", "blue", "red"]

_ASCII_CHART_SYMBOLS = ["+", ":", ".", ".", "-", "\\", "/", "\\", "/", "|"]
_UNICODE_AXIS_MARKERS = {"┤", "┼"}
_ASCII_AXIS_MARKERS = {"|", "+"}


def get_line_series_ansi_color(index: int) -> str:
    return LINE_SERIES_ANSI_COLORS[index % len(LINE_SERIES_ANSI_COLORS)]


def get_line_series_rich_style(index: int) -> str:
    return LINE_SERIES_RICH_STYLES[index % len(LINE_SERIES_RICH_STYLES)]


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


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
    color: str | None = None,
    ascii_only: bool = False,
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

    config: dict[str, object] = {"height": height}
    if color is not None:
        config["colors"] = [color]
    if ascii_only:
        config["symbols"] = _ASCII_CHART_SYMBOLS
    chart = asciichartpy.plot(series, config)

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

    plain_lines = [_strip_ansi(line) for line in lines]

    axis_markers = _ASCII_AXIS_MARKERS if ascii_only else _UNICODE_AXIS_MARKERS
    axis_index = next(
        (index for index, ch in enumerate(plain_lines[0]) if ch in axis_markers),
        None,
    )
    if axis_index is None:
        return "\n".join([chart, f"{left_label}  {right_label}"])

    plot_start = axis_index + 1
    plot_width = max(0, max(len(line) - plot_start for line in plain_lines))
    gap = max(1, plot_width - len(left_label) - len(right_label))
    footer = " " * plot_start + left_label + (" " * gap) + right_label
    footer_corner = "+" if ascii_only else "└"
    footer_bar = "-" if ascii_only else "─"
    footer_line = (
        " " * (plot_start - 1) + footer_corner + footer_bar * (len(footer) - plot_start)
    )

    return "\n".join([chart, footer_line, footer])
