from __future__ import annotations

from collections.abc import Sequence


_MAX_LABEL_WIDTH = 35


def _truncate_label(label: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(label) <= width:
        return label
    if width <= 3:
        return label[:width]
    return f"{label[: width - 3]}..."


def render_bar_chart(
    items: Sequence[tuple[str, int]],
    *,
    width: int = 24,
    fill: str = "█",
    max_width: int | None = None,
) -> str:
    if not items:
        return ""
    maximum = max(value for _, value in items) or 1
    requested_bar_width = max(1, width)
    value_width = max(len(str(value)) for _, value in items)
    max_label_width = min(_MAX_LABEL_WIDTH, max(len(label) for label, _ in items))

    if max_width is None:
        label_width = max_label_width
        bar_width = requested_bar_width
    else:
        line_width = max(6, max_width)
        # Row format: <label> <bar> <count>
        # Keep labels single-line, prefer up to 35 chars, and squeeze bars first.
        content_budget = max(1, line_width - value_width - 2)
        label_width = min(max_label_width, max(1, content_budget - 1))
        bar_width = max(1, min(requested_bar_width, content_budget - label_width))

    lines: list[str] = []
    for label, value in items:
        label_display = _truncate_label(label, label_width).ljust(label_width)
        filled = (
            max(1, round((value / maximum) * bar_width))
            if value and bar_width > 0
            else 0
        )
        bar = fill * filled + " " * (bar_width - filled)
        lines.append(f"{label_display} {bar} {value:>{value_width}}")
    return "\n".join(lines)
