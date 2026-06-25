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
    value_strings = [str(value) for _, value in items]
    value_width = max(len(value) for value in value_strings)
    max_label_width = min(_MAX_LABEL_WIDTH, max(len(label) for label, _ in items))

    if max_width is None:
        label_width = max_label_width
        bar_width = requested_bar_width
        line_width = None
    else:
        line_width = max(1, max_width)
        # Reserve one trailing space before the count when there is room for
        # label/bar content, and squeeze the bar before the label.
        prefix_budget = max(0, line_width - value_width - 1)
        label_width = min(max_label_width, prefix_budget)

        remaining = prefix_budget - label_width
        bar_gap = 1 if label_width > 0 and remaining > 0 else 0
        bar_width = min(requested_bar_width, max(0, remaining - bar_gap))

    lines: list[str] = []
    for (label, value), value_text in zip(items, value_strings, strict=False):
        if line_width is not None and line_width <= len(value_text):
            lines.append(value_text[:line_width])
            continue

        label_display = _truncate_label(label, label_width).ljust(label_width)
        if bar_width > 0:
            filled = max(1, round((value / maximum) * bar_width)) if value else 0
            bar = fill * filled + " " * (bar_width - filled)
            prefix = f"{label_display} {bar}" if label_width > 0 else bar
        else:
            prefix = label_display if label_width > 0 else ""

        line = (
            f"{prefix} {value:>{value_width}}" if prefix else f"{value:>{value_width}}"
        )
        if line_width is not None and len(line) > line_width:
            line = line[:line_width]
        lines.append(line)
    return "\n".join(lines)
