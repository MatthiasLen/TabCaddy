from __future__ import annotations

from collections.abc import Sequence


def render_bar_chart(items: Sequence[tuple[str, int]], *, width: int = 24) -> str:
    if not items:
        return ""
    maximum = max(value for _, value in items) or 1
    lines: list[str] = []
    for label, value in items:
        filled = max(1, round((value / maximum) * width)) if value else 0
        bar = "█" * filled + " " * (width - filled)
        lines.append(f"{label:<18} {bar} {value}")
    return "\n".join(lines)
