from __future__ import annotations

from collections.abc import Sequence

import asciichartpy


def render_line_chart(values: Sequence[float], *, height: int = 10) -> str:
    if not values:
        return ""
    return asciichartpy.plot(list(values), {"height": height})
