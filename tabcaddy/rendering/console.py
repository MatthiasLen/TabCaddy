from __future__ import annotations

from rich.console import Console


def create_console(*, record: bool = False, width: int | None = None) -> Console:
    return Console(record=record, width=width, soft_wrap=True)
