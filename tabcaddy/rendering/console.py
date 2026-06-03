from __future__ import annotations

from rich.console import Console


def create_console(
    *,
    record: bool = False,
    width: int | None = None,
    legacy_windows: bool | None = None,
) -> Console:
    options: dict[str, object] = {
        "record": record,
        "width": width,
        "soft_wrap": True,
    }
    if legacy_windows is not None:
        options["legacy_windows"] = legacy_windows
    return Console(**options)
