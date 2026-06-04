from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


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


@dataclass(frozen=True)
class RenderProfile:
    ascii_only: bool = False

    @property
    def bar_fill(self) -> str:
        return "#" if self.ascii_only else "█"

    def table(self, **kwargs: Any) -> Table:
        return Table(**kwargs, **table_render_options(ascii_only=self.ascii_only))

    def panel(self, renderable: object, **kwargs: Any) -> Panel:
        return Panel(
            renderable,
            **kwargs,
            **panel_render_options(ascii_only=self.ascii_only),
        )


def console_supports_unicode(console: Console) -> bool:
    encoding = getattr(console.file, "encoding", None) or getattr(
        console, "encoding", None
    )
    if not encoding:
        return True
    try:
        "█┌".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def resolve_render_profile(console: Console | None = None) -> RenderProfile:
    if console is None:
        return RenderProfile()
    return RenderProfile(ascii_only=not console_supports_unicode(console))


def table_render_options(*, ascii_only: bool) -> dict[str, object]:
    if not ascii_only:
        return {}
    return {"box": box.ASCII, "safe_box": True}


def panel_render_options(*, ascii_only: bool) -> dict[str, object]:
    if not ascii_only:
        return {}
    return {"box": box.ASCII}
