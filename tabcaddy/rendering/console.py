from __future__ import annotations

from dataclasses import dataclass
import locale
import os
import sys
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table


class SafeConsole(Console):
    def print(self, *objects: Any, **kwargs: Any) -> None:
        try:
            super().print(*objects, **kwargs)
        except UnicodeEncodeError:
            stream = getattr(self, "file", sys.stdout)
            encoding = getattr(stream, "encoding", None) or "utf-8"
            sep = str(kwargs.get("sep", " "))
            end = str(kwargs.get("end", "\n"))
            rendered_parts: list[str] = []
            for obj in objects:
                try:
                    segments = self.render(obj, options=self.options)
                    plain = "".join(
                        segment.text for segment in Segment.strip_styles(segments)
                    )
                except Exception:
                    plain = str(obj)
                rendered_parts.append(plain)
            raw = sep.join(rendered_parts) + end
            safe_text = raw.encode(encoding, errors="replace").decode(
                encoding, errors="replace"
            )
            stream.write(safe_text)
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()


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
    return SafeConsole(**options)


@dataclass(frozen=True)
class RenderProfile:
    ascii_only: bool = False
    console_width: int | None = None

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
    # Rich's legacy Windows renderer may ultimately encode through the active
    # ANSI code page, which can reject box-drawing characters even when the
    # stream reports UTF-8. Prefer ASCII rendering in that mode.

    encoding = getattr(console.file, "encoding", None) or getattr(
        console, "encoding", None
    )

    # Some isolated runners on Windows expose a stream without an encoding.
    # Fall back to process defaults before deciding whether unicode is safe.
    if not encoding:
        encoding = locale.getpreferredencoding(False) or getattr(
            sys.stdout, "encoding", None
        )

    if not encoding:
        return os.name != "nt"
    try:
        "█┌".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def resolve_render_profile(console: Console | None = None) -> RenderProfile:
    if console is None:
        return RenderProfile()
    return RenderProfile(
        ascii_only=not console_supports_unicode(console),
        console_width=console.width,
    )


def table_render_options(*, ascii_only: bool) -> dict[str, object]:
    if not ascii_only:
        return {}
    return {"box": box.ASCII, "safe_box": True}


def panel_render_options(*, ascii_only: bool) -> dict[str, object]:
    if not ascii_only:
        return {}
    return {"box": box.ASCII}
