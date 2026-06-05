from __future__ import annotations

from pathlib import Path

import polars as pl
from rich.console import Group
from rich.text import Text

from tabcaddy.rendering.console import RenderProfile, resolve_render_profile

# Columns injected by the compile step; hidden unless --showmeta is set.
_META_COLUMNS = {"_source_file"}


def _df_to_table(
    df: pl.DataFrame,
    *,
    render: RenderProfile,
    show_meta: bool,
    title: str | None = None,
) -> object:
    columns = [c for c in df.columns if show_meta or c not in _META_COLUMNS]
    table = render.table(title=title, expand=True, show_lines=False)
    for col in columns:
        table.add_column(col, no_wrap=True)
    for row in df.select(columns).iter_rows():
        table.add_row(*[str(v) if v is not None else "" for v in row])
    return table


def build_file_head_view(
    df: pl.DataFrame,
    path: Path,
    *,
    render: RenderProfile | None = None,
    show_meta: bool = False,
) -> object:
    render = resolve_render_profile() if render is None else render
    return _df_to_table(df, render=render, show_meta=show_meta, title=str(path))


def build_folder_head_view(
    frames: list[tuple[Path, pl.DataFrame]],
    *,
    render: RenderProfile | None = None,
    show_meta: bool = False,
) -> object:
    render = resolve_render_profile() if render is None else render
    blocks: list[object] = []
    for path, df in frames:
        blocks.append(Text(str(path), style="cyan bold"))
        blocks.append(_df_to_table(df, render=render, show_meta=show_meta))
    return Group(*blocks)
