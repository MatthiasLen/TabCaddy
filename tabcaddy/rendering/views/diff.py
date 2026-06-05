from __future__ import annotations

from rich.console import Group

from tabcaddy.domain.models import DiffLevel
from tabcaddy.domain.models import DiffReport
from tabcaddy.rendering.console import RenderProfile
from tabcaddy.rendering.console import resolve_render_profile


def build_diff_view(
    report: DiffReport,
    *,
    level: DiffLevel = DiffLevel.FULL,
    render: RenderProfile | None = None,
):
    render = resolve_render_profile() if render is None else render
    blocks: list[object] = []
    blocks.append(
        _build_section(
            "Metadata Changes", report.metadata_changes, "cyan", render=render
        )
    )
    if level == DiffLevel.FULL:
        blocks.append(
            _build_section(
                "Schema Changes", report.schema_changes, "blue", render=render
            )
        )
    if level in {DiffLevel.STATISTICS, DiffLevel.FULL}:
        blocks.append(
            _build_section(
                "Statistics Changes",
                report.statistics_changes,
                "green",
                render=render,
            )
        )
    blocks.append(_build_section("Warnings", report.warnings, "yellow", render=render))
    return Group(*blocks)


def _build_section(
    title: str,
    lines: list[str],
    color: str,
    *,
    render: RenderProfile,
):
    if not lines:
        return render.panel(
            "No changes.",
            title=title,
            border_style=color,
        )
    table = render.table(show_header=False, expand=True)
    table.add_column("Change", style=color)
    for line in lines:
        table.add_row(line)
    return render.panel(
        table,
        title=title,
        border_style=color,
    )
