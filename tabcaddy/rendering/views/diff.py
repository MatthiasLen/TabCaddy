from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from tabcaddy.domain.models import DiffReport


def build_diff_view(report: DiffReport):
    blocks: list[object] = []
    blocks.append(_build_section("Metadata Changes", report.metadata_changes, "cyan"))
    blocks.append(_build_section("Schema Changes", report.schema_changes, "blue"))
    blocks.append(_build_section("Statistics Changes", report.statistics_changes, "green"))
    blocks.append(_build_section("Warnings", report.warnings, "yellow"))
    return Group(*blocks)


def _build_section(title: str, lines: list[str], color: str):
    if not lines:
        return Panel("No changes.", title=title, border_style=color)
    table = Table(show_header=False, expand=True)
    table.add_column("Change", style=color)
    for line in lines:
        table.add_row(line)
    return Panel(table, title=title, border_style=color)
