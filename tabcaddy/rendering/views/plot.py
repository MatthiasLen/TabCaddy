from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from rich.console import Group
from rich.text import Text

from tabcaddy.plot.service import PlotResult, PlotRunResult
from tabcaddy.rendering.charts.axis_formatters import format_epoch_seconds_utc
from tabcaddy.rendering.charts.line_chart import render_line_chart
from tabcaddy.rendering.charts.scatter_chart import render_scatter_chart
from tabcaddy.rendering.console import RenderProfile, resolve_render_profile


def build_plot_view(
    result: PlotRunResult,
    *,
    render: RenderProfile | None = None,
) -> object:
    render = resolve_render_profile() if render is None else render

    if len(result.plots) == 1 and result.skipped_files == 0 and not result.warnings:
        return _build_single_plot_view(result.plots[0].result, render=render)

    summary = render.table(title="Folder Plot Summary", expand=False)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Total files", str(result.total_files))
    summary.add_row("Plotted files", str(result.plotted_files))
    summary.add_row("Skipped files", str(result.skipped_files))

    blocks: list[object] = [summary]
    for index, file_plot in enumerate(result.plots, start=1):
        file_name = Path(file_plot.path).name
        file_group = _build_single_plot_view(file_plot.result, render=render)
        blocks.append(
            render.panel(
                file_group,
                title=f"File {index}: {file_name}",
                border_style="blue",
            )
        )

    if result.warnings:
        warning_text = Text(
            "\n".join(f"- {warning}" for warning in result.warnings), style="yellow"
        )
        blocks.append(
            render.panel(
                warning_text,
                title="Warnings",
                border_style="yellow",
            )
        )

    return Group(*blocks)


def _build_single_plot_view(
    result: PlotResult,
    *,
    render: RenderProfile,
) -> Group:
    metadata = render.table(title="Plot", expand=False)
    metadata.add_column("Field", style="cyan")
    metadata.add_column("Value", style="white")
    metadata.add_row("Kind", result.chart_kind)
    metadata.add_row("X", result.x_column)
    metadata.add_row("Y", result.y_column)
    metadata.add_row("Input rows", str(result.row_count))
    metadata.add_row("Plotted rows", str(result.plotted_rows))
    metadata.add_row("Dropped rows", str(result.dropped_rows))
    metadata.add_row("Duplicate x", str(result.duplicate_x_count))
    metadata.add_row("Sorted x", "yes" if result.sorted_x else "no")
    metadata.add_row("Auto-sorted", "yes" if result.auto_sorted else "no")
    metadata.add_row("Aggregated", "yes" if result.aggregated else "no")
    metadata.add_row("X Axis Kind", result.x_axis_kind)
    if result.x_axis_time_unit is not None:
        metadata.add_row("X Time Unit", result.x_axis_time_unit)
    if result.x_axis_timezone is not None:
        metadata.add_row("X Time Zone", result.x_axis_timezone)

    x_tick_formatter = None
    if (
        result.x_axis_kind == "temporal"
        and result.x_axis_time_unit == "epoch_seconds"
        and result.x_axis_timezone == "UTC"
    ):
        x_tick_formatter = format_epoch_seconds_utc

    if result.chart_kind == "line":
        chart = cast(Any, render_line_chart)(
            result.line_values,
            x_values=result.line_x_values,
            x_tick_formatter=x_tick_formatter,
        )
    else:
        chart = cast(Any, render_scatter_chart)(
            result.scatter_points,
            point="#" if render.ascii_only else "•",
            x_tick_formatter=x_tick_formatter,
        )

    chart_panel = render.panel(
        chart or "No chart data",
        title="Chart",
        border_style="blue",
    )

    blocks: list[object] = [metadata, chart_panel]
    if result.warnings:
        warning_text = Text(
            "\n".join(f"- {warning}" for warning in result.warnings), style="yellow"
        )
        blocks.append(
            render.panel(
                warning_text,
                title="Warnings",
                border_style="yellow",
            )
        )
    return Group(*blocks)
