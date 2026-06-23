from __future__ import annotations

from rich.console import Group
from rich.text import Text

from tabcaddy.plot import PlotResult
from tabcaddy.rendering.charts.line_chart import render_line_chart
from tabcaddy.rendering.charts.scatter_chart import render_scatter_chart
from tabcaddy.rendering.console import RenderProfile, resolve_render_profile


def build_plot_view(
    result: PlotResult,
    *,
    render: RenderProfile | None = None,
) -> object:
    render = resolve_render_profile() if render is None else render

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

    if result.chart_kind == "line":
        chart = render_line_chart(result.line_values, x_values=result.line_x_values)
    else:
        chart = render_scatter_chart(
            result.scatter_points,
            point="#" if render.ascii_only else "•",
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
