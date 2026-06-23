from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.text import Text

from tabcaddy.plot.service import PlotResult, PlotRunResult
from tabcaddy.rendering.charts.axis_formatters import format_epoch_seconds_utc
from tabcaddy.rendering.charts.line_chart import get_line_series_ansi_color
from tabcaddy.rendering.charts.line_chart import get_line_series_rich_style
from tabcaddy.rendering.charts.line_chart import render_line_chart
from tabcaddy.rendering.charts.scatter_chart import render_scatter_chart
from tabcaddy.rendering.console import RenderProfile, resolve_render_profile


_NARROW_CONSOLE_MAX_WIDTH = 80
_STANDARD_MIN_WIDTH = 60
_SPARSE_POINT_THRESHOLD = 12
_WIDE_SCALING_FACTOR = 0.6
_MAX_AUTOMATED_WIDTH = 120
_CHART_AXIS_OVERHEAD = 11


def _resolve_chart_width(*, console_width: int | None, point_count: int) -> int:
    if console_width is None:
        return _STANDARD_MIN_WIDTH

    console_width = max(2, console_width)
    if console_width <= _NARROW_CONSOLE_MAX_WIDTH:
        return console_width

    scaled_width = min(
        _MAX_AUTOMATED_WIDTH,
        _NARROW_CONSOLE_MAX_WIDTH
        + int((console_width - _NARROW_CONSOLE_MAX_WIDTH) * _WIDE_SCALING_FACTOR),
    )

    if point_count <= _SPARSE_POINT_THRESHOLD:
        return min(_STANDARD_MIN_WIDTH, scaled_width)

    point_based_cap = max(_STANDARD_MIN_WIDTH, min(point_count, _MAX_AUTOMATED_WIDTH))
    return min(scaled_width, point_based_cap)


def _build_warning_panel(warnings: list[str], *, render: RenderProfile) -> object:
    warning_text = Text(
        "\n".join(f"- {warning}" for warning in warnings), style="yellow"
    )
    return render.panel(
        warning_text,
        title="Warnings",
        border_style="yellow",
    )


def build_plot_view(
    result: PlotRunResult,
    *,
    render: RenderProfile | None = None,
    y_value_style: str | None = None,
    line_color_ansi: str | None = None,
    include_metadata: bool = True,
    chart_title: str = "Chart",
) -> object:
    render = resolve_render_profile() if render is None else render

    if len(result.plots) == 1 and result.skipped_files == 0 and not result.warnings:
        return _build_single_plot_view(
            result.plots[0].result,
            render=render,
            y_value_style=y_value_style,
            line_color_ansi=line_color_ansi,
            include_metadata=include_metadata,
            chart_title=chart_title,
        )

    summary = render.table(title="Folder Plot Summary", expand=False)
    summary.add_column("Field", style="cyan")
    summary.add_column("Value", style="white")
    summary.add_row("Total files", str(result.total_files))
    summary.add_row("Plotted files", str(result.plotted_files))
    summary.add_row("Skipped files", str(result.skipped_files))

    blocks: list[object] = []
    if include_metadata:
        blocks.append(summary)

    for index, file_plot in enumerate(result.plots, start=1):
        file_name = Path(file_plot.path).name
        blocks.append(Text(f"File {index}: {file_name}", style="blue"))
        file_group = _build_single_plot_view(
            file_plot.result,
            render=render,
            y_value_style=y_value_style,
            line_color_ansi=line_color_ansi,
            include_metadata=include_metadata,
            chart_title=chart_title,
        )
        blocks.append(file_group)

    if result.warnings:
        blocks.append(_build_warning_panel(result.warnings, render=render))

    return Group(*blocks)


def build_multi_y_plot_view(
    results_by_y: list[tuple[str, PlotRunResult]],
    *,
    render: RenderProfile | None = None,
) -> object:
    render = resolve_render_profile() if render is None else render

    if not results_by_y:
        return "No plot data"

    first_plot_result = results_by_y[0][1].plots[0].result
    metadata = _build_plot_metadata_table(
        first_plot_result,
        render=render,
        y_value=_build_multi_y_value_text(results_by_y),
    )

    blocks: list[object] = []
    blocks.append(metadata)
    for index, (y_column, result) in enumerate(results_by_y, start=1):
        style = get_line_series_rich_style(index - 1)
        ansi_color = get_line_series_ansi_color(index - 1)
        blocks.append(
            build_plot_view(
                result,
                render=render,
                y_value_style=style,
                line_color_ansi=ansi_color,
                include_metadata=False,
                chart_title=y_column,
            )
        )

    return Group(*blocks)


def _build_multi_y_value_text(results_by_y: list[tuple[str, PlotRunResult]]) -> Text:
    y_text = Text()
    for index, (y_column, _) in enumerate(results_by_y):
        if index > 0:
            y_text.append("\n")
        y_text.append(y_column, style=get_line_series_rich_style(index))
    return y_text


def _build_plot_metadata_table(
    result: PlotResult,
    *,
    render: RenderProfile,
    y_value: str | Text,
) -> object:
    metadata = render.table(title="Plot", expand=False)
    metadata.add_column("Field", style="cyan")
    metadata.add_column("Value", style="white")
    metadata.add_row("Kind", result.chart_kind)
    metadata.add_row("X", result.x_column)
    metadata.add_row("Y", y_value)
    metadata.add_row("Input rows", str(result.row_count))
    metadata.add_row("Plotted rows", str(result.plotted_rows))
    metadata.add_row("Dropped rows", str(result.dropped_rows))
    metadata.add_row("Duplicate x", str(result.duplicate_x_count))
    metadata.add_row("Sorted x", "yes" if result.sorted_x else "no")
    metadata.add_row("Auto-sorted", "yes" if result.auto_sorted else "no")
    metadata.add_row("Aggregated", "yes" if result.aggregated else "no")
    if result.line_interpolation is not None:
        metadata.add_row("Interpolation", result.line_interpolation)
    metadata.add_row("X Axis Kind", result.x_axis_kind)
    if result.x_axis_time_unit is not None:
        metadata.add_row("X Time Unit", result.x_axis_time_unit)
    if result.x_axis_timezone is not None:
        metadata.add_row("X Time Zone", result.x_axis_timezone)
    return metadata


def _build_single_plot_view(
    result: PlotResult,
    *,
    render: RenderProfile,
    y_value_style: str | None = None,
    line_color_ansi: str | None = None,
    include_metadata: bool = True,
    chart_title: str = "Chart",
) -> Group:
    metadata = _build_plot_metadata_table(
        result,
        render=render,
        y_value=Text(result.y_column, style=y_value_style),
    )

    target_width = _resolve_chart_width(
        console_width=render.console_width,
        point_count=result.plotted_rows,
    )
    chart_width = max(2, target_width - _CHART_AXIS_OVERHEAD)

    x_tick_formatter = None
    if (
        result.x_axis_kind == "temporal"
        and result.x_axis_time_unit == "epoch_seconds"
        and result.x_axis_timezone == "UTC"
    ):
        x_tick_formatter = format_epoch_seconds_utc

    if result.chart_kind == "line":
        chart = render_line_chart(
            result.line_values,
            x_values=result.line_x_values,
            interpolation=result.line_interpolation or "linear",
            x_tick_formatter=x_tick_formatter,
            color=line_color_ansi,
            width=chart_width,
        )
    else:
        chart = render_scatter_chart(
            result.scatter_points,
            point="#" if render.ascii_only else "•",
            x_tick_formatter=x_tick_formatter,
            width=chart_width,
        )

    caption_style = y_value_style or "white"
    chart_renderable = Text.from_ansi(chart or "No chart data")
    blocks: list[object] = [Text(chart_title, style=caption_style), chart_renderable]
    if include_metadata:
        blocks.insert(0, metadata)
    if result.warnings:
        blocks.append(_build_warning_panel(result.warnings, render=render))
    return Group(*blocks)
