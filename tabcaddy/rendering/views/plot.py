from __future__ import annotations

from pathlib import Path

from rich.console import Group
from rich.text import Text

from tabcaddy.plot.service import PlotResult, PlotRunResult
from tabcaddy.rendering.charts.axis_formatters import format_epoch_seconds_utc
from tabcaddy.rendering.charts.axis_formatters import format_numeric_axis
from tabcaddy.rendering.charts.line_chart import get_line_series_ansi_color
from tabcaddy.rendering.charts.line_chart import get_line_series_rich_style
from tabcaddy.rendering.charts.line_chart import render_line_chart
from tabcaddy.rendering.charts.scatter_chart import render_scatter_chart
from tabcaddy.rendering.console import RenderProfile, resolve_render_profile


_WIDTH_BREAKPOINT = 100
_WIDE_SCALING_FACTOR = 0.8
_MAX_AUTO_CHART_WIDTH = 480
_CHART_AXIS_OVERHEAD = 11


def _resolve_chart_width(*, console_width: int | None) -> int:
    if console_width is None:
        return _WIDTH_BREAKPOINT

    console_width = max(2, console_width)
    if console_width <= _WIDTH_BREAKPOINT:
        return console_width

    scaled_width = _WIDTH_BREAKPOINT + int(
        (console_width - _WIDTH_BREAKPOINT) * _WIDE_SCALING_FACTOR
    )
    return min(scaled_width, _MAX_AUTO_CHART_WIDTH)


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

    metadata = _build_multi_y_plot_metadata_table(results_by_y, render=render)

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


def _summarize_int(values: list[int]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return str(values[0])
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return str(minimum)
    return f"{minimum}..{maximum}"


def _summarize_bool(values: list[bool]) -> str:
    if not values:
        return "n/a"
    unique = set(values)
    if len(unique) > 1:
        return "mixed"
    return "yes" if values[0] else "no"


def _summarize_str(values: list[str]) -> str:
    if not values:
        return "n/a"
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    return " / ".join(unique)


def _series_cell_value(
    plot_run_result: PlotRunResult,
    selector,
    *,
    summarizer,
) -> str:
    series_values = [selector(file_plot.result) for file_plot in plot_run_result.plots]
    return summarizer(series_values)


def _build_multi_y_plot_metadata_table(
    results_by_y: list[tuple[str, PlotRunResult]],
    *,
    render: RenderProfile,
) -> object:
    metadata = render.table(title="Plot", expand=False)
    metadata.add_column("Field", style="cyan")
    for index, (y_column, _) in enumerate(results_by_y):
        metadata.add_column(y_column, style=get_line_series_rich_style(index))

    def add_series_row(
        label: str,
        selector,
        *,
        summarizer,
    ) -> None:
        metadata.add_row(
            label,
            *[
                _series_cell_value(plot_run_result, selector, summarizer=summarizer)
                for _, plot_run_result in results_by_y
            ],
        )

    add_series_row("Kind", lambda result: result.chart_kind, summarizer=_summarize_str)
    add_series_row("X", lambda result: result.x_column, summarizer=_summarize_str)
    add_series_row(
        "Input rows",
        lambda result: result.row_count,
        summarizer=_summarize_int,
    )
    add_series_row(
        "Plotted rows",
        lambda result: result.plotted_rows,
        summarizer=_summarize_int,
    )
    add_series_row(
        "Dropped rows",
        lambda result: result.dropped_rows,
        summarizer=_summarize_int,
    )
    add_series_row(
        "Duplicate x",
        lambda result: result.duplicate_x_count,
        summarizer=_summarize_int,
    )
    add_series_row(
        "Sorted x",
        lambda result: result.sorted_x,
        summarizer=_summarize_bool,
    )
    add_series_row(
        "Auto-sorted",
        lambda result: result.auto_sorted,
        summarizer=_summarize_bool,
    )
    add_series_row(
        "Aggregated",
        lambda result: result.aggregated,
        summarizer=_summarize_bool,
    )
    add_series_row(
        "Interpolation",
        lambda result: result.line_interpolation or "n/a",
        summarizer=_summarize_str,
    )
    add_series_row(
        "X Axis Kind",
        lambda result: result.x_axis_kind,
        summarizer=_summarize_str,
    )
    add_series_row(
        "X Time Unit",
        lambda result: result.x_axis_time_unit or "n/a",
        summarizer=_summarize_str,
    )
    add_series_row(
        "X Time Zone",
        lambda result: result.x_axis_timezone or "n/a",
        summarizer=_summarize_str,
    )
    return metadata


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
    if result.chart_kind == "scatter":
        metadata.add_row("Outliers", str(len(result.scatter_outlier_points)))
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
    )
    chart_width = max(2, target_width - _CHART_AXIS_OVERHEAD)

    x_tick_formatter = None
    if (
        result.x_axis_kind == "temporal"
        and result.x_axis_time_unit == "epoch_seconds"
        and result.x_axis_timezone == "UTC"
    ):
        x_tick_formatter = format_epoch_seconds_utc
    elif result.x_axis_kind == "numeric":
        x_tick_formatter = format_numeric_axis

    if result.chart_kind == "line":
        chart = render_line_chart(
            result.line_values,
            x_values=result.line_x_values,
            interpolation=result.line_interpolation or "linear",
            x_tick_formatter=x_tick_formatter,
            color=line_color_ansi,
            ascii_only=render.ascii_only,
            width=chart_width,
        )
    else:
        chart = render_scatter_chart(
            result.scatter_inlier_points or result.scatter_points,
            outlier_points=result.scatter_outlier_points,
            point="." if render.ascii_only else "●",
            outlier_point="*" if render.ascii_only else "◦",
            y_axis_char="|" if render.ascii_only else "│",
            x_axis_char="-" if render.ascii_only else "─",
            axis_corner_char="+" if render.ascii_only else "└",
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
