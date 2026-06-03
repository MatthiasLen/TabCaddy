from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.rendering.charts.bar_chart import render_bar_chart


def build_summary_view(analysis: DatasetAnalysis):
    blocks: list[object] = []

    metadata = Table(title="Metadata", expand=False)
    metadata.add_column("Field", style="cyan")
    metadata.add_column("Value", style="white")
    metadata.add_row("Files", str(analysis.metadata.source_file_count))
    metadata.add_row("Rows", str(analysis.metadata.row_count))
    metadata.add_row("Columns", str(analysis.metadata.column_count))
    metadata.add_row("Schemas", str(len(analysis.schemas)))
    metadata.add_row("Created", analysis.metadata.created_at.isoformat())
    blocks.append(metadata)

    schema_table = Table(title="Schema Overview", expand=True)
    schema_table.add_column("Schema", style="cyan")
    schema_table.add_column("Files", justify="right")
    schema_table.add_column("Columns")
    for index, schema in enumerate(analysis.schemas, start=1):
        sample = ", ".join(column.name for column in schema.columns[:4])
        if len(schema.columns) > 4:
            sample += ", ..."
        schema_table.add_row(f"Schema {index}", str(schema.occurrence_count), sample)
    blocks.append(schema_table)

    distribution = render_bar_chart([(f"Schema {index}", schema.occurrence_count) for index, schema in enumerate(analysis.schemas, start=1)])
    if distribution:
        blocks.append(Panel(distribution, title="Schema Distribution", border_style="blue"))

    if analysis.statistics is not None:
        stats = Table(title="Statistics", expand=True)
        stats.add_column("Column", style="cyan")
        stats.add_column("Type")
        stats.add_column("Null %", justify="right")
        stats.add_column("Min")
        stats.add_column("Max")
        stats.add_column("Mean", justify="right")
        for name, column in analysis.statistics.columns.items():
            stats.add_row(
                name,
                column.dtype,
                f"{column.null_rate * 100:.1f}",
                "" if column.min_value is None else str(column.min_value),
                "" if column.max_value is None else str(column.max_value),
                "" if column.mean is None else f"{column.mean:.3f}",
            )
        blocks.append(stats)

        temporal = Table(title="Date Ranges", expand=True)
        temporal.add_column("Column", style="cyan")
        temporal.add_column("Min")
        temporal.add_column("Max")
        added = False
        for name, column in analysis.statistics.columns.items():
            if any(token in column.dtype for token in ("Date", "Datetime", "Time")):
                temporal.add_row(name, "" if column.min_value is None else str(column.min_value), "" if column.max_value is None else str(column.max_value))
                added = True
        if added:
            blocks.append(temporal)

        for name, column in list(analysis.statistics.columns.items())[:3]:
            if column.histogram:
                blocks.append(Panel(render_bar_chart(column.histogram, width=18), title=f"Histogram: {name}", border_style="blue"))

    if analysis.warnings:
        warning_text = Text("\n".join(f"- {warning}" for warning in analysis.warnings), style="yellow")
        blocks.append(Panel(warning_text, title="Warnings", border_style="yellow"))
    else:
        blocks.append(Panel("No warnings detected.", title="Warnings", border_style="green"))

    return Group(*blocks)
