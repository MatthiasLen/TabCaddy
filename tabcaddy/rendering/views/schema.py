from __future__ import annotations

from rich.console import Group

from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.analysis.schema import FileSchemaRecord
from tabcaddy.analysis.schema import schema_type_changes
from tabcaddy.rendering.charts.bar_chart import render_bar_chart
from tabcaddy.rendering.console import RenderProfile
from tabcaddy.rendering.console import resolve_render_profile


def build_schema_view(
    analysis: DatasetAnalysis,
    files: list[FileSchemaRecord],
    *,
    render: RenderProfile | None = None,
):
    render = resolve_render_profile() if render is None else render
    blocks: list[object] = []

    schemas = render.table(title="Schema Groups", expand=True)
    schemas.add_column("Schema", style="cyan", min_width=4)
    schemas.add_column("Files", justify="right", min_width=4)
    schemas.add_column("Hash", min_width=4)
    schemas.add_column("Columns", no_wrap=False)

    for index, schema in enumerate(analysis.schemas, start=1):
        columns = ", ".join(
            f"{column.name}:{column.dtype}" for column in schema.columns
        )
        schemas.add_row(
            f"Schema {index}", str(schema.occurrence_count), schema.hash[:12], columns
        )
    blocks.append(schemas)

    distribution = render_bar_chart(
        [
            (f"Schema {index}", schema.occurrence_count)
            for index, schema in enumerate(analysis.schemas, start=1)
        ],
        fill=render.bar_fill,
    )

    if distribution:
        blocks.append(
            render.panel(
                distribution,
                title="Occurrence Distribution",
                border_style="blue",
            )
        )

    drift = schema_type_changes(analysis.schemas)
    if drift:
        changes = render.table(title="Type Changes", expand=True)
        changes.add_column("Column", style="cyan")
        changes.add_column("Observed Types")
        for name, dtypes in sorted(drift.items()):
            changes.add_row(name, ", ".join(sorted(dtypes)))
        blocks.append(changes)

    if analysis.schemas:
        dominant_hash = analysis.schemas[0].hash
        violations = [record for record in files if record.schema_hash != dominant_hash]
        if violations:
            violating = render.table(
                title="Files Violating Dominant Schema", expand=True
            )
            violating.add_column("File", style="cyan")
            violating.add_column("Schema")
            violating.add_column("Rows", justify="right")
            for record in violations[:20]:
                violating.add_row(
                    record.relative_path.as_posix(),
                    record.schema_hash[:12],
                    str(record.row_count),
                )
            blocks.append(violating)

    return Group(*blocks)
