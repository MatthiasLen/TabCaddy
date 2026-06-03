from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from tabcaddy.domain.models import DatasetAnalysis
from tabcaddy.rendering.charts.bar_chart import render_bar_chart
from tabcaddy.infrastructure.schema_analyzer import FileSchemaRecord
from tabcaddy.infrastructure.schema_analyzer import schema_type_changes


def build_schema_view(analysis: DatasetAnalysis, files: list[FileSchemaRecord]):
    blocks: list[object] = []

    schemas = Table(title="Schema Groups", expand=True)
    schemas.add_column("Schema", style="cyan")
    schemas.add_column("Files", justify="right")
    schemas.add_column("Hash")
    schemas.add_column("Columns")
    for index, schema in enumerate(analysis.schemas, start=1):
        columns = ", ".join(f"{column.name}:{column.dtype}" for column in schema.columns[:4])
        if len(schema.columns) > 4:
            columns += ", ..."
        schemas.add_row(f"Schema {index}", str(schema.occurrence_count), schema.hash[:12], columns)
    blocks.append(schemas)

    distribution = render_bar_chart([(f"Schema {index}", schema.occurrence_count) for index, schema in enumerate(analysis.schemas, start=1)])
    if distribution:
        blocks.append(Panel(distribution, title="Occurrence Distribution", border_style="blue"))

    drift = schema_type_changes(analysis.schemas)
    if drift:
        changes = Table(title="Type Changes", expand=True)
        changes.add_column("Column", style="cyan")
        changes.add_column("Observed Types")
        for name, dtypes in sorted(drift.items()):
            changes.add_row(name, ", ".join(sorted(dtypes)))
        blocks.append(changes)

    if analysis.schemas:
        dominant_hash = analysis.schemas[0].hash
        violations = [record for record in files if record.schema_hash != dominant_hash]
        if violations:
            violating = Table(title="Files Violating Dominant Schema", expand=True)
            violating.add_column("File", style="cyan")
            violating.add_column("Schema")
            violating.add_column("Rows", justify="right")
            for record in violations[:20]:
                violating.add_row(record.relative_path.as_posix(), record.schema_hash[:12], str(record.row_count))
            blocks.append(violating)

    return Group(*blocks)
