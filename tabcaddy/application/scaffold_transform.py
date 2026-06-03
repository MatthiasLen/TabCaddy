from __future__ import annotations

from pathlib import Path

from tabcaddy.domain.models import DatasetSource, ProfileMode
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder


class ScaffoldTransform:
    def __init__(self, analysis_builder: AnalysisBuilder | None = None) -> None:
        self._analysis_builder = analysis_builder or AnalysisBuilder()

    def run(self, source: DatasetSource, output_path: Path) -> Path:
        analysis = self._analysis_builder.build(source, ProfileMode.STANDARD).analysis
        lines = [
            '"""TabCaddy transform scaffold."""',
            "",
            "import polars as pl",
            "",
            "",
            "# Observed schemas",
        ]
        for index, schema in enumerate(analysis.schemas, start=1):
            lines.append(
                f"# Schema {index}: {schema.occurrence_count} files, hash={schema.hash}"
            )
            for column in schema.columns:
                lines.append(f"#   - {column.name}: {column.dtype}")
        lines.extend(
            [
                "",
                "def transform(df: pl.DataFrame, context=None) -> pl.DataFrame:",
                "    # Example: rename a column",
                "    # if 'old_name' in df.columns:",
                "    #     df = df.rename({'old_name': 'new_name'})",
                "",
                "    # Example: filter rows",
                "    # df = df.filter(pl.col('quantity') > 0)",
                "",
                "    return df",
                "",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
