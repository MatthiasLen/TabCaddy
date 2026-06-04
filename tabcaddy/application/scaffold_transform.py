from __future__ import annotations

from pathlib import Path

from tabcaddy.domain.models import DatasetSource, ProfileMode
from tabcaddy.application.generate_analysis import GenerateAnalysis


class ScaffoldTransform:
    def __init__(self, generate_analysis: GenerateAnalysis | None = None) -> None:
        self._generate_analysis = generate_analysis or GenerateAnalysis()

    def run(self, source: DatasetSource, output_path: Path) -> Path:
        if output_path.exists():
            raise FileExistsError(
                f"Output file '{output_path}' already exists. Please provide another filename."
            )

        analysis = self._generate_analysis.run(source, ProfileMode.STANDARD)
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
                "    # Example: drop a column",
                "    # if 'debug_notes' in df.columns:",
                "    #     df = df.drop('debug_notes')",
                "",
                "    # Example: drop rows based on a condition",
                "    # df = df.filter(pl.col('quantity') > 0)",
                "",
                "    # Example: convert a column from int to float",
                "    # if 'quantity' in df.columns:",
                "    #     df = df.with_columns(pl.col('quantity').cast(pl.Float64))",
                "",
                "    # Example: fill nulls in important fields",
                "    # if 'status' in df.columns:",
                "    #     df = df.with_columns(pl.col('status').fill_null('unknown'))",
                "",
                "    # Example: parse timestamp strings",
                "    # if 'event_time' in df.columns:",
                "    #     df = df.with_columns(pl.col('event_time').str.to_datetime(strict=False))",
                "",
                "    # Example: remove duplicate rows",
                "    # df = df.unique()",
                "",
                "    # Example: choose output columns and ordering",
                "    # selected = [name for name in ['id', 'event_time', 'quantity'] if name in df.columns]",
                "    # if selected:",
                "    #     df = df.select(selected).sort(selected[0])",
                "",
                "    # Example: use context metadata (file_name, relative_path, source_root)",
                "    # if context is not None:",
                "    #     df = df.with_columns(pl.lit(context.file_name).alias('SOURCE_FILE'))",
                "",
                "    return df",
                "",
            ]
        )
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
