from __future__ import annotations

import json
from pathlib import Path


from tabcaddy.domain.models import DatasetSource, ProfileMode, SourceType
from tabcaddy.domain.serialization import analysis_to_dict
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder
import polars as pl

from tabcaddy.infrastructure.csv_reader import read_csv
from tabcaddy.infrastructure.feather_reader import read_feather
from tabcaddy.infrastructure.parquet_dataset_writer import write_parquet_dataset


def _read_dataframe(path: Path):
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_feather(path)


class CompileDataset:
    def __init__(self, analysis_builder: AnalysisBuilder | None = None) -> None:
        self._analysis_builder = analysis_builder or AnalysisBuilder()

    def run(
        self, source: DatasetSource, output_path: Path, schema_index: int | None = None
    ) -> tuple[Path, list[str]]:
        if source.source_type != SourceType.FOLDER:
            raise ValueError("Compile expects a folder source.")

        build_result = self._analysis_builder.build(source, ProfileMode.STANDARD)
        schemas = build_result.analysis.schemas

        if not schemas:
            raise ValueError("No schemas found to compile.")

        # If multiple schemas are present and no index is specified, prompt the user to choose.
        if len(schemas) > 1 and schema_index is None:
            labels = [
                f"Schema {index} ({schema.occurrence_count} files)"
                for index, schema in enumerate(schemas, start=1)
            ]
            raise ValueError(
                "Multiple schemas detected. Re-run with --schema. Available: "
                + ", ".join(labels)
            )

        chosen_index = schema_index or 1
        if chosen_index < 1 or chosen_index > len(schemas):
            raise ValueError(f"Schema index must be between 1 and {len(schemas)}")

        selected_schema = schemas[chosen_index - 1]
        selected_files = [
            record.path
            for record in build_result.files
            if record.schema_hash == selected_schema.hash
        ]

        output_path.mkdir(parents=True, exist_ok=False)

        def _read_with_source(path: Path):
            df = _read_dataframe(path)
            rel = path.relative_to(source.path).as_posix()
            return df.with_columns(pl.lit(rel).alias("_source_file"))

        # Read selected files and write to Parquet dataset
        written = write_parquet_dataset(
            (_read_with_source(path) for path in selected_files),
            output_path,
            total=len(selected_files),
        )

        # Build analysis for the selected files to include in metadata
        selected_analysis = self._analysis_builder.build_file_set(
            files=selected_files,
            base_path=source.path,
            source_type=SourceType.FOLDER,
            profile_mode=ProfileMode.DEEP,
        ).analysis

        payload = analysis_to_dict(selected_analysis)
        payload["compiled"] = {
            "source": str(source.path),
            "selected_schema_hash": selected_schema.hash,
            "written_parts": [str(path.relative_to(output_path)) for path in written],
        }

        (output_path / "metadata.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        skipped = [
            record.relative_path.as_posix()
            for record in build_result.files
            if record.schema_hash != selected_schema.hash
        ]
        return output_path, skipped
