from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tabcaddy.analysis.builder import AnalysisBuildResult, AnalysisBuilder
from tabcaddy.domain.models import DatasetSource, ProfileMode, SourceType
from tabcaddy.shared.dataset_io import read_dataframe, write_parquet_dataset
from tabcaddy.shared.serialization import analysis_to_dict


class CompileDataset:
    def __init__(self, analysis_builder: AnalysisBuilder | None = None) -> None:
        self._analysis_builder = analysis_builder or AnalysisBuilder()

    def preview_selection(self, source: DatasetSource) -> AnalysisBuildResult:
        if source.source_type != SourceType.FOLDER:
            raise ValueError("Compile expects a folder source.")
        return self._analysis_builder.build(source, ProfileMode.QUICK)

    def run(
        self,
        source: DatasetSource,
        output_path: Path,
        schema_index: int | None = None,
        precomputed_selection: AnalysisBuildResult | None = None,
    ) -> tuple[Path, list[str]]:
        if source.source_type != SourceType.FOLDER:
            raise ValueError("Compile expects a folder source.")

        selection = precomputed_selection or self.preview_selection(source)
        schemas = selection.analysis.schemas

        if not schemas:
            raise ValueError("No schemas found to compile.")

        if len(schemas) > 1 and schema_index is None:
            labels = [
                f"Schema {index} ({schema.occurrence_count} files)"
                for index, schema in enumerate(schemas, start=1)
            ]
            raise ValueError(
                "Multiple schemas detected. Re-run with --schema. Available: "
                + ", ".join(labels)
            )

        chosen_index = schema_index if schema_index is not None else 1
        if chosen_index < 1 or chosen_index > len(schemas):
            raise ValueError(f"Schema index must be between 1 and {len(schemas)}")

        selected_schema = schemas[chosen_index - 1]
        selected_files = [
            record.path
            for record in selection.files
            if record.schema_hash == selected_schema.hash
        ]

        output_path.mkdir(parents=True, exist_ok=False)

        def _read_with_source(path: Path):
            df = read_dataframe(path)
            rel = path.relative_to(source.path).as_posix()
            return df.with_columns(pl.lit(rel).alias("_source_file"))

        written = write_parquet_dataset(
            (_read_with_source(path) for path in selected_files),
            output_path,
            total=len(selected_files),
        )

        compiled_output = self._analysis_builder.build_file_set(
            files=written,
            base_path=output_path,
            source_type=SourceType.COMPILED_DATASET,
            profile_mode=ProfileMode.DEEP,
        )
        compiled_output.analysis.metadata.source_file_count = len(selected_files)

        payload = analysis_to_dict(compiled_output.analysis)
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
            for record in selection.files
            if record.schema_hash != selected_schema.hash
        ]
        return output_path, skipped
