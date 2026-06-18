from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import polars as pl

from tabcaddy.analysis.builder import AnalysisBuildResult, AnalysisBuilder
from tabcaddy.domain.models import DatasetSource, ProfileMode, SourceType
from tabcaddy.shared.dataset_io import (
    SUPPORTED_FILE_SUFFIXES,
    read_dataframe,
    write_parquet_dataset,
)
from tabcaddy.shared.serialization import analysis_to_dict
from tabcaddy.compilation.validator import ValidateCompiledDataset, ValidationResult


@dataclass(frozen=True)
class CompileCoverage:
    total_supported_files: int
    analyzed_files: int
    selected_files: int
    skipped_schema_files: int
    unreadable_files: int


class CompileDataset:
    def __init__(self, analysis_builder: AnalysisBuilder | None = None) -> None:
        self._analysis_builder = analysis_builder or AnalysisBuilder()
        self._validator = ValidateCompiledDataset()

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
        validate: bool = False,
        validation_progress: Callable[[str], None] | None = None,
    ) -> tuple[
        Path,
        list[str],
        list[str],
        ValidationResult | None,
        CompileCoverage,
    ]:
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
            profile_mode=ProfileMode.STANDARD,
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
        resolved_output_path = output_path.resolve()
        total_supported_files = sum(
            1
            for path in source.path.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_FILE_SUFFIXES
                and not path.is_relative_to(resolved_output_path)
            )
        )
        analyzed_files = len(selection.files)
        unreadable_files = max(total_supported_files - analyzed_files, 0)
        coverage = CompileCoverage(
            total_supported_files=total_supported_files,
            analyzed_files=analyzed_files,
            selected_files=len(selected_files),
            skipped_schema_files=len(skipped),
            unreadable_files=unreadable_files,
        )

        validation_result: ValidationResult | None = None
        if validate:
            validation_result = self._validator.run(
                source_root=source.path,
                selected_files=selected_files,
                skipped_files=skipped,
                compiled_parts=written,
                expected_columns={column.name for column in selected_schema.columns}
                | {"_source_file"},
                progress_reporter=validation_progress,
            )

        return (
            output_path,
            skipped,
            list(selection.analysis.warnings),
            validation_result,
            coverage,
        )
