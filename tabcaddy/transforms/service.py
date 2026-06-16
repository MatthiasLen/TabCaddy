from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from tabcaddy.analysis.builder import AnalysisBuilder
from tabcaddy.analysis.schema import SchemaAnalyzer
from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.domain.models import DatasetSource, ProfileMode, SourceType
from tabcaddy.shared.dataset_io import (
    SUPPORTED_FILE_SUFFIXES,
    read_dataframe,
    write_dataframe,
)
from tabcaddy.shared.serialization import analysis_to_dict
from tabcaddy.transforms.loader import (
    TransformContext,
    TransformLoader,
    TransformMetadata,
)


def _apply_transform(
    transform: Callable[..., Any],
    expects_context: bool,
    df: pl.DataFrame,
    context: TransformContext,
) -> Any:
    if expects_context:
        return transform(df, context)
    return transform(df)


class TransformDataset:
    def __init__(
        self,
        transform_loader: TransformLoader | None = None,
        schema_analyzer: SchemaAnalyzer | None = None,
        analysis_builder: AnalysisBuilder | None = None,
    ) -> None:
        self._transform_loader = transform_loader or TransformLoader()
        self._schema_analyzer = schema_analyzer or SchemaAnalyzer()
        self._analysis_builder = analysis_builder or AnalysisBuilder()

    def run(
        self,
        source: DatasetSource,
        transform_path: Path,
        output_path: Path | None,
        workers: int,
    ) -> Path:
        write_to_single_file = (
            source.source_type == SourceType.FILE
            and output_path is not None
            and self._is_file_output_path(output_path)
        )

        output_file: Path | None = None
        output_root: Path | None = None
        if write_to_single_file:
            output_file = output_path
            if output_file.exists():
                raise FileExistsError(f"Output file already exists: {output_file}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_root = output_path or self._default_output_path(source.path)
            if output_root.exists():
                raise FileExistsError(f"Output folder already exists: {output_root}")
            output_root.mkdir(parents=True)

        transform, expects_context = self._transform_loader.load(transform_path)
        files = iter_dataset_files(source)
        schema_result = self._schema_analyzer.analyze_files(
            files, base_path=source.path, source_type=source.source_type
        )
        record_map = {record.path: record for record in schema_result.files}

        def process(path: Path) -> Path:
            record = record_map[path]
            df = read_dataframe(path)
            context = TransformContext(
                file_name=path.name,
                file_path=str(path),
                schema=[
                    {"name": column.name, "dtype": column.dtype}
                    for column in record.columns
                ],
                metadata=TransformMetadata(
                    row_count=record.row_count, schema_hash=record.schema_hash
                ),
            )
            result = _apply_transform(transform, expects_context, df, context)
            if not isinstance(result, pl.DataFrame):
                raise TypeError(
                    f"Transform must return a Polars DataFrame for {path.name}"
                )
            if output_file is not None:
                target = output_file
            else:
                relative_path = (
                    record.relative_path
                    if source.source_type != SourceType.FILE
                    else Path(path.name)
                )
                target = output_root / relative_path
            write_dataframe(result, target)
            return target

        if workers <= 1:
            written_files = [process(path) for path in files]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                written_files = list(executor.map(process, files))

        if source.source_type == SourceType.COMPILED_DATASET:
            if output_root is None:
                raise ValueError(
                    "Compiled dataset transform requires directory output."
                )
            self._write_compiled_metadata(
                source=source,
                transform_path=transform_path,
                output_root=output_root,
                written_files=written_files,
            )

        return output_file or output_root

    def _is_file_output_path(self, output_path: Path) -> bool:
        if output_path.exists():
            return output_path.is_file()
        return output_path.suffix.lower() in SUPPORTED_FILE_SUFFIXES

    def _write_compiled_metadata(
        self,
        source: DatasetSource,
        transform_path: Path,
        output_root: Path,
        written_files: list[Path],
    ) -> None:
        analysis = self._analysis_builder.build_file_set(
            files=written_files,
            base_path=output_root,
            source_type=SourceType.COMPILED_DATASET,
            profile_mode=ProfileMode.DEEP,
        ).analysis

        source_analysis = self._analysis_builder.load_compiled_analysis(source)
        if source_analysis is not None:
            analysis.metadata.source_file_count = (
                source_analysis.metadata.source_file_count
            )

        payload = analysis_to_dict(analysis)
        payload["compiled"] = {
            "source": str(source.path),
            "source_type": source.source_type.value,
            "transform_script": str(transform_path.expanduser().resolve()),
            "written_parts": [
                str(path.relative_to(output_root)) for path in written_files
            ],
        }

        (output_root / "metadata.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def _default_output_path(self, input_path: Path) -> Path:
        if input_path.is_dir():
            return input_path.parent / f"{input_path.name}_transformed"
        return input_path.parent / f"{input_path.stem}_transformed"
