from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from tabcaddy.domain.models import DatasetSource, ProfileMode, SourceType
from tabcaddy.domain.serialization import analysis_to_dict
from tabcaddy.infrastructure.analysis_builder import AnalysisBuilder
from tabcaddy.infrastructure.csv_reader import read_csv
from tabcaddy.infrastructure.csv_writer import write_csv
from tabcaddy.infrastructure.feather_reader import read_feather
from tabcaddy.infrastructure.feather_writer import write_feather
from tabcaddy.infrastructure.parquet_dataset_reader import read_parquet_file
from tabcaddy.infrastructure.schema_analyzer import SchemaAnalyzer
from tabcaddy.infrastructure.source_resolver import iter_dataset_files
from tabcaddy.infrastructure.transform_loader import (
    TransformContext,
    TransformLoader,
    TransformMetadata,
)


def _read_dataframe(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    if path.suffix.lower() == ".parquet":
        return read_parquet_file(path)
    return read_feather(path)


def _write_dataframe(df: pl.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        write_csv(df, path)
        return
    if path.suffix.lower() == ".parquet":
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)
        return
    write_feather(df, path)


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
            df = _read_dataframe(path)
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
            relative_path = (
                record.relative_path
                if source.source_type != SourceType.FILE
                else Path(path.name)
            )
            target = output_root / relative_path
            _write_dataframe(result, target)
            return target

        if workers <= 1:
            written_files = [process(path) for path in files]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                written_files = list(executor.map(process, files))

        if source.source_type == SourceType.COMPILED_DATASET:
            self._write_compiled_metadata(
                source=source,
                transform_path=transform_path,
                output_root=output_root,
                written_files=written_files,
            )

        return output_root

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
            # Preserve logical source file count rather than parquet part count.
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
