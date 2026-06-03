from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

from tabcaddy.domain.models import DatasetSource, SourceType
from tabcaddy.infrastructure.csv_reader import read_csv
from tabcaddy.infrastructure.csv_writer import write_csv
from tabcaddy.infrastructure.feather_reader import read_feather
from tabcaddy.infrastructure.feather_writer import write_feather
from tabcaddy.infrastructure.schema_analyzer import SchemaAnalyzer
from tabcaddy.infrastructure.source_resolver import iter_dataset_files
from tabcaddy.infrastructure.transform_loader import TransformContext, TransformLoader, TransformMetadata


def _read_dataframe(path: Path) -> pl.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_csv(path)
    return read_feather(path)


def _write_dataframe(df: pl.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        write_csv(df, path)
        return
    write_feather(df, path)


class TransformDataset:
    def __init__(self, transform_loader: TransformLoader | None = None, schema_analyzer: SchemaAnalyzer | None = None) -> None:
        self._transform_loader = transform_loader or TransformLoader()
        self._schema_analyzer = schema_analyzer or SchemaAnalyzer()

    def run(self, source: DatasetSource, transform_path: Path, output_path: Path | None, workers: int) -> Path:
        if source.source_type == SourceType.COMPILED_DATASET:
            raise ValueError("Transform currently supports files and folders, not compiled datasets.")
        output_root = output_path or self._default_output_path(source.path)
        output_root.mkdir(parents=True, exist_ok=True)
        transform, expects_context = self._transform_loader.load(transform_path)
        files = iter_dataset_files(source)
        schema_result = self._schema_analyzer.analyze_files(files, base_path=source.path, source_type=source.source_type)
        record_map = {record.path: record for record in schema_result.files}

        def process(path: Path) -> None:
            record = record_map[path]
            df = _read_dataframe(path)
            context = TransformContext(
                file_name=path.name,
                file_path=str(path),
                schema=[{"name": column.name, "dtype": column.dtype} for column in record.columns],
                metadata=TransformMetadata(row_count=record.row_count, schema_hash=record.schema_hash),
            )
            result = transform(df, context) if expects_context else transform(df)
            if not isinstance(result, pl.DataFrame):
                raise TypeError(f"Transform must return a Polars DataFrame for {path.name}")
            relative_path = record.relative_path if source.source_type != SourceType.FILE else Path(path.name)
            _write_dataframe(result, output_root / relative_path)

        if workers <= 1:
            for path in files:
                process(path)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(process, files))
        return output_root

    def _default_output_path(self, input_path: Path) -> Path:
        if input_path.is_dir():
            return input_path.parent / f"{input_path.name}_transformed"
        return input_path.parent / f"{input_path.stem}_transformed"
