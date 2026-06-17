from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import polars as pl
from tqdm import tqdm

from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.domain.models import (
    ColumnDefinition,
    DatasetSource,
    SchemaSignature,
    SourceType,
)
from tabcaddy.shared.dataset_io import scan_dataframe


@dataclass(frozen=True)
class FileSchemaRecord:
    path: Path
    relative_path: Path
    schema_hash: str
    columns: list[ColumnDefinition]
    row_count: int


@dataclass(frozen=True)
class SchemaAnalysisResult:
    schemas: list[SchemaSignature]
    files: list[FileSchemaRecord]
    warnings: list[str]


def hash_schema(columns: list[ColumnDefinition]) -> str:
    digest = sha256()
    digest.update(
        "|".join(f"{column.name}:{column.dtype}" for column in columns).encode("utf-8")
    )
    return digest.hexdigest()


def schema_type_changes(schemas: list[SchemaSignature]) -> dict[str, set[str]]:
    changes: dict[str, set[str]] = {}
    for schema in schemas:
        for column in schema.columns:
            changes.setdefault(column.name, set()).add(column.dtype)
    return {name: dtypes for name, dtypes in changes.items() if len(dtypes) > 1}


class SchemaAnalyzer:
    def analyze(self, source: DatasetSource) -> SchemaAnalysisResult:
        return self.analyze_files(
            iter_dataset_files(source),
            base_path=source.path,
            source_type=source.source_type,
        )

    def analyze_files(
        self, files: list[Path], base_path: Path, source_type: SourceType
    ) -> SchemaAnalysisResult:
        grouped: dict[str, dict[str, object]] = {}
        records: list[FileSchemaRecord] = []
        warnings: list[str] = []
        failed: list[tuple[Path, Exception]] = []
        for path in tqdm(files, desc="Running schema analysis", unit="files"):
            try:
                scan = scan_dataframe(path)
                schema = scan.collect_schema()
                columns = [
                    ColumnDefinition(name=name, dtype=str(dtype))
                    for name, dtype in schema.items()
                ]
                schema_hash = hash_schema(columns)
                row_count = int(
                    scan.select(pl.len().alias("row_count")).collect().item()
                )
                relative_path = (
                    path.relative_to(base_path)
                    if source_type != SourceType.FILE
                    else Path(path.name)
                )
                records.append(
                    FileSchemaRecord(
                        path=path,
                        relative_path=relative_path,
                        schema_hash=schema_hash,
                        columns=columns,
                        row_count=row_count,
                    )
                )
                if schema_hash not in grouped:
                    grouped[schema_hash] = {"columns": columns, "count": 0}
                grouped[schema_hash]["count"] = int(grouped[schema_hash]["count"]) + 1
            except (
                OSError,
                TypeError,
                ValueError,
                pl.exceptions.PolarsError,
            ) as exc:
                failed.append((path, exc))
                warnings.append(f"Failed to inspect {path.name}: {exc}")

        if files and not records:
            failed_path, failed_error = failed[0]
            raise ValueError(
                "Failed to inspect any dataset files. "
                f"First failure: {failed_path.name}: {failed_error}"
            )

        schemas = [
            SchemaSignature(
                columns=value["columns"],
                hash=schema_hash,
                occurrence_count=int(value["count"]),
            )
            for schema_hash, value in grouped.items()
        ]
        schemas.sort(key=lambda item: (-item.occurrence_count, item.hash))
        return SchemaAnalysisResult(schemas=schemas, files=records, warnings=warnings)
