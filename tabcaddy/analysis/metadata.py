from __future__ import annotations

from datetime import datetime, timezone

from tabcaddy.analysis.schema import SchemaAnalysisResult
from tabcaddy.domain.models import DatasetMetadata


class MetadataBuilder:
    def build(
        self,
        schema_result: SchemaAnalysisResult,
        row_count: int,
        column_count: int,
        column_hashes: dict[str, str] | None,
    ) -> DatasetMetadata:
        schema_hash = (
            schema_result.schemas[0].hash if len(schema_result.schemas) == 1 else None
        )
        return DatasetMetadata(
            version=1,
            created_at=datetime.now(timezone.utc),
            row_count=row_count,
            column_count=column_count,
            source_file_count=len(schema_result.files),
            schema_hash=schema_hash,
            column_hashes=column_hashes,
        )
