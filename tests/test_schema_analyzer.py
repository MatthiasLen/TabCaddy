from __future__ import annotations

from tabcaddy.domain.models import ColumnDefinition
from tabcaddy.infrastructure.schema_analyzer import SchemaAnalyzer
from tabcaddy.infrastructure.schema_analyzer import hash_schema
from tabcaddy.infrastructure.source_resolver import resolve_source


def test_hash_schema_is_deterministic() -> None:
    columns = [ColumnDefinition("id", "Int64"), ColumnDefinition("value", "Float64")]
    assert hash_schema(columns) == hash_schema(list(columns))
    assert hash_schema(columns) != hash_schema([ColumnDefinition("value", "Float64"), ColumnDefinition("id", "Int64")])


def test_schema_analyzer_groups_matching_files(drift_folder) -> None:
    result = SchemaAnalyzer().analyze(resolve_source(drift_folder))
    counts = [schema.occurrence_count for schema in result.schemas]
    assert counts == [2, 1]
