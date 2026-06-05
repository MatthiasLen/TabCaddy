from __future__ import annotations

import math
from typing import Iterable

from tabcaddy.domain.models import DatasetAnalysis, DiffLevel, DiffReport


def compare_analyses(
    left: DatasetAnalysis, right: DatasetAnalysis, level: DiffLevel
) -> DiffReport:
    """Compare two dataset analyses and return a report of differences.
    
    The level parameter controls which categories of differences are included:
    - METADATA: Always includes row count, column count, file count, schema hashes.
    - STATISTICS: Includes metadata + column-level statistics (null rates, min/max, etc).
    - FULL: Includes metadata + statistics + schema details (added/removed schemas, column type changes).
    
    Args:
        left: The left dataset analysis for comparison.
        right: The right dataset analysis for comparison.
        level: DiffLevel controlling the depth of comparison (METADATA, STATISTICS, or FULL).
        
    Returns:
        DiffReport containing metadata_changes, schema_changes, statistics_changes, and warnings.
    """
    metadata_changes = _diff_metadata(left, right)
    schema_changes = _diff_schema(left, right) if level == DiffLevel.FULL else []
    statistics_changes = (
        _diff_statistics(left, right)
        if level in {DiffLevel.STATISTICS, DiffLevel.FULL}
        else []
    )
    warnings = sorted({*left.warnings, *right.warnings})
    return DiffReport(
        metadata_changes=metadata_changes,
        schema_changes=schema_changes,
        statistics_changes=statistics_changes,
        warnings=warnings,
    )


def _diff_metadata(left: DatasetAnalysis, right: DatasetAnalysis) -> list[str]:
    """Compare dataset metadata (counts, hashes).
    
    Always called regardless of DiffLevel. Reports changes in row count, column count,
    source file count, schema hash, and column hashes.
    """
    changes: list[str] = []
    for field in ("row_count", "column_count", "source_file_count", "schema_hash"):
        left_value = getattr(left.metadata, field)
        right_value = getattr(right.metadata, field)
        if left_value != right_value:
            changes.append(
                f"{field.replace('_', ' ').title()}: {left_value} -> {right_value}"
            )
    if left.metadata.column_hashes != right.metadata.column_hashes:
        changes.append("Column hashes changed")
    return changes


def _diff_schema(left: DatasetAnalysis, right: DatasetAnalysis) -> list[str]:
    """Compare dataset schemas (added/removed schemas and column type changes).
    
    Only called when level == DiffLevel.FULL. Reports schema additions, removals,
    and column data type changes across all schemas.
    """
    changes: list[str] = []
    left_schemas = {schema.hash: schema for schema in left.schemas}
    right_schemas = {schema.hash: schema for schema in right.schemas}
    added = sorted(set(right_schemas) - set(left_schemas))
    removed = sorted(set(left_schemas) - set(right_schemas))
    for schema_hash in added:
        changes.append(
            f"Added schema {schema_hash[:12]} ({right_schemas[schema_hash].occurrence_count} files)"
        )
    for schema_hash in removed:
        changes.append(
            f"Removed schema {schema_hash[:12]} ({left_schemas[schema_hash].occurrence_count} files)"
        )
    left_columns = _column_type_map(left.schemas)
    right_columns = _column_type_map(right.schemas)
    for column_name in sorted(set(left_columns) | set(right_columns)):
        if left_columns.get(column_name) != right_columns.get(column_name):
            changes.append(
                f"Column {column_name}: {sorted(left_columns.get(column_name, set()))} -> {sorted(right_columns.get(column_name, set()))}"
            )
    return changes


def _column_type_map(schemas: Iterable) -> dict[str, set[str]]:
    """Build a map of column names to their observed data types across schemas.
    
    Helper used by _diff_schema to detect when column types change between datasets.
    """
    mapping: dict[str, set[str]] = {}
    for schema in schemas:
        for column in schema.columns:
            mapping.setdefault(column.name, set()).add(column.dtype)
    return mapping


def _diff_statistics(left: DatasetAnalysis, right: DatasetAnalysis) -> list[str]:
    """Compare column-level statistics (null rates, min/max, mean, median, stddev).
    
    Only called when level in {DiffLevel.STATISTICS, DiffLevel.FULL}. Reports changes
    in column statistics, including missing columns and differing statistical values.
    """
    if left.statistics is None or right.statistics is None:
        return ["Statistics unavailable for one or both datasets"]
    changes: list[str] = []
    all_columns = sorted(set(left.statistics.columns) | set(right.statistics.columns))
    for column_name in all_columns:
        left_stats = left.statistics.columns.get(column_name)
        right_stats = right.statistics.columns.get(column_name)
        if left_stats is None or right_stats is None:
            changes.append(f"Column {column_name} only present in one statistics set")
            continue
        for field in (
            "null_rate",
            "unique_estimate",
            "min_value",
            "max_value",
            "mean",
            "median",
            "stddev",
        ):
            left_value = getattr(left_stats, field)
            right_value = getattr(right_stats, field)
            if _values_differ(left_value, right_value):
                changes.append(f"{column_name}.{field}: {left_value} -> {right_value}")
    return changes


def _values_differ(left_value: object, right_value: object) -> bool:
    if isinstance(left_value, float) and isinstance(right_value, float):
        return not math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9)
    return left_value != right_value
