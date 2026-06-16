from __future__ import annotations

import math
from typing import Iterable

from tabcaddy.domain.models import DatasetAnalysis, DiffLevel, DiffReport


def compare_analyses(
    left: DatasetAnalysis, right: DatasetAnalysis, level: DiffLevel
) -> DiffReport:
    report = DiffReport()
    report.metadata_changes.extend(_diff_metadata(left, right))
    if level == DiffLevel.FULL:
        report.schema_changes.extend(_diff_schema(left, right))
    if level in {DiffLevel.STATISTICS, DiffLevel.FULL}:
        report.statistics_changes.extend(_diff_statistics(left, right))
    report.warnings.extend(sorted({*left.warnings, *right.warnings}))
    return report


def _diff_metadata(left: DatasetAnalysis, right: DatasetAnalysis) -> list[str]:
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
    mapping: dict[str, set[str]] = {}
    for schema in schemas:
        for column in schema.columns:
            mapping.setdefault(column.name, set()).add(column.dtype)
    return mapping


def _diff_statistics(left: DatasetAnalysis, right: DatasetAnalysis) -> list[str]:
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
        if math.isnan(left_value) and math.isnan(right_value):
            return False
        return not math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9)
    return left_value != right_value
