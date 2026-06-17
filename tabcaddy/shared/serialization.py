from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time
from typing import Any

from tabcaddy.domain.models import (
    ColumnDefinition,
    ColumnStatistics,
    DatasetAnalysis,
    DatasetMetadata,
    DatasetStatistics,
    DiffReport,
    SchemaSignature,
)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    return value


def analysis_to_dict(analysis: DatasetAnalysis) -> dict[str, Any]:
    return {
        "metadata": {
            "version": analysis.metadata.version,
            "created_at": analysis.metadata.created_at.isoformat(),
            "row_count": analysis.metadata.row_count,
            "column_count": analysis.metadata.column_count,
            "source_file_count": analysis.metadata.source_file_count,
            "schema_hash": analysis.metadata.schema_hash,
            "column_hashes": analysis.metadata.column_hashes,
        },
        "schemas": [
            {
                "columns": [
                    {"name": column.name, "dtype": column.dtype}
                    for column in schema.columns
                ],
                "hash": schema.hash,
                "occurrence_count": schema.occurrence_count,
            }
            for schema in analysis.schemas
        ],
        "statistics": None
        if analysis.statistics is None
        else {
            "columns": {
                name: {
                    "dtype": stats.dtype,
                    "null_rate": stats.null_rate,
                    "unique_estimate": stats.unique_estimate,
                    "min_value": _serialize_value(stats.min_value),
                    "max_value": _serialize_value(stats.max_value),
                    "mean": stats.mean,
                    "median": stats.median,
                    "stddev": stats.stddev,
                    "histogram": None
                    if stats.histogram is None
                    else [
                        {"label": label, "count": count}
                        for label, count in stats.histogram
                    ],
                }
                for name, stats in analysis.statistics.columns.items()
            }
        },
        "warnings": list(analysis.warnings),
    }


def analysis_from_dict(payload: dict[str, Any]) -> DatasetAnalysis:
    metadata_payload = payload["metadata"]
    statistics_payload = payload.get("statistics")
    return DatasetAnalysis(
        metadata=DatasetMetadata(
            version=metadata_payload["version"],
            created_at=datetime.fromisoformat(metadata_payload["created_at"]),
            row_count=metadata_payload["row_count"],
            column_count=metadata_payload["column_count"],
            source_file_count=metadata_payload["source_file_count"],
            schema_hash=metadata_payload.get("schema_hash"),
            column_hashes=metadata_payload.get("column_hashes"),
        ),
        schemas=[
            SchemaSignature(
                columns=[
                    ColumnDefinition(name=column["name"], dtype=column["dtype"])
                    for column in schema["columns"]
                ],
                hash=schema["hash"],
                occurrence_count=schema["occurrence_count"],
            )
            for schema in payload.get("schemas", [])
        ],
        statistics=None
        if statistics_payload is None
        else DatasetStatistics(
            columns={
                name: ColumnStatistics(
                    dtype=stats["dtype"],
                    null_rate=stats["null_rate"],
                    unique_estimate=stats.get("unique_estimate"),
                    min_value=stats.get("min_value"),
                    max_value=stats.get("max_value"),
                    mean=stats.get("mean"),
                    median=stats.get("median"),
                    stddev=stats.get("stddev"),
                    histogram=None
                    if stats.get("histogram") is None
                    else [
                        (entry["label"], entry["count"]) for entry in stats["histogram"]
                    ],
                )
                for name, stats in statistics_payload.get("columns", {}).items()
            }
        ),
        warnings=list(payload.get("warnings", [])),
    )


def diff_report_to_dict(report: DiffReport) -> dict[str, Any]:
    return {
        "file_changes": list(report.file_changes),
        "metadata_changes": list(report.metadata_changes),
        "schema_changes": list(report.schema_changes),
        "statistics_changes": list(report.statistics_changes),
        "warnings": list(report.warnings),
        "row_diff_summary": None
        if report.row_diff_summary is None
        else _serialize_value(asdict(report.row_diff_summary)),
        "row_change_examples": _serialize_value(
            [asdict(example) for example in report.row_change_examples]
        ),
        "row_added_key_samples": _serialize_value(report.row_added_key_samples),
        "row_removed_key_samples": _serialize_value(report.row_removed_key_samples),
        "summary": None
        if report.summary is None
        else _serialize_value(asdict(report.summary)),
    }
