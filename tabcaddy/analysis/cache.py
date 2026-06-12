from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from tabcaddy.analysis.builder import AnalysisBuildResult
from tabcaddy.analysis.schema import FileSchemaRecord
from tabcaddy.analysis.sources import iter_dataset_files
from tabcaddy.domain.models import (
    ColumnDefinition,
    DatasetSource,
    ProfileMode,
    SourceType,
)
from tabcaddy.shared.serialization import analysis_from_dict, analysis_to_dict


class CacheManager:
    CACHE_FORMAT_VERSION = 1

    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root or Path(".tabcaddy") / "cache"

    def get(
        self, source: DatasetSource, profile_mode: ProfileMode
    ) -> AnalysisBuildResult | None:
        cache_file = self.cache_file(source, profile_mode)
        if not cache_file.exists():
            return None
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            return self._result_from_payload(payload)
        except (OSError, ValueError, TypeError, KeyError):
            cache_file.unlink(missing_ok=True)
            return None

    def set(
        self,
        source: DatasetSource,
        profile_mode: ProfileMode,
        result: AnalysisBuildResult,
    ) -> Path:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_file(source, profile_mode)
        cache_file.write_text(
            json.dumps(
                {
                    "format_version": self.CACHE_FORMAT_VERSION,
                    "analysis": analysis_to_dict(result.analysis),
                    "files": [
                        {
                            "path": str(record.path),
                            "relative_path": str(record.relative_path),
                            "schema_hash": record.schema_hash,
                            "columns": [
                                {"name": column.name, "dtype": column.dtype}
                                for column in record.columns
                            ],
                            "row_count": record.row_count,
                        }
                        for record in result.files
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return cache_file

    def cache_file(self, source: DatasetSource, profile_mode: ProfileMode) -> Path:
        profile_mode = self._normalize_profile_mode(profile_mode)
        return self._cache_root / f"{self._build_cache_key(source, profile_mode)}.json"

    def _result_from_payload(self, payload: object) -> AnalysisBuildResult:
        if not isinstance(payload, dict):
            raise TypeError("Cache payload must be a JSON object")
        if payload.get("format_version") != self.CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported cache format version")

        return AnalysisBuildResult(
            analysis=analysis_from_dict(payload["analysis"]),
            files=[self._file_record_from_payload(item) for item in payload["files"]],
        )

    def _file_record_from_payload(self, payload: object) -> FileSchemaRecord:
        if not isinstance(payload, dict):
            raise TypeError("File record payload must be a JSON object")
        return FileSchemaRecord(
            path=Path(payload["path"]),
            relative_path=Path(payload["relative_path"]),
            schema_hash=payload["schema_hash"],
            columns=[
                ColumnDefinition(name=column["name"], dtype=column["dtype"])
                for column in payload["columns"]
            ],
            row_count=payload["row_count"],
        )

    def _normalize_profile_mode(self, profile_mode: ProfileMode) -> ProfileMode:
        if isinstance(profile_mode, ProfileMode):
            return profile_mode
        if hasattr(profile_mode, "default") and isinstance(
            profile_mode.default, ProfileMode
        ):
            return profile_mode.default
        if isinstance(profile_mode, str):
            return ProfileMode(profile_mode)
        raise TypeError(
            f"Cannot normalize profile_mode: expected ProfileMode, got {type(profile_mode).__name__}"
        )

    def _build_cache_key(self, source: DatasetSource, profile_mode: ProfileMode) -> str:
        profile_mode = self._normalize_profile_mode(profile_mode)
        fingerprint = {
            "source": str(source.path),
            "type": source.source_type.value,
            "profile": profile_mode.value,
            "files": [
                {
                    "path": str(
                        path.relative_to(source.path)
                        if source.source_type != SourceType.FILE
                        else path.name
                    ),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in iter_dataset_files(source)
            ],
        }

        if source.source_type == SourceType.COMPILED_DATASET:
            metadata_path = source.path / "metadata.json"
            if metadata_path.exists():
                fingerprint["metadata"] = {
                    "size": metadata_path.stat().st_size,
                    "mtime_ns": metadata_path.stat().st_mtime_ns,
                }

        return sha256(
            json.dumps(fingerprint, sort_keys=True).encode("utf-8")
        ).hexdigest()