from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from tabcaddy.domain.models import DatasetAnalysis, DatasetSource, ProfileMode, SourceType
from tabcaddy.domain.serialization import analysis_from_dict, analysis_to_dict
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


class CacheManager:
    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root or Path(".tabcaddy") / "cache"

    def get(self, source: DatasetSource, profile_mode: ProfileMode) -> DatasetAnalysis | None:
        cache_file = self._cache_root / f"{self._build_cache_key(source, profile_mode)}.json"
        if not cache_file.exists():
            return None
        return analysis_from_dict(json.loads(cache_file.read_text(encoding="utf-8")))

    def set(self, source: DatasetSource, profile_mode: ProfileMode, analysis: DatasetAnalysis) -> Path:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_root / f"{self._build_cache_key(source, profile_mode)}.json"
        cache_file.write_text(json.dumps(analysis_to_dict(analysis), indent=2), encoding="utf-8")
        return cache_file

    def _build_cache_key(self, source: DatasetSource, profile_mode: ProfileMode) -> str:
        fingerprint = {
            "source": str(source.path),
            "type": source.source_type.value,
            "profile": profile_mode.value,
            "files": [
                {
                    "path": str(path.relative_to(source.path) if source.source_type != SourceType.FILE else path.name),
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
        return sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
