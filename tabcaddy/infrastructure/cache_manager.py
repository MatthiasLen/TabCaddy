from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from tabcaddy.domain.models import (
    DatasetAnalysis,
    DatasetSource,
    ProfileMode,
    SourceType,
)
from tabcaddy.domain.serialization import analysis_from_dict, analysis_to_dict
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


class CacheManager:
    def __init__(self, cache_root: Path | None = None) -> None:
        self._cache_root = cache_root or Path(".tabcaddy") / "cache"

    def get(
        self, source: DatasetSource, profile_mode: ProfileMode
    ) -> DatasetAnalysis | None:
        profile_mode = self._normalize_profile_mode(profile_mode)
        cache_file = (
            self._cache_root / f"{self._build_cache_key(source, profile_mode)}.json"
        )
        if not cache_file.exists():
            return None
        return analysis_from_dict(json.loads(cache_file.read_text(encoding="utf-8")))

    def set(
        self,
        source: DatasetSource,
        profile_mode: ProfileMode,
        analysis: DatasetAnalysis,
    ) -> Path:
        profile_mode = self._normalize_profile_mode(profile_mode)
        self._cache_root.mkdir(parents=True, exist_ok=True)
        cache_file = (
            self._cache_root / f"{self._build_cache_key(source, profile_mode)}.json"
        )
        cache_file.write_text(
            json.dumps(analysis_to_dict(analysis), indent=2), encoding="utf-8"
        )
        return cache_file

    def _normalize_profile_mode(self, profile_mode: ProfileMode) -> ProfileMode:
        """Extract ProfileMode value if wrapped in framework objects like Typer's OptionInfo."""
        if isinstance(profile_mode, ProfileMode):
            return profile_mode
        # Handle framework wrappers (e.g., Typer OptionInfo) with default attribute
        if hasattr(profile_mode, "default") and isinstance(
            profile_mode.default, ProfileMode
        ):
            return profile_mode.default
        # Fallback: try to convert string representation to ProfileMode
        if isinstance(profile_mode, str):
            return ProfileMode(profile_mode)
        raise TypeError(
            f"Cannot normalize profile_mode: expected ProfileMode, got {type(profile_mode).__name__}"
        )

    def _build_cache_key(self, source: DatasetSource, profile_mode: ProfileMode) -> str:
        profile_mode = self._normalize_profile_mode(profile_mode)

        # Build a fingerprint of the dataset source that includes file paths, sizes, and modification times
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

        # Include metadata.json in the fingerprint for compiled datasets, as it can affect profiling results
        if source.source_type == SourceType.COMPILED_DATASET:
            metadata_path = source.path / "metadata.json"
            if metadata_path.exists():
                fingerprint["metadata"] = {
                    "size": metadata_path.stat().st_size,
                    "mtime_ns": metadata_path.stat().st_mtime_ns,
                }

        # Use a stable hash of the fingerprint as the cache key
        return sha256(
            json.dumps(fingerprint, sort_keys=True).encode("utf-8")
        ).hexdigest()
