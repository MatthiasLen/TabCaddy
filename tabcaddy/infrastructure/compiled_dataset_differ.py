from __future__ import annotations

import json

from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
    ProfileMode,
)
from tabcaddy.infrastructure.diff_support import compare_analyses


class CompiledDatasetDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def _load_compiled_provenance(self, source: DatasetSource) -> dict | None:
        payload = json.loads((source.path / "metadata.json").read_text(encoding="utf-8"))
        compiled = payload.get("compiled")
        return compiled if isinstance(compiled, dict) else None

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        profile_mode = (
            ProfileMode.DEEP if level != DiffLevel.METADATA else ProfileMode.STANDARD
        )
        left_analysis = self._generate_analysis.run(left, profile_mode).analysis
        right_analysis = self._generate_analysis.run(right, profile_mode).analysis
        report = compare_analyses(left_analysis, right_analysis, level)
        if self._load_compiled_provenance(left) != self._load_compiled_provenance(right):
            report.metadata_changes.append("Compiled dataset provenance changed")
        report.summary = DiffSummary(comparison_type=DiffComparisonType.COMPILED)
        return report
