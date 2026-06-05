from __future__ import annotations

from tabcaddy.domain.models import DatasetSource, DiffLevel, DiffReport, ProfileMode
from tabcaddy.infrastructure.diff_support import compare_analyses


class FileDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        profile_mode = (
            ProfileMode.DEEP if level != DiffLevel.METADATA else ProfileMode.STANDARD
        )
        left_analysis = self._generate_analysis.run(left, profile_mode).analysis
        right_analysis = self._generate_analysis.run(right, profile_mode).analysis
        return compare_analyses(left_analysis, right_analysis, level)
