from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.compiled_metadata import load_compiled_provenance
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
)


class CompiledDatasetDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self,
        left: DatasetSource,
        right: DatasetSource,
        level: DiffLevel,
        *,
        key_columns: tuple[str, ...] = (),
        row_examples: int = 20,
    ) -> DiffReport:
        _ = row_examples
        if key_columns:
            raise ValueError(
                "Row-level key diff is not supported for compiled dataset comparisons."
            )
        left_analysis, right_analysis = analyze_pair(
            self._generate_analysis,
            left,
            right,
            level,
        )
        report = compare_analyses(left_analysis, right_analysis, level)
        if load_compiled_provenance(left) != load_compiled_provenance(right):
            report.metadata_changes.append("Compiled dataset provenance changed")
        report.summary = DiffSummary(comparison_type=DiffComparisonType.COMPILED)
        return report
