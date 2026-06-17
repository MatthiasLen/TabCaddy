from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.compiled_metadata import load_compiled_provenance
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.diff.folder_inventory import diff_folder_inventory
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
        inventory = diff_folder_inventory(left, right)
        left_analysis, right_analysis = analyze_pair(
            self._generate_analysis,
            left,
            right,
            level,
        )
        report = compare_analyses(left_analysis, right_analysis, level)
        for file_name in inventory.added_files:
            report.file_changes.append(f"Added file: {file_name}")
        for file_name in inventory.removed_files:
            report.file_changes.append(f"Removed file: {file_name}")
        for file_name in inventory.modified_files:
            report.file_changes.append(f"Modified file: {file_name}")
        if load_compiled_provenance(left) != load_compiled_provenance(right):
            report.metadata_changes.append("Compiled dataset provenance changed")
        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.COMPILED,
            matching_files=inventory.matching_files,
            modified_files=len(inventory.modified_files),
            only_in_left=inventory.only_in_left,
            only_in_right=inventory.only_in_right,
        )
        return report
