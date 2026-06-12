from __future__ import annotations

from tabcaddy.differ.common import analyze_pair
from tabcaddy.differ.comparison import compare_analyses
from tabcaddy.differ.folder_inventory import diff_folder_inventory
from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
)


class FolderDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        inventory = diff_folder_inventory(left, right)
        report = DiffReport()

        for file_name in inventory.added_files:
            report.file_changes.append(f"Added file: {file_name}")
        for file_name in inventory.removed_files:
            report.file_changes.append(f"Removed file: {file_name}")
        for file_name in inventory.modified_files:
            report.file_changes.append(f"Modified file: {file_name}")

        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.FOLDER,
            matching_files=inventory.matching_files,
            modified_files=len(inventory.modified_files),
            only_in_left=inventory.only_in_left,
            only_in_right=inventory.only_in_right,
        )

        if level == DiffLevel.METADATA:
            return report

        left_analysis, right_analysis = analyze_pair(
            self._generate_analysis,
            left,
            right,
            level,
        )
        content_report = compare_analyses(left_analysis, right_analysis, level)
        report.metadata_changes.extend(content_report.metadata_changes)
        report.schema_changes.extend(content_report.schema_changes)
        report.statistics_changes.extend(content_report.statistics_changes)
        report.warnings.extend(content_report.warnings)
        return report
