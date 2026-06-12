from __future__ import annotations

from tabcaddy.diff.common import analyze_pair
from tabcaddy.diff.comparison import compare_analyses
from tabcaddy.diff.matching import MatchStatus, resolve_file_folder_match
from tabcaddy.domain.models import (
    DatasetSource,
    DiffComparisonType,
    DiffLevel,
    DiffReport,
    DiffSummary,
    SourceType,
)


class MixedDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        file_source, folder_source = self._split_sources(left, right)
        match = resolve_file_folder_match(file_source, folder_source)

        if match.status is MatchStatus.MISSING:
            report = DiffReport()
            report.summary = DiffSummary(
                comparison_type=DiffComparisonType.FILE_FOLDER,
                match_status="missing",
            )
            return report

        if match.status is MatchStatus.AMBIGUOUS:
            report = DiffReport()
            report.summary = DiffSummary(
                comparison_type=DiffComparisonType.FILE_FOLDER,
                match_status="ambiguous",
                candidate_paths=match.candidate_paths,
            )
            return report

        if match.status is MatchStatus.UNMODIFIED:
            report = DiffReport()
            report.summary = DiffSummary(
                comparison_type=DiffComparisonType.FILE_FOLDER,
                match_status="unmodified",
                matched_path=match.matched_path,
            )
            return report

        if match.matched_file is None:
            raise ValueError("Modified file-folder match requires a matched file")

        matched_source = DatasetSource(path=match.matched_file, source_type=SourceType.FILE)
        left_analysis, right_analysis = analyze_pair(
            self._generate_analysis,
            file_source,
            matched_source,
            level,
        )
        report = compare_analyses(left_analysis, right_analysis, level)
        report.summary = DiffSummary(
            comparison_type=DiffComparisonType.FILE_FOLDER,
            match_status=MatchStatus.MODIFIED.value,
            matched_path=match.matched_path,
        )
        return report

    def _split_sources(
        self, left: DatasetSource, right: DatasetSource
    ) -> tuple[DatasetSource, DatasetSource]:
        if left.source_type == SourceType.FILE and right.source_type == SourceType.FOLDER:
            return left, right
        if left.source_type == SourceType.FOLDER and right.source_type == SourceType.FILE:
            return right, left
        raise ValueError("MixedDiffer requires one file source and one folder source.")