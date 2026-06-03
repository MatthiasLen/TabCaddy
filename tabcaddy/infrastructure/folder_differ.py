from __future__ import annotations

from hashlib import file_digest

from tabcaddy.domain.models import DatasetSource, DiffLevel, DiffReport, ProfileMode
from tabcaddy.infrastructure.diff_support import compare_analyses
from tabcaddy.infrastructure.source_resolver import iter_dataset_files


def _content_hash(path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


class FolderDiffer:
    def __init__(self, generate_analysis) -> None:
        self._generate_analysis = generate_analysis

    def diff(self, left: DatasetSource, right: DatasetSource, level: DiffLevel) -> DiffReport:
        profile_mode = ProfileMode.DEEP if level != DiffLevel.METADATA else ProfileMode.STANDARD
        left_analysis = self._generate_analysis.run(left, profile_mode)
        right_analysis = self._generate_analysis.run(right, profile_mode)
        report = compare_analyses(left_analysis, right_analysis, level)
        left_files = {str(path.relative_to(left.path)): path for path in iter_dataset_files(left)}
        right_files = {str(path.relative_to(right.path)): path for path in iter_dataset_files(right)}
        for file_name in sorted(right_files.keys() - left_files.keys()):
            report.metadata_changes.append(f"Added file: {file_name}")
        for file_name in sorted(left_files.keys() - right_files.keys()):
            report.metadata_changes.append(f"Removed file: {file_name}")
        for file_name in sorted(left_files.keys() & right_files.keys()):
            left_path = left_files[file_name]
            right_path = right_files[file_name]
            if left_path.stat().st_size != right_path.stat().st_size or _content_hash(left_path) != _content_hash(right_path):
                report.metadata_changes.append(f"Modified file: {file_name}")
        return report
