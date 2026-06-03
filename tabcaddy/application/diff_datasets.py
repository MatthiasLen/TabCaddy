from __future__ import annotations

from tabcaddy.domain.models import DatasetSource, DiffLevel, DiffReport, SourceType
from tabcaddy.infrastructure.compiled_dataset_differ import CompiledDatasetDiffer
from tabcaddy.infrastructure.file_differ import FileDiffer
from tabcaddy.infrastructure.folder_differ import FolderDiffer


class DiffDatasets:
    def __init__(self, generate_analysis) -> None:
        self._file_differ = FileDiffer(generate_analysis)
        self._folder_differ = FolderDiffer(generate_analysis)
        self._compiled_differ = CompiledDatasetDiffer(generate_analysis)

    def run(self, left: DatasetSource, right: DatasetSource, level: DiffLevel) -> DiffReport:
        if left.source_type == right.source_type == SourceType.FILE:
            return self._file_differ.diff(left, right, level)
        if left.source_type == right.source_type == SourceType.FOLDER:
            return self._folder_differ.diff(left, right, level)
        if left.source_type == right.source_type == SourceType.COMPILED_DATASET:
            return self._compiled_differ.diff(left, right, level)
        return self._file_differ.diff(left, right, level)