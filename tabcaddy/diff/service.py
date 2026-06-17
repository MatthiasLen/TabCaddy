from __future__ import annotations

from tabcaddy.diff.strategies import (
    CompiledDatasetDiffer,
    FileDiffer,
    FolderDiffer,
    MixedDiffer,
)
from tabcaddy.domain.models import DatasetSource, DiffLevel, DiffReport, SourceType


class DiffDatasets:
    def __init__(self, generate_analysis) -> None:
        file_differ = FileDiffer(generate_analysis)
        folder_differ = FolderDiffer(generate_analysis)
        compiled_differ = CompiledDatasetDiffer(generate_analysis)
        mixed_differ = MixedDiffer(generate_analysis)
        self._strategies = {
            (SourceType.FILE, SourceType.FILE): file_differ.diff,
            (SourceType.FOLDER, SourceType.FOLDER): folder_differ.diff,
            (
                SourceType.COMPILED_DATASET,
                SourceType.COMPILED_DATASET,
            ): compiled_differ.diff,
            (SourceType.FILE, SourceType.FOLDER): mixed_differ.diff,
            (SourceType.FOLDER, SourceType.FILE): mixed_differ.diff,
        }

    def run(
        self,
        left: DatasetSource,
        right: DatasetSource,
        level: DiffLevel,
        *,
        key_columns: tuple[str, ...] = (),
        row_examples: int = 20,
    ) -> DiffReport:
        strategy = self._strategies.get((left.source_type, right.source_type))
        if strategy is not None:
            return strategy(
                left,
                right,
                level,
                key_columns=key_columns,
                row_examples=row_examples,
            )
        raise ValueError(
            "Unsupported diff source combination: "
            f"{left.source_type.value} vs {right.source_type.value}"
        )
