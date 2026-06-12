from __future__ import annotations

from tabcaddy.differ.service import DiffDatasets as _DiffDatasets
from tabcaddy.domain.models import DatasetSource, DiffLevel, DiffReport


class DiffDatasets:
    def __init__(self, generate_analysis) -> None:
        self._service = _DiffDatasets(generate_analysis)

    def run(
        self, left: DatasetSource, right: DatasetSource, level: DiffLevel
    ) -> DiffReport:
        return self._service.run(left, right, level)
