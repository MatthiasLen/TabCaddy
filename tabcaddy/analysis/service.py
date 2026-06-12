from __future__ import annotations

from tabcaddy.analysis.builder import AnalysisBuildResult, AnalysisBuilder
from tabcaddy.analysis.cache import CacheManager
from tabcaddy.domain.models import DatasetSource, ProfileMode, SourceType


class GenerateAnalysis:
    def __init__(
        self,
        analysis_builder: AnalysisBuilder | None = None,
        cache_manager: CacheManager | None = None,
    ) -> None:
        self._analysis_builder = analysis_builder or AnalysisBuilder()
        self._cache_manager = cache_manager or CacheManager()

    def run(
        self, source: DatasetSource, profile_mode: ProfileMode
    ) -> AnalysisBuildResult:
        if source.source_type == SourceType.COMPILED_DATASET:
            compiled = self._analysis_builder.load_compiled_result(source)
            if compiled is not None:
                return compiled

        cached = self._cache_manager.get(source, profile_mode)
        if cached is not None:
            return cached

        result = self._analysis_builder.build(source, profile_mode)
        self._cache_manager.set(source, profile_mode, result)

        return result
