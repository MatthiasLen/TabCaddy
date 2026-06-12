from __future__ import annotations

from tabcaddy.domain.models import DatasetAnalysis, DatasetSource, DiffLevel, ProfileMode


def profile_mode_for_level(level: DiffLevel) -> ProfileMode:
    return ProfileMode.DEEP if level != DiffLevel.METADATA else ProfileMode.STANDARD


def analyze_pair(
    generate_analysis,
    left: DatasetSource,
    right: DatasetSource,
    level: DiffLevel,
) -> tuple[DatasetAnalysis, DatasetAnalysis]:
    profile_mode = profile_mode_for_level(level)
    left_analysis = generate_analysis.run(left, profile_mode).analysis
    right_analysis = generate_analysis.run(right, profile_mode).analysis
    return left_analysis, right_analysis