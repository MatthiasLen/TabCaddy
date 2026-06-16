from __future__ import annotations

from rich.console import Group

from tabcaddy.domain.models import DiffLevel
from tabcaddy.domain.models import DiffReport
from tabcaddy.rendering.console import RenderProfile
from tabcaddy.rendering.console import resolve_render_profile


def build_diff_view(
    report: DiffReport,
    *,
    level: DiffLevel = DiffLevel.FULL,
    render: RenderProfile | None = None,
):
    render = resolve_render_profile() if render is None else render
    blocks: list[object] = []
    if report.summary is not None:
        blocks.append(_build_summary_section(report, render=render))
    summary = report.summary
    if summary is not None and summary.candidate_paths:
        blocks.append(
            _build_section(
                "Match Candidates",
                summary.candidate_paths,
                "magenta",
                render=render,
            )
        )

    sections: list[tuple[str, list[str], str]] = [
        ("File Changes", report.file_changes, "cyan"),
        ("Dataset Metadata", report.metadata_changes, "cyan"),
    ]
    if level == DiffLevel.FULL:
        sections.append(("Schema Changes", report.schema_changes, "blue"))
    if level in {DiffLevel.STATISTICS, DiffLevel.FULL}:
        sections.append(("Statistics Changes", report.statistics_changes, "green"))
    sections.append(("Warnings", report.warnings, "yellow"))

    visible_sections = [section for section in sections if section[1]]
    if not visible_sections and _is_clean_diff(report):
        blocks.append(
            render.panel(
                "No differences detected.",
                title="Diff",
                border_style="green",
            )
        )
        return Group(*blocks)

    for title, lines, color in visible_sections:
        blocks.append(_build_section(title, lines, color, render=render))
    return Group(*blocks)


def _is_clean_diff(report: DiffReport) -> bool:
    summary = report.summary
    if summary is None:
        return True
    if summary.candidate_paths:
        return False
    if summary.match_status not in {None, "unmodified"}:
        return False
    if summary.content_status not in {None, "identical"}:
        return False
    if any(
        value not in {None, 0}
        for value in (
            summary.modified_files,
            summary.only_in_left,
            summary.only_in_right,
        )
    ):
        return False
    return True


def _build_summary_section(report: DiffReport, *, render: RenderProfile):
    assert report.summary is not None
    summary = report.summary
    lines = [f"Comparison Type: {summary.comparison_type.value}"]
    if summary.content_status is not None:
        lines.append(f"Content Status: {summary.content_status}")
    if summary.match_status is not None:
        lines.append(f"Match Status: {summary.match_status}")
    if summary.matched_path is not None:
        lines.append(f"Matched Path: {summary.matched_path}")
    if summary.candidate_paths:
        lines.append(f"Candidate Matches: {len(summary.candidate_paths)}")
    if summary.matching_files is not None:
        lines.append(f"Matching Files: {summary.matching_files}")
    if summary.modified_files is not None:
        lines.append(f"Modified Files: {summary.modified_files}")
    if summary.only_in_left is not None:
        lines.append(f"Only In Left: {summary.only_in_left}")
    if summary.only_in_right is not None:
        lines.append(f"Only In Right: {summary.only_in_right}")
    return _build_section("Summary", lines, "magenta", render=render)


def _build_section(
    title: str,
    lines: list[str],
    color: str,
    *,
    render: RenderProfile,
):
    table = render.table(show_header=False, expand=True)
    table.add_column("Change", style=color)
    for line in lines:
        table.add_row(line)
    return render.panel(
        table,
        title=title,
        border_style=color,
    )
