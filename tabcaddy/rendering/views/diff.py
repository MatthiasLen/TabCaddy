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

    row_summary_lines = _build_row_summary_lines(report)
    if row_summary_lines:
        sections.append(("Row Diff Summary", row_summary_lines, "magenta"))

    updated_row_lines = _build_updated_row_lines(
        report,
        ascii_only=render.ascii_only,
    )
    if updated_row_lines:
        sections.append(("Updated Rows", updated_row_lines, "blue"))

    added_key_lines = _build_key_sample_lines(
        report.row_added_key_samples,
        title="added",
        ascii_only=render.ascii_only,
    )
    if added_key_lines:
        sections.append(("Added Key Samples", added_key_lines, "green"))

    removed_key_lines = _build_key_sample_lines(
        report.row_removed_key_samples,
        title="removed",
        ascii_only=render.ascii_only,
    )
    if removed_key_lines:
        sections.append(("Removed Key Samples", removed_key_lines, "yellow"))

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
    row_summary = report.row_diff_summary
    if row_summary is not None and any(
        value > 0
        for value in (
            row_summary.added_rows,
            row_summary.removed_rows,
            row_summary.updated_rows,
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


def _build_row_summary_lines(report: DiffReport) -> list[str]:
    summary = report.row_diff_summary
    if summary is None:
        return []
    return [
        f"Keys: {', '.join(summary.key_columns)}",
        f"Compared Files: {summary.compared_files}",
        f"Added Rows: {summary.added_rows}",
        f"Removed Rows: {summary.removed_rows}",
        f"Updated Rows: {summary.updated_rows}",
        f"Unchanged Rows: {summary.unchanged_rows}",
    ]


def _value_repr(value: object, *, ascii_only: bool) -> str:
    return ascii(value) if ascii_only else repr(value)


def _build_updated_row_lines(
    report: DiffReport,
    *,
    ascii_only: bool,
) -> list[str]:
    lines: list[str] = []
    for example in report.row_change_examples:
        key_repr = ", ".join(
            f"{key}={_value_repr(value, ascii_only=ascii_only)}"
            for key, value in example.key.items()
        )
        delta_repr = ", ".join(
            f"{delta.column}: "
            f"{_value_repr(delta.left_value, ascii_only=ascii_only)} -> "
            f"{_value_repr(delta.right_value, ascii_only=ascii_only)}"
            for delta in example.deltas
        )
        prefix = f"[{example.source_path}] " if example.source_path else ""
        lines.append(f"{prefix}{key_repr} | {delta_repr}")
    return lines


def _build_key_sample_lines(
    keys: list[dict[str, object]],
    *,
    title: str,
    ascii_only: bool,
) -> list[str]:
    return [
        f"{title}: "
        + ", ".join(
            f"{key}={_value_repr(value, ascii_only=ascii_only)}"
            for key, value in entry.items()
        )
        for entry in keys
    ]
