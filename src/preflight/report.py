"""Rendering a :class:`~preflight.models.Report` for humans and for machines."""

from __future__ import annotations

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import Finding, Report, Severity

STYLES: dict[Severity, str] = {
    Severity.ERROR: "bold red",
    Severity.WARNING: "bold yellow",
    Severity.PASS: "green",
    Severity.SKIPPED: "dim",
    Severity.UNVERIFIABLE: "cyan",
}

LABELS: dict[Severity, str] = {
    Severity.ERROR: "ERROR",
    Severity.WARNING: "WARN",
    Severity.PASS: "PASS",
    Severity.SKIPPED: "SKIP",
    Severity.UNVERIFIABLE: "N/A",
}

_ORDER = [Severity.ERROR, Severity.WARNING, Severity.PASS, Severity.SKIPPED, Severity.UNVERIFIABLE]

_SECTION_TITLES = {
    Severity.ERROR: "Errors — likely desk rejection",
    Severity.WARNING: "Warnings — review manually",
    Severity.PASS: "Passed",
    Severity.SKIPPED: "Skipped",
    Severity.UNVERIFIABLE: "Not checkable from the PDF",
}


def headline(report: Report) -> Text:
    counts = report.counts()
    errors, warnings = counts["error"], counts["warning"]
    text = Text()
    text.append(f"{report.conference.upper()} pre-flight", style="bold")
    text.append(f" ({report.track}): ")
    text.append(f"{errors} error{'s' if errors != 1 else ''}",
                style="bold red" if errors else "green")
    text.append(", ")
    text.append(f"{warnings} warning{'s' if warnings != 1 else ''}",
                style="bold yellow" if warnings else "green")
    text.append(f", {counts['pass']} passed", style="dim")
    return text


def render_finding(finding: Finding, *, verbose: bool = False) -> Group:
    style = STYLES[finding.severity]
    head = Text()
    head.append(f"{finding.severity.glyph} ", style=style)
    head.append(f"{LABELS[finding.severity]:<5} ", style=style)
    head.append(f"{finding.title}", style="bold")
    head.append(f"  [{finding.check_id}]", style="dim")

    body: list[Text] = [Text(finding.message, style="" if finding.severity is not Severity.SKIPPED else "dim")]

    limit = None if verbose else 4
    for ev in finding.evidence[:limit]:
        rendered = ev.render()
        if rendered:
            body.append(Text(f"  · {rendered}", style="dim"))
    hidden = len(finding.evidence) - (limit or len(finding.evidence))
    if hidden > 0:
        body.append(Text(f"  · ... and {hidden} more (use --verbose)", style="dim italic"))

    if finding.confidence:
        body.append(Text(f"  confidence: {finding.confidence}", style="dim italic"))
    if finding.remedy:
        body.append(Text(f"  fix: {finding.remedy}", style="italic"))
    if verbose and finding.cfp_reference:
        body.append(Text(f"  CFP: {finding.cfp_reference}", style="dim italic"))

    return Group(head, Padding(Group(*body), (0, 0, 1, 6)))


def print_report(
    report: Report,
    console: Console | None = None,
    *,
    verbose: bool = False,
    show_passed: bool = True,
) -> None:
    console = console or Console()
    console.print()
    console.print(Panel(headline(report), border_style="red" if report.errors else "green",
                        subtitle=report.pdf_path, subtitle_align="right"))

    for severity in _ORDER:
        findings = [f for f in report.sorted_findings() if f.severity is severity]
        if not findings:
            continue
        if severity in (Severity.PASS, Severity.SKIPPED) and not show_passed:
            continue
        console.print()
        console.print(Text(_SECTION_TITLES[severity], style=f"{STYLES[severity]} underline"))
        console.print()
        for finding in findings:
            console.print(render_finding(finding, verbose=verbose))

    if report.scores:
        console.print(render_scores(report))

    console.print()
    console.print(_summary_line(report))
    console.print()


def render_scores(report: Report) -> Panel:
    scores = report.scores
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Dimension")
    table.add_column("Score", justify="right")
    table.add_column("Justification", overflow="fold")
    for name, value in (scores.get("dimensions") or {}).items():
        table.add_row(name.replace("_", " ").title(), f"{value.get('score')}/5",
                      str(value.get("justification", "")))

    parts: list[object] = [table]
    overall = scores.get("overall") or {}
    if overall:
        parts.append(Text(f"\nOverall {overall.get('score')}/5 — {overall.get('recommendation', '')}",
                          style="bold"))
    for key, style in (("strengths", "green"), ("weaknesses", "yellow"), ("desk_reject_risks", "red")):
        items = scores.get(key) or []
        if items:
            parts.append(Text(f"\n{key.replace('_', ' ').title()}:", style=f"bold {style}"))
            for item in items:
                parts.append(Text(f"  · {item}", style=style))

    return Panel(Group(*parts), title=f"Reviewer-style scores ({scores.get('model', 'model')})",
                 border_style="magenta", title_align="left",
                 subtitle="an estimate from a partial read, not a review", subtitle_align="right")


def _summary_line(report: Report) -> Text:
    text = Text()
    if report.errors:
        text.append("✗ Not ready to submit", style="bold red")
        text.append(" — fix the errors above; each one is a documented desk-rejection condition.")
    elif report.warnings:
        text.append("⚠ No blocking errors", style="bold yellow")
        text.append(" — review the warnings before submitting.")
    else:
        text.append("✓ All PDF-checkable rules pass", style="bold green")
        text.append(" — the items under 'Not checkable from the PDF' are still on you.")
    return text


def to_markdown(report: Report) -> str:
    """A copy-pasteable summary, for pasting into an issue or an email."""
    counts = report.counts()
    lines = [
        f"# {report.conference.upper()} pre-flight — {report.track} paper",
        "",
        f"`{report.pdf_path}`",
        "",
        f"**{counts['error']} errors, {counts['warning']} warnings, {counts['pass']} passed**",
        "",
    ]
    for severity in _ORDER:
        findings = [f for f in report.sorted_findings() if f.severity is severity]
        if not findings:
            continue
        lines += [f"## {_SECTION_TITLES[severity]}", ""]
        for f in findings:
            lines.append(f"- **{f.title}** (`{f.check_id}`) — {f.message}")
            for ev in f.evidence[:6]:
                rendered = ev.render()
                if rendered:
                    lines.append(f"  - {rendered}")
            if f.remedy:
                lines.append(f"  - _Fix:_ {f.remedy}")
        lines.append("")

    scores = report.scores
    if scores:
        lines += [f"## Reviewer-style scores ({scores.get('model', 'model')})", ""]
        for name, value in (scores.get("dimensions") or {}).items():
            lines.append(f"- **{name}**: {value.get('score')}/5 — {value.get('justification', '')}")
        overall = scores.get("overall") or {}
        if overall:
            lines.append(f"- **overall**: {overall.get('score')}/5 — {overall.get('recommendation', '')}")
        lines.append("")
    return "\n".join(lines)
