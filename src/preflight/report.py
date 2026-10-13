"""Rendering a :class:`~preflight.models.Report` for humans and for machines."""

from __future__ import annotations

import html
import io
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import LLM, NETWORK, Finding, Report, Severity

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


_MODE_STYLES = {"offline": "green", "online": "cyan", "llm": "magenta"}


def _mode_kind(finding: Finding) -> str:
    """A model's reading stands out from a measurement; a lookup sits between."""
    uses = finding.uses or ()
    return "llm" if LLM in uses else "online" if NETWORK in uses else "offline"


def mode_style(finding: Finding) -> str:
    return _MODE_STYLES[_mode_kind(finding)]


def shows_mode(finding: Finding) -> bool:
    # "Not checkable from the PDF" items were never checked, offline or otherwise.
    return finding.mode is not None and finding.severity is not Severity.UNVERIFIABLE


def _modes_key(report: Report) -> list[tuple[str, str | None]] | None:
    """The key to each finding's tag, as (text, mode) runs; None for plain text."""
    if not any(f.uses is not None for f in report.findings):
        return None                     # a run cached before findings were tagged
    meta = report.meta
    online = meta.get("refcheck") or meta.get("hallucinator")
    model = meta.get("llm_model")
    return [
        ("offline", "offline"),
        (" — read from the PDF alone · ", None),
        ("online", "online"),
        (" — bibliographic lookups" + ("" if online else " (off this run)") + " · ", None),
        ("LLM", "llm"),
        (f" — read by {model}" if model else " — a model's reading (off this run)", None),
    ]


def modes_line(report: Report) -> Text | None:
    """The key to each finding's tag, and what this run was allowed to reach."""
    key = _modes_key(report)
    if key is None:
        return None
    text = Text("Tags: ", style="dim")
    for chunk, kind in key:
        text.append(chunk, style=_MODE_STYLES[kind] if kind else "dim")
    return text


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


def render_finding(finding: Finding, *, verbose: bool = True) -> Group:
    style = STYLES[finding.severity]
    head = Text()
    head.append(f"{finding.severity.glyph} ", style=style)
    head.append(f"{LABELS[finding.severity]:<5} ", style=style)
    head.append(f"{finding.title}", style="bold")
    head.append(f"  [{finding.check_id}]", style="dim")
    if shows_mode(finding):
        head.append(f"  {finding.mode}", style=mode_style(finding))

    body: list[Text] = [Text(finding.message, style="" if finding.severity is not Severity.SKIPPED else "dim")]

    limit = None if verbose else 4
    for ev in finding.evidence[:limit]:
        rendered = ev.render()
        if rendered:
            body.append(Text(f"  · {rendered}", style="dim"))
    hidden = len(finding.evidence) - (limit or len(finding.evidence))
    if hidden > 0:
        body.append(Text(f"  · ... and {hidden} more (preflight show {finding.check_id})",
                         style="dim italic"))

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
    verbose: bool = True,
    show_passed: bool = True,
) -> None:
    console = console or Console()
    console.print()
    console.print(Panel(headline(report), border_style="red" if report.errors else "green",
                        subtitle=report.pdf_path, subtitle_align="right"))
    modes = modes_line(report)
    if modes is not None:
        console.print(modes)

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


def _verdict(report: Report, where: str) -> tuple[Severity, str, str]:
    """The one-line answer, with ``where`` the errors sit relative to it."""
    if report.errors:
        return (Severity.ERROR, "✗ Not ready to submit",
                f" — fix the errors {where}; each one is a documented desk-rejection condition.")
    if report.warnings:
        return Severity.WARNING, "⚠ No blocking errors", " — review the warnings before submitting."
    return (Severity.PASS, "✓ All PDF-checkable rules pass",
            " — the items under 'Not checkable from the PDF' are still on you.")


def _summary_line(report: Report) -> Text:
    severity, head, tail = _verdict(report, "above")
    text = Text()
    text.append(head, style={Severity.ERROR: "bold red", Severity.WARNING: "bold yellow"}.get(severity, "bold green"))
    text.append(tail)
    return text


def to_markdown(report: Report, *, verbose: bool = True) -> str:
    """A copy-pasteable summary, for pasting into an issue or an email.

    Verbose lists every piece of evidence; otherwise each finding keeps its first six.
    """
    counts = report.counts()
    lines = [
        f"# {report.conference.upper()} pre-flight — {report.track} paper",
        "",
        f"`{report.pdf_path}`",
        "",
        f"**{counts['error']} errors, {counts['warning']} warnings, {counts['pass']} passed**",
        "",
    ]
    modes = modes_line(report)
    if modes is not None:
        lines += [f"_{modes.plain}_", ""]
    for severity in _ORDER:
        findings = [f for f in report.sorted_findings() if f.severity is severity]
        if not findings:
            continue
        lines += [f"## {_SECTION_TITLES[severity]}", ""]
        for f in findings:
            tag = f", {f.mode}" if shows_mode(f) else ""
            lines.append(f"- **{f.title}** (`{f.check_id}`{tag}) — {f.message}")
            for ev in f.evidence[: None if verbose else 6]:
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


# The HTML and PDF reports share one body. PyMuPDF lays the PDF out with a
# subset of CSS, so the shared stylesheet sticks to plain properties (no
# variables, no flexbox); what only a browser needs goes in _SCREEN_CSS. That
# includes background tints: PyMuPDF repaints a box's background along the top
# of every later page.
_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.45; color: #1f2328; }
h1 { font-size: 16pt; margin: 0 0 4pt 0; }
h2 { font-size: 12pt; margin: 18pt 0 8pt 0; padding-bottom: 3pt; border-bottom: 0.75pt solid #d0d7de; }
p { margin: 0 0 4pt 0; }
code { font-family: monospace; font-size: 9pt; color: #57606a; }
ul { margin: 2pt 0 4pt 0; padding-left: 14pt; }
li { margin: 0 0 1pt 0; color: #57606a; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt 0; }
th, td { text-align: left; vertical-align: top; padding: 3pt 6pt; border-bottom: 0.5pt solid #d0d7de; }
.muted { color: #6e7781; }
.note { color: #6e7781; font-style: italic; font-size: 9pt; }
.fix { font-style: italic; }
.verdict { padding: 2pt 8pt; margin: 8pt 0; }
.v-error { border-left: 3pt solid #cf222e; }
.v-warning { border-left: 3pt solid #d4a72c; }
.v-pass { border-left: 3pt solid #2da44e; }
.finding { margin: 0 0 8pt 0; padding: 5pt 8pt; border-left: 3pt solid #d0d7de; }
.f-error { border-left: 3pt solid #cf222e; }
.f-warning { border-left: 3pt solid #d4a72c; }
.f-pass { border-left: 3pt solid #2da44e; }
.f-skipped { border-left: 3pt solid #d0d7de; color: #6e7781; }
.f-unverifiable { border-left: 3pt solid #54aeff; }
.t-error { color: #cf222e; }
.t-warning { color: #9a6700; }
.t-pass { color: #1a7f37; }
.t-skipped { color: #6e7781; }
.t-unverifiable { color: #0969da; }
.m-offline { color: #1a7f37; }
.m-online { color: #1b7c83; }
.m-llm { color: #8250df; }
"""

_SCREEN_CSS = """
body { max-width: 52rem; margin: 2rem auto; padding: 0 1rem; background-color: #ffffff;
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.verdict { padding: 6pt 8pt; }
.v-error { background-color: #ffebe9; }
.v-warning { background-color: #fff8c5; }
.v-pass { background-color: #dafbe1; }
.f-error { background-color: #fff5f5; }
.f-warning { background-color: #fffaeb; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _html_finding(finding: Finding, *, verbose: bool) -> str:
    sev = finding.severity.value
    head = (f'<span class="t-{sev}"><b>{finding.severity.glyph} {LABELS[finding.severity]}</b></span> '
            f"<b>{_esc(finding.title)}</b> <code>[{_esc(finding.check_id)}]</code>")
    if shows_mode(finding):
        head += f' <span class="m-{_mode_kind(finding)}">{_esc(finding.mode)}</span>'
    parts = [f'<div class="finding f-{sev}">', f"<p>{head}</p>", f"<p>{_esc(finding.message)}</p>"]

    limit = None if verbose else 6
    items = [ev.render() for ev in finding.evidence[:limit]]
    hidden = len(finding.evidence) - (limit or len(finding.evidence))
    if hidden > 0:
        items.append(f"... and {hidden} more (preflight show {finding.check_id})")
    if any(items):
        parts.append("<ul>" + "".join(f"<li>{_esc(i)}</li>" for i in items if i) + "</ul>")

    if finding.confidence:
        parts.append(f'<p class="note">confidence: {_esc(finding.confidence)}</p>')
    if finding.remedy:
        parts.append(f'<p class="fix">Fix: {_esc(finding.remedy)}</p>')
    if verbose and finding.cfp_reference:
        parts.append(f'<p class="note">CFP: {_esc(finding.cfp_reference)}</p>')
    parts.append("</div>")
    return "\n".join(parts)


def _html_scores(scores: dict) -> str:
    parts = [f"<h2>Reviewer-style scores ({_esc(scores.get('model', 'model'))})</h2>",
             '<p class="note">An estimate from a partial read, not a review.</p>']
    dimensions = scores.get("dimensions") or {}
    if dimensions:
        rows = "".join(
            f"<tr><td>{_esc(name.replace('_', ' ').title())}</td><td>{_esc(value.get('score'))}/5</td>"
            f"<td>{_esc(value.get('justification', ''))}</td></tr>"
            for name, value in dimensions.items()
        )
        parts.append(f"<table><tr><th>Dimension</th><th>Score</th><th>Justification</th></tr>{rows}</table>")
    overall = scores.get("overall") or {}
    if overall:
        parts.append(f"<p><b>Overall {_esc(overall.get('score'))}/5</b> — "
                     f"{_esc(overall.get('recommendation', ''))}</p>")
    for key, sev in (("strengths", "pass"), ("weaknesses", "warning"), ("desk_reject_risks", "error")):
        items = scores.get(key) or []
        if items:
            parts.append(f'<p class="t-{sev}"><b>{key.replace("_", " ").title()}</b></p>')
            parts.append("<ul>" + "".join(f"<li>{_esc(item)}</li>" for item in items) + "</ul>")
    return "\n".join(parts)


def _html_body(report: Report, *, verbose: bool) -> str:
    counts = report.counts()
    errors, warnings = counts["error"], counts["warning"]
    meta = report.meta
    when = datetime.fromtimestamp(meta["saved_at"]) if meta.get("saved_at") else datetime.now()
    facts = [meta.get("profile_name"),
             f"{meta['page_count']} pages" if meta.get("page_count") else None,
             f"checked {when:%Y-%m-%d %H:%M}"]

    parts = [
        f"<h1>{_esc(report.conference.upper())} pre-flight — {_esc(report.track)} paper</h1>",
        f"<p><code>{_esc(report.pdf_path)}</code></p>",
        f'<p class="muted">{_esc(" · ".join(f for f in facts if f))}</p>',
        f'<p><b><span class="t-{"error" if errors else "pass"}">{errors} error{"s" if errors != 1 else ""}</span>, '
        f'<span class="t-{"warning" if warnings else "pass"}">{warnings} warning{"s" if warnings != 1 else ""}</span>,'
        f' <span class="muted">{counts["pass"]} passed</span></b></p>',
    ]
    severity, head, tail = _verdict(report, "below")
    parts.append(f'<p class="verdict v-{severity.value}"><b>{_esc(head)}</b>{_esc(tail)}</p>')
    key = _modes_key(report)
    if key is not None:
        tags = "".join(f'<span class="m-{kind}">{_esc(chunk)}</span>' if kind else _esc(chunk)
                       for chunk, kind in key)
        parts.append(f'<p class="note">Tags: {tags}</p>')

    for sev in _ORDER:
        findings = [f for f in report.sorted_findings() if f.severity is sev]
        if not findings:
            continue
        parts.append(f'<h2 class="t-{sev.value}">{_esc(_SECTION_TITLES[sev])}</h2>')
        parts.extend(_html_finding(f, verbose=verbose) for f in findings)

    if report.scores:
        parts.append(_html_scores(report.scores))
    return "\n".join(parts)


def _report_title(report: Report) -> str:
    return f"{report.conference.upper()} pre-flight — {Path(report.pdf_path).name}"


def to_html(report: Report, *, verbose: bool = True) -> str:
    """A self-contained page: open it in a browser, or attach it to an email.

    Verbose lists every piece of evidence; otherwise each finding keeps its first six.
    """
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(_report_title(report))}</title>\n"
        f"<style>{_CSS}{_SCREEN_CSS}</style>\n</head>\n<body>\n"
        f"{_html_body(report, verbose=verbose)}\n</body>\n</html>\n"
    )


def to_pdf(report: Report, *, verbose: bool = True) -> bytes:
    """The HTML report laid out on Letter pages, numbered in the footer."""
    import pymupdf

    story = pymupdf.Story(html=_html_body(report, verbose=verbose), user_css=_CSS)
    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    page = pymupdf.paper_rect("letter")
    frame = page + (54, 54, -54, -66)
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(frame)
        story.draw(device)
        writer.end_page()
    writer.close()

    doc = pymupdf.open("pdf", buffer.getvalue())
    for number, pdf_page in enumerate(doc, 1):
        pdf_page.insert_text((54, page.height - 36), f"preflight report · page {number} of {doc.page_count}",
                             fontsize=8, color=(0.43, 0.47, 0.51))
    doc.set_metadata({"title": _report_title(report), "creator": "preflight"})
    doc.subset_fonts()                  # the whole embedded fonts are most of the file
    return doc.tobytes(garbage=3, deflate=True)
