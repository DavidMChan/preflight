"""Statements a venue asks for beside the paper: AI use, ethics, reproducibility.

Each is declared in the profile's ``statements:`` map, so this module knows only
the shape of the rules: whether the statement is required or recommended, where
it belongs, how long it may run, and which template boilerplate must not survive
into the submission. A venue that declares none gets no findings.

A missing *required* statement is an error even when the venue names no
penalty: the point of a pre-flight check is to catch it before a reviewer does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..context import CheckContext
from ..document import Line
from ..models import Evidence, Finding, Severity
from ..registry import register

MODULE = "core.statements"

# A run-in head: "\paragraph{AI use statement}" or "\textbf{Ethics statement.}"
# sets the label in bold at the start of the paragraph rather than as a heading.
_RUN_IN_RE = re.compile(r"^\s*([A-Za-z][A-Za-z()\-/ ]{1,70}?)\s*[.:—–]")


@dataclass(slots=True)
class _Statement:
    key: str
    spec: dict[str, Any]
    start: int | None = None       # index into reading order of its heading line
    run_in: bool = False

    @property
    def title(self) -> str:
        return str(self.spec.get("title") or self.key.replace("_", " ").capitalize())

    @property
    def phrase(self) -> str:
        """The title as it reads mid-sentence: "ethics statement", "AI use statement"."""
        first = self.title.split()[0]
        return self.title if first.isupper() else self.title[0].lower() + self.title[1:]

    @property
    def required(self) -> bool:
        return str(self.spec.get("requirement", "recommended")).lower() == "required"

    @property
    def check_id(self) -> str:
        return f"statement.{self.key}"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def _aliases(spec: dict[str, Any]) -> set[str]:
    return {_normalize(str(a)) for a in (spec.get("aliases") or [])}


def _heading_indices(ctx: CheckContext) -> list[int]:
    """Reading-order indices of every detected heading line."""
    marks = {(h.page, round(h.bbox[1], 1)) for h in ctx.doc.headings}
    return [i for i, ln in enumerate(ctx.doc.reading_order) if (ln.page, round(ln.bbox[1], 1)) in marks]


def _line_index(ctx: CheckContext, page: int, y: float) -> int | None:
    return next(
        (i for i, ln in enumerate(ctx.doc.reading_order) if ln.page == page and abs(ln.bbox[1] - y) < 0.6),
        None,
    )


def _locate(ctx: CheckContext, stmt: _Statement) -> None:
    heading = ctx.doc.find_heading([str(a) for a in (stmt.spec.get("aliases") or [])])
    if heading is not None:
        stmt.start = _line_index(ctx, heading.page, heading.bbox[1])
        if stmt.start is not None:
            return
    wanted = _aliases(stmt.spec)
    for i, line in enumerate(ctx.doc.reading_order):
        spans = [s for s in line.spans if s.text.strip()]
        if not spans or not spans[0].bold:
            continue
        m = _RUN_IN_RE.match(line.text)
        if m and _normalize(m.group(1)) in wanted:
            stmt.start, stmt.run_in = i, True
            return


def _extent(ctx: CheckContext, stmt: _Statement, starts: list[int]) -> list[Line]:
    """The statement's lines: from its head up to the next heading or statement."""
    assert stmt.start is not None
    end = min((i for i in starts if i > stmt.start), default=len(ctx.doc.reading_order))
    return ctx.doc.reading_order[stmt.start:end]


def _body_text(lines: list[Line]) -> str:
    return " ".join(" ".join(ln.text for ln in lines).split())


def _page_span(ctx: CheckContext, lines: list[Line]) -> float:
    """How many text-block heights the lines occupy, summed over their pages."""
    furniture = re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))
    top = float(ctx.conf("geometry.margin_top_pt", 72.0))
    bottom = float(ctx.conf("geometry.margin_bottom_pt", 72.0))
    by_page: dict[int, list[Line]] = {}
    for ln in lines:
        text = ln.text.strip()
        if text and not furniture.search(text):
            by_page.setdefault(ln.page, []).append(ln)
    total = 0.0
    for number, page_lines in by_page.items():
        block = ctx.doc.pages[number - 1].height - top - bottom
        if block <= 0:
            continue
        span = max(ln.bbox[3] for ln in page_lines) - min(ln.bbox[1] for ln in page_lines)
        total += span / block
    return total


def _missing(ctx: CheckContext, stmt: _Statement) -> Finding:
    names = ", ".join(repr(str(a)) for a in (stmt.spec.get("aliases") or [])[:3])
    why = str(stmt.spec.get("why", "")).strip()
    remedy = str(stmt.spec.get("remedy", "")).strip() or f"Add a section titled {stmt.title!r}."
    if stmt.required:
        return ctx.error(
            stmt.check_id, stmt.title,
            f"No {stmt.phrase} was found. {why or 'The venue requires one.'}",
            category="structure", remedy=remedy,
            confidence=f"high — searched for a heading or bold run-in head such as {names}",
        )
    return ctx.warn(
        stmt.check_id, stmt.title,
        f"No {stmt.phrase} was found. {why or 'The venue recommends one.'}",
        category="structure", remedy=remedy,
        confidence=f"high — searched for a heading or bold run-in head such as {names}",
    )


def _judge(ctx: CheckContext, stmt: _Statement, starts: list[int]) -> Finding:
    assert stmt.start is not None
    lines = _extent(ctx, stmt, starts)
    head = lines[0]
    body = _body_text(lines)
    problems: list[tuple[Severity, str, list[Evidence]]] = []

    # Template boilerplate: a required statement still in placeholder form has
    # not actually been written, so it is as missing as no statement at all.
    leftovers = [
        m.group(0) for p in (stmt.spec.get("template_text") or [])
        if (m := re.search(str(p), body, flags=re.IGNORECASE))
    ]
    if leftovers:
        problems.append((
            Severity.ERROR if stmt.required else Severity.WARNING,
            f"The {stmt.phrase} still contains the template's placeholder or instruction text "
            f"({len(leftovers)} passage(s)), so it has not been written for this paper.",
            [Evidence(page=head.page, detail="template text left in place", quote=q)
             for q in leftovers[:4]],
        ))

    if stmt.spec.get("before_references"):
        refs = ctx.doc.find_heading([str(a) for a in (ctx.conf("structure.references_aliases", []) or [])])
        refs_at = None if refs is None else _line_index(ctx, refs.page, refs.bbox[1])
        if refs is not None and refs_at is not None and stmt.start > refs_at:
            problems.append((
                Severity.WARNING,
                f"The {stmt.phrase} (page {head.page}) comes after the references (page {refs.page}); "
                "the venue places it at the end of the main text, before the references.",
                [Evidence(page=head.page, detail=f"{stmt.title} starts here"),
                 Evidence(page=refs.page, detail=f"references start here: {refs.text!r}")],
            ))

    max_pages = stmt.spec.get("max_pages")
    # A run-in statement has no heading to end on, so its extent is a guess.
    if max_pages is not None and not stmt.run_in:
        pages = _page_span(ctx, lines)
        if pages > float(max_pages) + 0.05:
            problems.append((
                Severity.WARNING,
                f"The {stmt.phrase} runs about {pages:.1f} pages; the venue caps it at {max_pages}.",
                [Evidence(page=head.page, detail="length up to the next heading",
                          measured=round(pages, 1), expected=f"<= {max_pages} page(s)")],
            ))

    where = f"page {head.page}" + (" (as a run-in paragraph head)" if stmt.run_in else "")
    if not problems:
        return ctx.ok(stmt.check_id, stmt.title, f"{stmt.title} found on {where}.",
                      category="structure",
                      evidence=[Evidence(page=head.page, detail=f"head {head.text.strip()[:60]!r}")])
    severity = min((p[0] for p in problems), key=lambda s: s.rank)
    return ctx.finding(
        stmt.check_id, stmt.title, severity,
        f"{stmt.title} found on {where}. " + " ".join(p[1] for p in problems),
        category="structure",
        evidence=[e for p in problems for e in p[2]],
        remedy=str(stmt.spec.get("remedy", "")).strip() or None,
    )


def _statements(ctx: CheckContext) -> tuple[list[_Statement], list[int]]:
    """Every declared statement, located once per run, and where sections start."""
    cached = ctx.shared.get("statements")
    if cached is not None:
        return cached
    specs = ctx.conf("statements", {}) or {}
    stmts = [_Statement(str(k), v) for k, v in specs.items() if isinstance(v, dict)]
    for stmt in stmts:
        _locate(ctx, stmt)
    # Each statement ends at the next heading, or at the next statement's head
    # when several are set as run-in paragraphs under one heading.
    starts = sorted({*_heading_indices(ctx), *(s.start for s in stmts if s.start is not None)})
    ctx.shared["statements"] = (stmts, starts)
    return stmts, starts


def statement_text(ctx: CheckContext, key: str) -> tuple[int, str] | None:
    """(page, text) of a declared statement, head included, or None if absent."""
    stmts, starts = _statements(ctx)
    stmt = next((s for s in stmts if s.key == key and s.start is not None), None)
    if stmt is None:
        return None
    lines = _extent(ctx, stmt, starts)
    return lines[0].page, _body_text(lines)


@register("statements", "Required and recommended statements", module=MODULE,
          category="structure", order=39)
def check_statements(ctx: CheckContext) -> list[Finding]:
    """One finding per statement the venue declares: present, finished, placed, sized."""
    stmts, starts = _statements(ctx)
    return [_judge(ctx, s, starts) if s.start is not None else _missing(ctx, s) for s in stmts]
