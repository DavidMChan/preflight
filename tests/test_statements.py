"""Venue-declared statements, the template running head, and the abstract's paragraphs.

The papers here follow the ICLR 2027 template: US Letter, one column at x=108,
10pt body, small-caps headings, and the AI use / ethics / reproducibility
statements set as unnumbered subsections before the references.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from conftest import _small_caps, build_paper

from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.llm.client import AsyncLLMClient
from preflight.models import Report, Severity
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks
from preflight.runner import run_checks

HEAD_2027 = "Under review as a conference paper at ICLR 2027"
LEFT, RIGHT, TOP, BOTTOM = 108.0, 504.0, 82.0, 730.0

_BODY = (
    "Activation steering constructs a direction from examples of a behavior. We find that "
    "sequences sampled at random can replace those examples across four model families. "
) * 40

AI_USE = (
    "In this work, we used generative AI tools to implement methods and to edit the paper for "
    "readability. We have not used generative AI tools to generate synthetic data sets, develop "
    "theoretical models, formulate or prove mathematical claims, write proofs, propose "
    "hypotheses, design experiments, clean datasets, analyse qualitative data or interpret "
    "results, and translation is not applicable to this work. We have reviewed all AI-assisted "
    "work and take responsibility for the final content."
)
AI_USE_TEMPLATE = (
    "(This section is required and does not count toward the page limit.) In this work, we used "
    "generative AI tools for [tasks with required disclosure]. We have not used generative AI "
    "tools for [other tasks with required disclosure], and [the rest of the required disclosure "
    "tasks] are not applicable to this work. [Elaborate. For example, we checked the code.]"
)
ETHICS = "The study uses only public benchmarks and involves no human subjects or personal data."
REPRO = "Appendix A lists every prompt and hyperparameter, and anonymous code is in the supplement."


def _heading(page: pymupdf.Page, y: float, text: str, size: float = 10.0) -> None:
    """Set ``text`` as \\textsc does: capitals full size, lower case as small capitals."""
    x = LEFT
    for run in re.findall(r"[A-Z]+|[^A-Z]+", text):
        glyphs, fontsize = (run, size) if run.isupper() else (run.upper(), size * 0.8)
        page.insert_text((x, y), glyphs, fontname="tiro", fontsize=fontsize)
        x += pymupdf.get_text_length(glyphs, fontname="tiro", fontsize=fontsize)


def _text(page: pymupdf.Page, y: float, text: str, bottom: float = BOTTOM) -> float:
    """Set a paragraph at ``y``; return the y just below it."""
    height = bottom - y
    left = page.insert_textbox(pymupdf.Rect(LEFT, y, RIGHT, bottom), text, fontname="tiro", fontsize=10)
    assert left >= 0, "paragraph did not fit"
    return y + height - left + 4


def build_iclr_paper(
    path: Path,
    *,
    content_pages: int = 8,
    statements: list[tuple[str, str]] | None = None,
    statements_after_references: bool = False,
    run_in: bool = False,
    running_head: str | None = HEAD_2027,
    abstract_paragraphs: int = 1,
    statement_pages: int = 0,
) -> Path:
    """``content_pages`` full pages of main text, then the statements and references.

    ``statement_pages`` extra full pages of text are added to the first statement,
    to push it past a length cap.
    """
    statements = [] if statements is None else statements
    doc = pymupdf.open()

    def new_page() -> pymupdf.Page:
        page = doc.new_page(width=612, height=792)
        if running_head:
            page.insert_text((LEFT, 40), running_head, fontname="tiro", fontsize=10)
        return page

    page = new_page()
    x = LEFT
    for word in ("Driverless", "Steering"):
        x = _small_caps(page, x, 100, word, 17.2, 13.8) + 5
    page.insert_text((LEFT, 140), "Anonymous authors", fontname="tibo", fontsize=10)
    _small_caps(page, 280, 175, "Abstract", 12.0, 9.6)
    y = 185.0
    abstract = ("We study whether random sequences can stand in for examples when steering a "
                "model toward a behavior, and show that they can in most settings we test. ")
    for _ in range(abstract_paragraphs):
        y = _text(page, y, abstract * 3, bottom=y + 70) + 4    # half a line between paragraphs
    page.insert_text((LEFT, y + 20), "1", fontname="tiro", fontsize=12)
    _small_caps(page, 127, y + 20, "Introduction", 12.0, 9.6)
    _text(page, y + 30, _BODY[:1800])
    for _ in range(content_pages - 1):
        _text(new_page(), TOP, _BODY[:3400])

    def put_statements(page: pymupdf.Page, y: float) -> tuple[pymupdf.Page, float]:
        for i, (title, body) in enumerate(statements):
            if y > BOTTOM - 60:
                page, y = new_page(), TOP + 10
            if run_in:
                page.insert_text((LEFT, y + 10), title + ".", fontname="tibo", fontsize=10)
                offset = pymupdf.get_text_length(title + ". ", fontname="tibo", fontsize=10)
                page.insert_text((LEFT + offset, y + 10), body[:60], fontname="tiro", fontsize=10)
                y = _text(page, y + 14, body[60:])
            else:
                _heading(page, y + 10, title)
                y = _text(page, y + 16, body)
            for _ in range(statement_pages if i == 0 else 0):
                page = new_page()
                y = _text(page, TOP, _BODY[:3400])
        return page, y

    page = new_page()
    y = TOP + 10
    if not statements_after_references:
        page, y = put_statements(page, y)
    if y > BOTTOM - 60:
        page, y = new_page(), TOP + 10
    _heading(page, y + 14, "References", size=12.0)
    y = _text(page, y + 22, "Jane Doe. A paper about things. CoRR, 2021.")
    if statements_after_references:
        put_statements(page, y + 10)
    doc.save(path)
    doc.close()
    return path


ALL_THREE = [("AI use statement", AI_USE), ("Ethics statement", ETHICS),
             ("Reproducibility statement", REPRO)]


def _run(path: Path, track: str = "main", conference: str = "iclr") -> Report:
    return run_checks(path, conference, track, Settings.offline())


def _finding(report: Report, check_id: str):
    matches = [f for f in report.findings if f.check_id == check_id]
    assert matches, f"{check_id} did not run; ran: {sorted(f.check_id for f in report.findings)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


def test_statements_written_as_the_template_asks_pass(tmp_path: Path) -> None:
    report = _run(build_iclr_paper(tmp_path / "ok.pdf", statements=ALL_THREE))
    for key in ("ai_use", "ethics", "reproducibility"):
        finding = _finding(report, f"statement.{key}")
        assert finding.severity is Severity.PASS, finding.message
    assert _finding(report, "page_limit").severity is Severity.PASS


def test_a_missing_ai_use_statement_is_an_error(tmp_path: Path) -> None:
    """ICLR names no penalty, but the section is required: that is enough to error."""
    report = _run(build_iclr_paper(tmp_path / "none.pdf"))
    ai_use = _finding(report, "statement.ai_use")
    assert ai_use.severity is Severity.ERROR
    assert "No AI use statement" in ai_use.message
    assert ai_use.remedy
    # Recommended statements are still raised, one level down.
    assert _finding(report, "statement.ethics").severity is Severity.WARNING
    assert _finding(report, "statement.reproducibility").severity is Severity.WARNING


def test_template_placeholders_mean_the_statement_was_not_written(tmp_path: Path) -> None:
    path = build_iclr_paper(tmp_path / "boilerplate.pdf",
                            statements=[("AI use statement", AI_USE_TEMPLATE), *ALL_THREE[1:]])
    finding = _finding(_run(path), "statement.ai_use")
    assert finding.severity is Severity.ERROR
    assert "placeholder" in finding.message
    assert any("tasks with required disclosure" in (e.quote or "") for e in finding.evidence)


def test_statements_do_not_count_toward_the_page_limit(tmp_path: Path) -> None:
    """Nine full pages, then the AI use statement: page 10 is not main text.

    Before the statement had an alias, the page count ran on to the References
    heading and reported ten content pages as a desk-rejection error.
    """
    path = build_iclr_paper(tmp_path / "nine.pdf", content_pages=9,
                            statements=[("AI use statement", AI_USE)])
    finding = _finding(_run(path), "page_limit")
    assert finding.severity is Severity.PASS, finding.message
    assert "9 content page" in finding.message
    assert "AI USE STATEMENT" in finding.message.upper()


def test_a_statement_after_the_references_is_raised(tmp_path: Path) -> None:
    path = build_iclr_paper(tmp_path / "late.pdf", statements=ALL_THREE, statements_after_references=True)
    finding = _finding(_run(path), "statement.ethics")
    assert finding.severity is Severity.WARNING
    assert "after the references" in finding.message


def test_a_statement_over_its_page_cap_is_raised(tmp_path: Path) -> None:
    path = build_iclr_paper(tmp_path / "long.pdf", statements=ALL_THREE, statement_pages=2)
    finding = _finding(_run(path), "statement.ai_use")
    assert finding.severity is Severity.WARNING
    assert "caps it at 1" in finding.message
    # The ethics statement that follows is short, so its own finding is clean.
    assert _finding(_run(path), "statement.ethics").severity is Severity.PASS


def test_a_run_in_statement_head_counts(tmp_path: Path) -> None:
    """\\paragraph{AI use statement} sets a bold run-in head, not a heading."""
    path = build_iclr_paper(tmp_path / "runin.pdf", statements=ALL_THREE, run_in=True)
    report = _run(path)
    for key in ("ai_use", "ethics", "reproducibility"):
        finding = _finding(report, f"statement.{key}")
        assert finding.severity is Severity.PASS, finding.message
        assert "run-in" in finding.message


def test_venues_without_statements_are_untouched(clean_paper: Path) -> None:
    report = _run(clean_paper, track="long", conference="arr")
    ids = {f.check_id for f in report.findings}
    assert not [i for i in ids if i.startswith("statement.")]
    assert "running_head" not in ids and "abstract_paragraphs" not in ids


# ---------------------------------------------------------------------------
# Running head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("head", "track", "expected", "phrase"),
    [
        (HEAD_2027, "main", Severity.PASS, "found on"),
        ("Under review as a conference paper at ICLR 2026", "main", Severity.ERROR, "ICLR 2026"),
        ("Published as a conference paper at ICLR 2027", "main", Severity.ERROR, "author names"),
        ("Published as a conference paper at ICLR 2027", "camera_ready", Severity.PASS, "found on"),
        (HEAD_2027, "camera_ready", Severity.ERROR, "final-copy switch"),
        (None, "main", Severity.ERROR, "No running head"),
    ],
)
def test_running_head_names_the_year_and_mode(
    tmp_path: Path, head: str | None, track: str, expected: Severity, phrase: str
) -> None:
    path = build_iclr_paper(tmp_path / "head.pdf", statements=ALL_THREE, running_head=head)
    finding = _finding(_run(path, track=track), "running_head")
    assert finding.severity is expected, finding.message
    assert phrase in finding.message


# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("paragraphs", "expected"), [(1, Severity.PASS), (2, Severity.WARNING)])
def test_the_abstract_is_one_paragraph(tmp_path: Path, paragraphs: int, expected: Severity) -> None:
    path = build_iclr_paper(tmp_path / "abstract.pdf", statements=ALL_THREE,
                            abstract_paragraphs=paragraphs)
    finding = _finding(_run(path), "abstract_paragraphs")
    assert finding.severity is expected, finding.message


def test_two_column_abstracts_are_not_judged_without_a_limit(tmp_path: Path) -> None:
    path = build_paper(tmp_path / "acl.pdf", pages=9)
    assert "abstract_paragraphs" not in {f.check_id for f in _run(path, "long", "arr").findings}


# ---------------------------------------------------------------------------
# Coverage of the required task list (model-backed)
# ---------------------------------------------------------------------------


class _FakeClient(AsyncLLMClient):
    def __init__(self, reply: dict[str, Any]) -> None:
        super().__init__(api_key="test")
        self.reply = reply
        self.prompts: list[str] = []

    async def json(self, system: str, user: str, *, schema_hint: str = "") -> dict[str, Any]:
        self.prompts.append(user)
        return self.reply


def _coverage(path: Path, reply: dict[str, Any]) -> tuple[list, _FakeClient]:
    profile = load_profile("iclr")
    ctx = CheckContext(doc=Document(path), profile=profile, track=profile.track("main"),
                       settings=Settings(enable_llm=True, openai_api_key="test"))
    client = _FakeClient(reply)
    ctx.shared["llm_client"] = client
    try:
        check = load_builtin_checks().checks["llm_statement_coverage"]
        return asyncio.run(check.arun(ctx)), client
    finally:
        ctx.doc.close()


def test_the_model_sees_the_statement_and_every_required_task(tmp_path: Path) -> None:
    path = build_iclr_paper(tmp_path / "ok.pdf", statements=ALL_THREE)
    tasks = load_profile("iclr").get("statements.ai_use.must_address")
    reply = {"items": [{"item": t, "status": "named"} for t in tasks], "confidence": "high"}
    findings, client = _coverage(path, reply)
    assert [f.severity for f in findings] == [Severity.PASS]
    assert findings[0].check_id == "llm_statement_coverage.ai_use"
    (prompt,) = client.prompts
    assert "implement methods" in prompt.lower()        # the statement itself
    assert all(t in prompt for t in tasks)                # all 12 tasks
    assert "Ethics statement" not in prompt               # and nothing after it


def test_unaddressed_and_skipped_tasks_are_raised(tmp_path: Path) -> None:
    path = build_iclr_paper(tmp_path / "partial.pdf", statements=ALL_THREE)
    tasks = load_profile("iclr").get("statements.ai_use.must_address")
    reply = {"items": [{"item": tasks[0], "status": "unaddressed"},
                       {"item": tasks[1], "status": "blanket"},
                       *({"item": t, "status": "named"} for t in tasks[2:-1])]}   # last one skipped
    (finding,), _ = _coverage(path, reply)
    assert finding.severity is Severity.WARNING
    assert "2 of 12 required items are not addressed" in finding.message
    assert tasks[0] in finding.message and tasks[-1] in finding.message
    assert "catch-all" in finding.message and tasks[1] in finding.message


def test_no_model_call_when_the_statement_is_missing(tmp_path: Path) -> None:
    findings, client = _coverage(build_iclr_paper(tmp_path / "none.pdf"), {})
    assert findings == [] and client.prompts == []
