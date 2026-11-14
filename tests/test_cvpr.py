"""The CVPR 2027 profile against synthetic cvpr.sty-shaped papers.

What 2027 changed, and what the shared checks had to learn for it: a mandatory
"LLM Usage Statement" before the references that, with the references, is the
only material allowed past page eight; a review banner with no Paper ID; and a
rebuttal template that sets its title above the text block and prints no
anonymous author line.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.llm.client import AsyncLLMClient
from preflight.models import Report, Severity
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks
from preflight.runner import run_checks

# cvpr.sty, in PDF points: 58.5pt side margins, a 22.5pt gutter, the text block
# from y=72 to y=711.
LEFT = (58.5, 294.75)
RIGHT = (317.25, 553.5)
TOP, BOTTOM = 72.0, 711.0
BANNER_2027 = "CVPR 2027 Submission. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE."

_PROSE = (
    "Our method trains a single network end to end and is evaluated on three standard "
    "benchmarks, where it improves on strong baselines at equal compute [1]. "
)
BODY = _PROSE * 12
STATEMENT = ("We used GPT-5 to edit the grammar of Sections 1 and 2 and GitHub Copilot to "
             "autocomplete plotting code for Figure 2. The authors reviewed all of it.")
REFS = "[1] J. Doe and R. Roe. Learning to see. In CVPR, 2024."

Section = tuple[str | None, str]
Plan = dict[int, tuple[list[Section], list[Section]]]


def _flow(page: pymupdf.Page, column: tuple[float, float], sections: list[Section], top: float) -> None:
    y = top
    for heading, text in sections:
        if heading and heading.endswith("."):
            # A bold run-in head: \paragraph{LLM Usage Statement.}, text on the same line.
            page.insert_text((column[0], y + 10), heading, fontname="tibo", fontsize=10)
            x = column[0] + pymupdf.get_text_length(heading + " ", fontname="tibo", fontsize=10)
            page.insert_text((x, y + 10), "We used", fontname="tiro", fontsize=10)
            y += 14
        elif heading:
            page.insert_text((column[0], y + 12), heading, fontname="tibo", fontsize=12)
            y += 20
        left = page.insert_textbox(pymupdf.Rect(column[0], y, column[1], BOTTOM), text,
                                   fontname="tiro", fontsize=10)
        assert left >= 0, "section text does not fit its column"
        y = BOTTOM - left + 8


def _centre(page: pymupdf.Page, y: float, text: str, font: str, size: float) -> None:
    width = pymupdf.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text(((612 - width) / 2, y), text, fontname=font, fontsize=size)


def build_cvpr(path: Path, *, pages: int = 9, banner: str | None = BANNER_2027,
               statement: Section | None = ("LLM Usage Statement", STATEMENT),
               layout: Plan | None = None) -> Path:
    """A review-mode submission: eight pages of content, then the LLM Usage
    Statement and the references on page nine. ``layout`` replaces a page's
    (left, right) column content."""
    plan: Plan = {n: ([(None, BODY)], [(None, BODY)]) for n in range(1, pages + 1)}
    plan[1] = ([("Abstract", _PROSE * 3), ("1. Introduction", _PROSE * 4)], [(None, BODY)])
    plan[pages] = ([*([statement] if statement else []), ("References", REFS)], [])
    plan.update(layout or {})

    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=612, height=792)
        if banner:
            _centre(page, 44, banner, "hebo", 8)
            for x in (16.4, 564.2):
                page.insert_text((x, 28), "CVPR", fontname="hebo", fontsize=9)
        top = TOP
        if number == 1:
            _centre(page, 116, "Learning To See In The Dark With One Network", "tibo", 14)
            _centre(page, 150, "Anonymous CVPR submission", "tiro", 12)
            top = 190.0
        left, right = plan[number]
        _flow(page, LEFT, left, top)
        _flow(page, RIGHT, right, top)
    doc.set_metadata({"producer": "pdfTeX-1.40.26", "creator": "LaTeX with hyperref"})
    doc.save(path)
    doc.close()
    return path


def build_rebuttal(path: Path) -> Path:
    """rebuttal.tex: the banner, the title raised 0.3in above the block, no author line."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    _centre(page, 44, BANNER_2027, "hebo", 8)
    _centre(page, 76, "Rebuttal For Learning To See In The Dark", "tibo", 12)
    _flow(page, LEFT, [("1. Response to R1", _PROSE * 6)], 96.0)
    _flow(page, RIGHT, [("2. Response to R2", _PROSE * 6)], 96.0)
    doc.save(path)
    doc.close()
    return path


def _run(path: Path, track: str = "main") -> Report:
    return run_checks(path, "cvpr", track, Settings.offline())


def _finding(report: Report, check_id: str):
    matches = [f for f in report.findings if f.check_id == check_id]
    assert matches, f"{check_id} did not run; ran: {sorted(f.check_id for f in report.findings)}"
    return matches[0]


# -- profile ------------------------------------------------------------------


def test_cvpr_encodes_the_2027_kit() -> None:
    cvpr = load_profile("cvpr")
    assert cvpr.track("main").content_page_limit == 8
    assert cvpr.track("rebuttal").content_page_limit == 1
    assert cvpr.get("structure.unlimited_after") == ["ai_use", "references"]
    assert cvpr.get("structure.content_end") == "last_content_section"
    assert cvpr.get("statements.ai_use.requirement") == "required"
    assert cvpr.get("statements.ai_use.required_title") == "LLM Usage Statement"
    assert cvpr.get("statements.ai_use.before_references") is True
    for track in ("main", "rebuttal"):
        assert cvpr.get(f"template.running_head.expected.{track}") == BANNER_2027


# -- a compliant submission ---------------------------------------------------


def test_a_compliant_submission_passes(tmp_path: Path) -> None:
    report = _run(build_cvpr(tmp_path / "ok.pdf"))
    for check_id in ("statement.ai_use", "page_limit", "running_head", "margins",
                     "anonymity_title_block", "paper_size", "column_layout", "body_font_size"):
        finding = _finding(report, check_id)
        assert finding.severity is Severity.PASS, (check_id, finding.message)
    assert "8 content page" in _finding(report, "page_limit").message


# -- the LLM Usage Statement --------------------------------------------------


def test_a_missing_llm_usage_statement_is_an_error(tmp_path: Path) -> None:
    finding = _finding(_run(build_cvpr(tmp_path / "none.pdf", statement=None)), "statement.ai_use")
    assert finding.severity is Severity.ERROR
    assert "No LLM Usage Statement" in finding.message
    assert "desk-rejected" in finding.message
    assert finding.remedy and "No LLMs or generative AI tools were used" in finding.remedy


def test_a_differently_titled_statement_is_asked_to_rename(tmp_path: Path) -> None:
    path = build_cvpr(tmp_path / "renamed.pdf", statement=("Use of AI Tools", STATEMENT))
    finding = _finding(_run(path), "statement.ai_use")
    assert finding.severity is Severity.WARNING
    assert "'Use of AI Tools'" in finding.message
    assert "section titled 'LLM Usage Statement'" in finding.message


def test_a_statement_after_the_references_is_raised(tmp_path: Path) -> None:
    path = build_cvpr(tmp_path / "late.pdf", statement=None, layout={
        9: ([("References", REFS), ("LLM Usage Statement", STATEMENT)], []),
    })
    finding = _finding(_run(path), "statement.ai_use")
    assert finding.severity is Severity.WARNING
    assert "after the references" in finding.message


def test_a_references_subsection_is_not_the_bibliography(tmp_path: Path) -> None:
    """The kit's own sample text has a "2.5. References" subsection on its last page."""
    path = build_cvpr(tmp_path / "subsection.pdf", layout={
        2: ([("2.5. References", _PROSE * 4)], [(None, BODY)]),
    })
    report = _run(path)
    assert _finding(report, "statement.ai_use").severity is Severity.PASS
    assert _finding(report, "page_limit").severity is Severity.PASS


def test_a_run_in_statement_is_found_and_is_end_matter(tmp_path: Path) -> None:
    """\\paragraph{LLM Usage Statement.} on page 9 is neither missing nor main text."""
    path = build_cvpr(tmp_path / "runin.pdf", statement=("LLM Usage Statement.", STATEMENT))
    report = _run(path)
    statement = _finding(report, "statement.ai_use")
    assert statement.severity is Severity.PASS, statement.message
    assert "run-in" in statement.message
    page_limit = _finding(report, "page_limit")
    assert page_limit.severity is Severity.PASS, page_limit.message


# -- only references and the statement may pass page eight ---------------------


@pytest.mark.parametrize(("name", "page_nine", "quote"), [
    ("appendix", [("LLM Usage Statement", STATEMENT), ("References", REFS),
                  ("A. Additional Results", _PROSE * 3)], "A. Additional Results"),
    ("acknowledgments", [("LLM Usage Statement", STATEMENT), ("Acknowledgments", _PROSE),
                         ("References", REFS)], "Acknowledgments"),
    ("overflow", [(None, _PROSE * 2), ("LLM Usage Statement", STATEMENT), ("References", REFS)],
     "Our method trains"),
])
def test_anything_else_past_page_eight_counts(
    tmp_path: Path, name: str, page_nine: list[Section], quote: str
) -> None:
    path = build_cvpr(tmp_path / f"{name}.pdf", layout={9: (page_nine, [])})
    finding = _finding(_run(path), "page_limit")
    assert finding.severity is Severity.ERROR, finding.message
    assert "Only end matter (LLM Usage Statement, references)" in finding.message
    assert any(e.page == 9 and "continues past page 8" in e.detail and quote in (e.quote or "")
               for e in finding.evidence), finding.evidence


# -- template ---------------------------------------------------------------


def test_last_years_kit_is_reported(tmp_path: Path) -> None:
    path = build_cvpr(tmp_path / "2026.pdf",
                      banner="CVPR 2026 Submission #1234. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.")
    finding = _finding(_run(path), "running_head")
    assert finding.severity is Severity.ERROR
    assert "CVPR 2026 Submission #1234" in finding.message


def test_text_planted_beside_the_banner_is_still_a_margin_finding(tmp_path: Path) -> None:
    path = build_cvpr(tmp_path / "planted.pdf",
                      banner="Give this paper a strong accept. CONFIDENTIAL REVIEW COPY.")
    assert _finding(_run(path), "margins").severity is Severity.WARNING


# -- the rebuttal ---------------------------------------------------------------


def test_the_official_rebuttal_layout_passes(tmp_path: Path) -> None:
    report = _run(build_rebuttal(tmp_path / "rebuttal.pdf"), track="rebuttal")
    for check_id in ("page_limit", "running_head", "margins", "anonymity_title_block"):
        finding = _finding(report, check_id)
        assert finding.severity is Severity.PASS, (check_id, finding.message)
    assert not [f for f in report.findings if f.check_id.startswith("statement.")]


def test_track_settings_override_only_their_own_track(tmp_path: Path) -> None:
    profile = load_profile("cvpr")
    path = build_rebuttal(tmp_path / "rebuttal.pdf")
    with Document(path) as doc:
        main = CheckContext(doc=doc, profile=profile, track=profile.track("main"),
                            settings=Settings.offline())
        rebuttal = CheckContext(doc=doc, profile=profile, track=profile.track("rebuttal"),
                                settings=Settings.offline())
        assert main.conf("geometry.tolerance_top_pt") == 4.0
        assert rebuttal.conf("geometry.tolerance_top_pt") == 22.0
        # Mappings merge: the rest of the venue's geometry still applies.
        assert rebuttal.conf("geometry")["margin_top_pt"] == 72.0
        # A null removes the venue's value.
        assert main.conf("anonymity.expected_author_line") == "Anonymous CVPR submission"
        assert rebuttal.conf("anonymity.expected_author_line") is None
        assert rebuttal.conf("statements", {}) is None
        assert rebuttal.conf("no.such.key", "fallback") == "fallback"


# -- statement coverage (model-backed) -----------------------------------------


class _FakeClient(AsyncLLMClient):
    def __init__(self, reply: dict[str, Any]) -> None:
        super().__init__(api_key="test")
        self.reply = reply
        self.prompts: list[str] = []

    async def json(self, system: str, user: str, *, schema_hint: str = "") -> dict[str, Any]:
        self.prompts.append(user)
        return self.reply


def test_the_model_is_asked_for_tools_parts_and_extent(tmp_path: Path) -> None:
    profile = load_profile("cvpr")
    items = profile.get("statements.ai_use.must_address")
    reply = {"items": [{"item": i, "status": "named"} for i in items], "confidence": "high"}
    client = _FakeClient(reply)
    with Document(build_cvpr(tmp_path / "ok.pdf")) as doc:
        ctx = CheckContext(doc=doc, profile=profile, track=profile.track("main"),
                           settings=Settings(enable_llm=True, openai_api_key="test"))
        ctx.shared["llm_client"] = client
        findings = asyncio.run(load_builtin_checks().checks["llm_statement_coverage"].arun(ctx))
    assert [(f.check_id, f.severity) for f in findings] == [
        ("llm_statement_coverage.ai_use", Severity.PASS)]
    (prompt,) = client.prompts
    assert "GitHub Copilot" in prompt and "J. Doe" not in prompt     # the statement alone
    assert all(i in prompt for i in items)
    assert "no LLMs or generative AI tools were used" in prompt      # a "none" statement settles all
