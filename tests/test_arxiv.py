"""The arXiv profile: no format rules, the content checks kept, and authors named.

arXiv takes any template at any length, so none of the page-measuring checks may
report on a preprint. What it does refuse is an anonymous paper, which in
practice means the review build of a conference template.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from preflight.context import Settings
from preflight.models import Report, Severity
from preflight.picker import _choices, _track_choices
from preflight.profile import ProfileError, load_profile
from preflight.registry import describe, load_builtin_checks
from preflight.runner import run_checks

#: The modules that measure a page against a template, and nothing else.
FORMAT_MODULES = {"core.geometry", "core.fonts", "core.template"}

_PROSE = (
    "We study how retrieval changes what a language model cites, and measure the effect on "
    "twelve benchmarks with three model families. The gains hold across sizes (Doe and Roe, "
    "2021), and we release the evaluation harness. "
)
BODY = _PROSE * 12
REFS = "Jane Doe and John Roe. 2021. A paper about things. In Proceedings of ACL, pages 1-9."


def _centre(page: pymupdf.Page, y: float, text: str, font: str, size: float) -> None:
    width = pymupdf.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text(((page.rect.width - width) / 2, y), text, fontname=font, fontsize=size)


def build_preprint(
    path: Path,
    *,
    pages: int = 3,
    title: str = "Retrieval Changes What Language Models Cite",
    authors: tuple[str, ...] = ("Jane Doe, John Roe", "University of Somewhere"),
    size: tuple[float, float] = (612.0, 792.0),
    body_size: float = 10.0,
) -> Path:
    """A single-column preprint: title, author lines, abstract, body, references.

    ``size`` and ``body_size`` default to NeurIPS's, but nothing here is a template;
    the tests bend both to show that arXiv does not care.
    """
    doc = pymupdf.open()
    width, height = size
    column = pymupdf.Rect(108, 72, width - 108, height - 72)
    for number in range(1, pages + 1):
        page = doc.new_page(width=width, height=height)
        top = column.y0
        if number == 1:
            _centre(page, 100, title, "tibo", 17)
            y = 140.0
            for line in authors:
                _centre(page, y, line, "tiro", 10)
                y += 13
            _centre(page, y + 20, "Abstract", "tibo", 12)
            page.insert_textbox(pymupdf.Rect(144, y + 28, width - 144, y + 140), _PROSE * 3,
                                fontname="tiro", fontsize=body_size)
            page.insert_text((column.x0, y + 160), "1 Introduction", fontname="tibo", fontsize=12)
            top = y + 170
        if number == pages:
            page.insert_text((column.x0, top + 10), "References", fontname="tibo", fontsize=12)
            page.insert_textbox(pymupdf.Rect(column.x0, top + 20, column.x1, column.y1), REFS,
                                fontname="tiro", fontsize=body_size)
            continue
        page.insert_textbox(pymupdf.Rect(column.x0, top, column.x1, column.y1), BODY,
                            fontname="tiro", fontsize=body_size)
    doc.set_metadata({"producer": "pdfTeX-1.40.26", "creator": "LaTeX with hyperref"})
    doc.save(path)
    doc.close()
    return path


def _run(path: Path) -> Report:
    return run_checks(path, "arxiv", None, Settings.offline())


def _finding(report: Report, check_id: str):
    matches = [f for f in report.findings if f.check_id == check_id]
    assert matches, f"{check_id} did not run; ran: {sorted(f.check_id for f in report.findings)}"
    return matches[0]


# -- profile ------------------------------------------------------------------


def test_arxiv_is_a_single_preprint_track_with_no_page_limit() -> None:
    arxiv = load_profile("arxiv")
    assert arxiv.lineage == ("arxiv", "base")
    assert set(arxiv.tracks) == {"preprint"}
    track = arxiv.track(None)
    assert track.content_page_limit is None and track.total_page_limit is None
    assert track.limit_label == "no limit"


def test_arxiv_runs_every_base_module_but_format() -> None:
    """Listed rather than inherited: a new base module must be placed on purpose."""
    base = set(load_profile("base").modules)
    assert set(load_profile("arxiv").modules) == base - FORMAT_MODULES


def test_arxiv_wants_authors_named_rather_than_hidden() -> None:
    arxiv = load_profile("arxiv")
    assert arxiv.get("anonymity.enabled") is False
    assert arxiv.get("anonymity.require_author_block") is True
    assert arxiv.check_enabled("author_block")
    assert "anonymity_semantics" not in arxiv.get("llm.checks")


def test_arxiv_keeps_the_reviewing_and_reference_checks() -> None:
    arxiv = load_profile("arxiv")
    for module in ("integrations.refcheck", "core.hidden", "core.markup", "core.citations",
                   "llm.semantic", "llm.review", "llm.audit", "llm.prose"):
        assert arxiv.module_enabled(module), module
    assert "injection_semantics" in arxiv.get("llm.checks")
    assert arxiv.get("llm.scoring") is True
    assert arxiv.get("refcheck.not_found_is_error") is True


def test_arxiv_disabled_checks_name_real_checks() -> None:
    load_builtin_checks()
    ids = {entry["id"] for entry in describe()}
    arxiv = load_profile("arxiv")
    assert set(arxiv.disabled_checks) <= ids
    assert set(arxiv.severity_overrides) <= ids


def test_the_picker_says_there_is_no_page_limit() -> None:
    arxiv = next(c for c in _choices() if c.key == "arxiv")
    assert arxiv.detail == "preprint (no limit)"
    assert _track_choices(load_profile("arxiv"))[0].name == "no page limit"


# -- no format rules ----------------------------------------------------------------


def test_any_size_length_and_type_is_accepted(tmp_path: Path) -> None:
    """A4 at 12pt over 14 pages: every format check would object, and none runs."""
    path = build_preprint(tmp_path / "long.pdf", pages=14, size=(595.0, 842.0), body_size=12.0)
    report = _run(path)
    ran = {f.check_id for f in report.findings}
    for check_id in ("paper_size", "margins", "column_layout", "body_font_size", "min_font_size",
                     "font_family", "page_limit", "total_page_limit", "limitations_present",
                     "limitations_position", "limitations_scope", "running_head", "page_numbers",
                     "anonymity_title_block", "anonymity_emails", "anonymity_urls",
                     "anonymity_metadata"):
        assert check_id not in ran, check_id
    assert not report.errors, [(f.check_id, f.message) for f in report.errors]
    assert report.meta["content_page_limit"] is None
    assert "page limit" not in _finding(report, "references_position").message


# -- authors named ------------------------------------------------------------------


@pytest.mark.parametrize("authors", [
    ("Anonymous Author(s)", "Affiliation", "Address", "email"),         # NeurIPS
    ("Anonymous authors", "Paper under double-blind review"),          # ICLR
    ("Anonymous Authors1",),                                           # ICML
    ("Anonymous ACL submission",),                                     # ACL
    ("Anonymous CVPR submission", "Paper ID 1234"),                    # CVPR
])
def test_a_review_build_is_an_error(tmp_path: Path, authors: tuple[str, ...]) -> None:
    finding = _finding(_run(build_preprint(tmp_path / "review.pdf", authors=authors)), "author_block")
    assert finding.severity is Severity.ERROR
    assert "placeholder" in finding.message
    assert "no anonymous submissions" in (finding.cfp_reference or "")


def test_a_blank_author_block_is_an_error(tmp_path: Path) -> None:
    finding = _finding(_run(build_preprint(tmp_path / "blank.pdf", authors=())), "author_block")
    assert finding.severity is Severity.ERROR
    assert "No author names" in finding.message


@pytest.mark.parametrize("authors", [
    ("Jane Doe, John Roe", "University of Somewhere"),
    # The placeholders are anchored and case-sensitive, so a paper about
    # anonymity keeps its authors.
    ("Jane Doe", "Center for Anonymous Communication, University of Somewhere"),
    ("Jane Doe", "anonymous systems group"),
])
def test_named_authors_pass(tmp_path: Path, authors: tuple[str, ...]) -> None:
    finding = _finding(_run(build_preprint(tmp_path / "named.pdf", authors=authors)), "author_block")
    assert finding.severity is Severity.PASS, finding.message


def test_a_paper_about_anonymity_is_not_a_review_build(tmp_path: Path) -> None:
    path = build_preprint(tmp_path / "topic.pdf", title="Anonymous Authors Considered Harmful")
    assert _finding(_run(path), "author_block").severity is Severity.PASS


# -- profile parsing ------------------------------------------------------------------


def test_a_track_must_still_say_it_has_no_limit(tmp_path: Path) -> None:
    """`content_page_limit: null` is a choice; leaving the key out is a mistake."""
    ok = tmp_path / "open.yaml"
    ok.write_text("conference: {key: open, extends: base, default_track: any}\ntracks_replace: true\n"
                  "tracks: {any: {content_page_limit: null}}\n")
    assert load_profile(str(ok)).track("any").content_page_limit is None

    missing = tmp_path / "missing.yaml"
    missing.write_text("conference: {key: missing, extends: base, default_track: any}\n"
                       "tracks_replace: true\ntracks: {any: {description: forgot}}\n")
    with pytest.raises(ProfileError, match="content_page_limit"):
        load_profile(str(missing))
