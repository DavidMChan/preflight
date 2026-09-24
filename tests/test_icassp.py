"""The ICASSP profile against synthetic spconf-shaped papers.

ICASSP differs from the other venues in ways the shared checks had to learn: a
4 + 1 page rule where only named end matter may sit on page five, US Letter or
A4, a 10pt or 9pt (\\ninept) body, no blind review, and no page numbers.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pymupdf
import pytest

from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.models import Finding, Report, Severity
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks
from preflight.runner import run_checks

LETTER = (612.0, 792.0)
A4 = (595.0, 842.0)
LEFT = (54.0, 298.0)
RIGHT = (315.0, 558.0)
BOTTOM = 718.0

_PROSE = (
    "Speech and music place different demands on spectral resolution, and a front end tuned "
    "for one is rarely the best choice for the other. We measure that trade-off with linear "
    "probes and propose a fixed-budget alternative [1]. "
)
BODY = _PROSE * 5
ETHICS = "This is a numerical simulation study for which no ethical approval was required."
ACKS = ("No funding was received for conducting this study. The authors have no relevant "
        "financial or non-financial interests to disclose.")
REFS = '[1] D. E. Ingalls, "Image processing for experts," IEEE Trans. ASSP, vol. 36, 1988.'

Section = tuple[str | None, str]


def _flow(page: pymupdf.Page, column: tuple[float, float], sections: list[Section],
          top: float, size: float) -> None:
    y = top
    for heading, text in sections:
        if heading:
            page.insert_text((column[0], y + 10), heading, fontname="tibo", fontsize=size)
            y += 18
        left = page.insert_textbox(pymupdf.Rect(column[0], y, column[1], BOTTOM), text,
                                   fontname="tiro", fontsize=size)
        assert left >= 0, "section text does not fit its column"
        y = BOTTOM - left + 8


def _centre(page: pymupdf.Page, y: float, text: str, font: str, size: float,
            left: float, right: float) -> None:
    width = pymupdf.get_text_length(text, fontname=font, fontsize=size)
    page.insert_text(((left + right - width) / 2, y), text, fontname=font, fontsize=size)


def build_icassp(
    path: Path,
    *,
    pages: int = 5,
    title: str = "FIXED-BUDGET FRONT ENDS FOR SPEECH AND MUSIC",
    names: str = "Jane Doe, John Roe",
    address: str = "University of Somewhere",
    keywords: str = "speech, music, front ends, spectrograms",
    abstract: str = _PROSE * 3,
    body_size: float = 10.0,
    page_size: tuple[float, float] = LETTER,
    folios: bool = False,
    layout: dict[int, tuple[list[Section], list[Section]]] | None = None,
) -> Path:
    """Title block, abstract and index terms on page 1, body on pages 1-4, and by
    default the ethics and acknowledgment sections on page 4 with the references on
    page 5. ``layout`` replaces a page's (left, right) column content."""
    plan: dict[int, tuple[list[Section], list[Section]]] = {
        n: ([(None, BODY)], [(None, BODY)]) for n in range(1, pages + 1)
    }
    if pages >= 5:
        plan[4] = ([(None, BODY)], [("5. COMPLIANCE WITH ETHICAL STANDARDS", ETHICS),
                                    ("6. ACKNOWLEDGMENTS", ACKS)])
        plan[5] = ([("7. REFERENCES", REFS)], [])
    plan[1] = ([("1. INTRODUCTION", _PROSE)], [(None, BODY)])
    plan.update(layout or {})

    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=page_size[0], height=page_size[1])
        top = 76.0
        if number == 1:
            for y, text, font, size in ((104, title, "tibo", 12), (122, names, "tiit", 12),
                                        (138, address, "tiro", 12), (186, "ABSTRACT", "tibo", body_size)):
                if text:
                    _centre(page, y, text, font, size, *(LEFT if text == "ABSTRACT" else (54.0, 558.0)))
            left = page.insert_textbox(pymupdf.Rect(LEFT[0], 194, LEFT[1], 560), abstract,
                                       fontname="tiro", fontsize=body_size)
            assert left >= 0, "abstract does not fit"
            below = 560 - left + 16
            # Base-14 Times cannot encode spconf's em dash; a double hyphen reads the same.
            page.insert_text((LEFT[0], below), f"Index Terms-- {keywords}", fontname="tiro",
                             fontsize=body_size)
            top = below + 11
        left, right = plan[number]
        _flow(page, LEFT, left, top, body_size)
        # Both columns start below the title block, which spans the page.
        _flow(page, RIGHT, right, 76.0 if number > 1 else 176.0, body_size)
        if folios:
            page.insert_text((303, 760), str(number), fontname="tiro", fontsize=10)
    doc.set_metadata({"producer": "pdfTeX-1.40.26", "creator": "LaTeX with hyperref"})
    doc.save(path)
    doc.close()
    return path


def _run(path: Path) -> Report:
    return run_checks(path, "icassp", "main", Settings.offline())


def _finding(report: Report, check_id: str):
    matches = [f for f in report.findings if f.check_id == check_id]
    assert matches, f"{check_id} did not run; ran: {sorted(f.check_id for f in report.findings)}"
    return matches[0]


# -- profile ------------------------------------------------------------------


def test_icassp_encodes_the_paper_kit() -> None:
    icassp = load_profile("icassp")
    main = icassp.track("main")
    assert (main.content_page_limit, main.total_page_limit) == (4, 5)
    assert icassp.get("structure.content_end") == "last_content_section"
    assert icassp.get("structure.unlimited_after") == ["references", "acknowledgments", "ethics"]
    assert icassp.get("geometry.alternate_page_sizes.a4.page_width_pt") == 595.0
    assert icassp.get("fonts.body_sizes_pt") == [10.0, 9.0]
    assert icassp.get("anonymity.enabled") is False
    assert icassp.get("anonymity.require_author_block") is True
    assert icassp.get("pdf.require_font_subsetting") is True
    assert icassp.get("pdf.max_file_size_mb") == 5
    for key in ("ethical_compliance", "funding_disclosure"):
        assert icassp.get(f"statements.{key}.requirement") == "required"


def test_other_venues_keep_a_single_paper_size_and_no_total_cap() -> None:
    for key in ("arr", "icra", "neurips"):
        profile = load_profile(key)
        assert not profile.get("geometry.alternate_page_sizes")
        assert all(t.total_page_limit is None for t in profile.tracks.values())


# -- the 4 + 1 page rule --------------------------------------------------------


def test_a_compliant_paper_passes_the_icassp_rules(tmp_path: Path) -> None:
    report = _run(build_icassp(tmp_path / "clean.pdf"))
    for check_id in ("page_limit", "total_page_limit", "paper_size", "body_font_size",
                     "author_block", "page_numbers", "title_format", "index_terms",
                     "statement.ethical_compliance", "statement.funding_disclosure"):
        finding = _finding(report, check_id)
        assert finding.severity is Severity.PASS, (check_id, finding.message)
    assert "4 content page" in _finding(report, "page_limit").message


def test_an_appendix_on_the_fifth_page_is_content(tmp_path: Path) -> None:
    path = build_icassp(tmp_path / "appendix.pdf",
                        layout={5: ([("7. REFERENCES", REFS)], [("A. APPENDIX", _PROSE)])})
    finding = _finding(_run(path), "page_limit")
    assert finding.severity is Severity.ERROR
    assert any("'APPENDIX' continues past page 4" in e.detail for e in finding.evidence)


def test_early_end_matter_does_not_stop_the_count(tmp_path: Path) -> None:
    """Acknowledgments on page 2 still leaves the sections after them counted."""
    path = build_icassp(tmp_path / "early.pdf", layout={
        2: ([(None, BODY)], [("2. ACKNOWLEDGMENTS", ACKS), ("3. METHOD", _PROSE)]),
        4: ([(None, BODY)], [(None, BODY)]),
        5: ([(None, BODY)], [("7. REFERENCES", REFS)]),
    })
    finding = _finding(_run(path), "page_limit")
    assert finding.severity is Severity.ERROR
    assert any("'METHOD'" in e.detail and e.page == 5 for e in finding.evidence)


def test_a_sixth_page_breaks_the_total_cap(tmp_path: Path) -> None:
    path = build_icassp(tmp_path / "six.pdf", pages=6,
                        layout={6: ([(None, REFS)], [])})
    finding = _finding(_run(path), "total_page_limit")
    assert finding.severity is Severity.ERROR
    assert "6 pages" in finding.message


def test_a_heading_atop_the_right_column_is_not_the_left_columns_first_line(tmp_path: Path) -> None:
    """Both columns start on one baseline; the left column's text still precedes the heading."""
    path = build_icassp(tmp_path / "right-refs.pdf", pages=4,
                        layout={4: ([(None, BODY)], [("5. REFERENCES", REFS)])})
    finding = _finding(run_checks(path, "arr", "long", Settings.offline()), "page_limit")
    assert "4 content page" in finding.message, finding.message


# -- format -----------------------------------------------------------------------


@pytest.mark.parametrize(("page_size", "expected"),
                         [(A4, Severity.PASS), ((600.0, 800.0), Severity.ERROR)])
def test_letter_or_a4(tmp_path: Path, page_size: tuple[float, float], expected: Severity) -> None:
    report = _run(build_icassp(tmp_path / "size.pdf", page_size=page_size))
    assert _finding(report, "paper_size").severity is expected


def test_a4_margins_use_the_a4_text_block(tmp_path: Path) -> None:
    """On A4 the block keeps its top-left anchor, so the right margin narrows to 13 mm."""
    report = _run(build_icassp(tmp_path / "a4.pdf", page_size=A4))
    assert _finding(report, "margins").severity is Severity.PASS


@pytest.mark.parametrize(("size", "expected"), [(9.0, Severity.PASS), (11.0, Severity.ERROR)])
def test_ninept_is_an_accepted_body_size(tmp_path: Path, size: float, expected: Severity) -> None:
    report = _run(build_icassp(tmp_path / "ninept.pdf", body_size=size, pages=4))
    assert _finding(report, "body_font_size").severity is expected


def test_page_numbers_are_reported(tmp_path: Path) -> None:
    finding = _finding(_run(build_icassp(tmp_path / "folios.pdf", folios=True)), "page_numbers")
    assert finding.severity is Severity.ERROR
    assert len(finding.evidence) >= 2


def test_a_title_in_mixed_case_is_reported(tmp_path: Path) -> None:
    path = build_icassp(tmp_path / "title.pdf", title="Fixed-Budget Front Ends for Speech")
    assert _finding(_run(path), "title_format").severity is Severity.WARNING


def test_more_than_five_index_terms_is_reported(tmp_path: Path) -> None:
    path = build_icassp(tmp_path / "terms.pdf", keywords="a1, b2, c3, d4, e5, f6")
    finding = _finding(_run(path), "index_terms")
    assert finding.severity is Severity.WARNING
    assert finding.evidence[0].measured == 6


@pytest.mark.parametrize(("words", "expected"), [(190, Severity.PASS), (210, Severity.ERROR)])
def test_the_abstract_is_capped_at_200_words(tmp_path: Path, words: int, expected: Severity) -> None:
    """Anything up to the cap passes, the kit's "approximately 100 to 150" included."""
    abstract = " ".join((_PROSE.split() * 8)[:words])
    finding = _finding(_run(build_icassp(tmp_path / "abstract.pdf", abstract=abstract)),
                       "abstract_present")
    assert finding.severity is expected, finding.message
    assert f"{words} words" in finding.message


# -- not blind: the author block ------------------------------------------------------


@pytest.mark.parametrize(("names", "address", "expected", "phrase"), [
    ("", "", Severity.ERROR, "No author names"),
    ("Author(s) Name(s)", "Author Affiliation(s)", Severity.ERROR, "placeholder"),
    ("Jane Doe, John Roe", "University of Somewhere", Severity.PASS, "author block"),
])
def test_the_author_block_must_be_filled_in(tmp_path: Path, names: str, address: str,
                                            expected: Severity, phrase: str) -> None:
    path = build_icassp(tmp_path / "authors.pdf", names=names, address=address)
    report = _run(path)
    finding = _finding(report, "author_block")
    assert finding.severity is expected
    assert phrase in finding.message
    assert _finding(report, "anonymity_title_block").severity is Severity.SKIPPED


def test_the_model_does_not_judge_anonymity_at_a_single_blind_venue(tmp_path: Path) -> None:
    """Even a profile that lists the model check skips it when anonymity is off."""
    assert "anonymity_semantics" not in load_profile("icassp").get("llm.checks")
    override = tmp_path / "icassp-model.yaml"
    override.write_text("conference: {key: icassp_model, extends: icassp}\n"
                        "llm: {checks: [anonymity_semantics]}\n")
    profile = load_profile(str(override))
    path = build_icassp(tmp_path / "named.pdf")
    ctx = CheckContext(doc=Document(path), profile=profile, track=profile.track("main"),
                       settings=Settings.offline())
    try:
        assert load_builtin_checks().checks["llm_anonymity"].run(ctx) == []
    finally:
        ctx.doc.close()


# -- required statements --------------------------------------------------------------


def test_a_funding_note_without_conflicts_of_interest_is_an_error(tmp_path: Path) -> None:
    path = build_icassp(tmp_path / "no-coi.pdf", layout={4: ([(None, BODY)], [
        ("5. COMPLIANCE WITH ETHICAL STANDARDS", ETHICS),
        ("6. FUNDING ACKNOWLEDGEMENT", "No specific external funding was received for this work."),
    ])})
    finding = _finding(_run(path), "statement.funding_disclosure")
    assert finding.severity is Severity.ERROR
    assert "never mentions conflicts of interest" in finding.message


def test_conflicts_may_be_declared_in_their_own_section(tmp_path: Path) -> None:
    path = build_icassp(tmp_path / "split.pdf", layout={4: ([(None, BODY)], [
        ("5. COMPLIANCE WITH ETHICAL STANDARDS", ETHICS),
        ("6. ACKNOWLEDGMENTS", "This work was supported by grant 123."),
        ("7. CONFLICT OF INTEREST", "The authors declare no competing interests."),
    ])})
    assert _finding(_run(path), "statement.funding_disclosure").severity is Severity.PASS


def test_the_ethics_section_must_carry_its_policy_title(tmp_path: Path) -> None:
    """An "Ethics statement" may sit on page five, but the policy names the section."""
    path = build_icassp(tmp_path / "ethics.pdf", layout={
        4: ([(None, BODY)], [("6. ACKNOWLEDGMENTS", ACKS)]),
        5: ([("7. ETHICS STATEMENT", ETHICS), ("8. REFERENCES", REFS)], []),
    })
    report = _run(path)
    assert _finding(report, "statement.ethical_compliance").severity is Severity.ERROR
    assert _finding(report, "page_limit").severity is Severity.PASS


def _ethics_coverage(tmp_path: Path, reply: dict) -> tuple[Finding, list[str]]:
    """Run the model-backed coverage check with a canned answer for the ethics section."""
    from preflight.llm.client import AsyncLLMClient

    prompts: list[str] = []

    class Canned(AsyncLLMClient):
        async def json(self, system: str, user: str, *, schema_hint: str = "") -> dict:
            prompts.append(user)
            return reply if "Compliance with Ethical Standards" in user.split("=== STATEMENT CHECK")[-1] \
                else {"items": [], "confidence": "high"}

    profile = load_profile("icassp")
    path = build_icassp(tmp_path / "ethics-model.pdf")
    ctx = CheckContext(doc=Document(path), profile=profile, track=profile.track("main"),
                       settings=Settings(enable_llm=True, openai_api_key="test"))
    ctx.shared["llm_client"] = Canned(api_key="test")
    try:
        findings = asyncio.run(load_builtin_checks().checks["llm_statement_coverage"].arun(ctx))
    finally:
        ctx.doc.close()
    finding = next(f for f in findings if f.check_id == "llm_statement_coverage.ethical_compliance")
    return finding, [q for q in prompts if "=== STATEMENT CHECK" in q]


def test_ethics_items_pass_a_paper_without_human_or_animal_subjects(tmp_path: Path) -> None:
    finding, (prompt,) = _ethics_coverage(tmp_path, {
        "applies": False, "applies_reason": "A simulation study with public benchmarks only.",
        "items": [], "confidence": "high"})
    assert finding.severity is Severity.PASS
    assert "Not required here" in finding.message and "simulation study" in finding.message
    # The model decides from the whole paper, not the statement alone.
    assert "=== BEGIN PAPER TEXT ===" in prompt and "listening test" in prompt
    assert ETHICS in prompt


@pytest.mark.parametrize("applies", [True, None])
def test_ethics_items_are_judged_when_the_paper_needs_them(tmp_path: Path, applies) -> None:
    """A listening test needs its approval named; a reply that skips "applies" is judged too."""
    items = load_profile("icassp").get("statements.ethical_compliance.must_address")
    reply = {"items": [{"item": items[0], "status": "named"},
                       {"item": items[1], "status": "unaddressed"}], "confidence": "medium"}
    if applies is not None:
        reply["applies"] = applies
    finding, _ = _ethics_coverage(tmp_path, reply)
    assert finding.severity is Severity.WARNING
    assert "1 of 2 required items are not addressed" in finding.message


# -- PDF container ----------------------------------------------------------------------


def _pdf_check(path: Path, check_id: str):
    profile = load_profile("icassp")
    ctx = CheckContext(doc=Document(path), profile=profile, track=profile.track("main"),
                       settings=Settings.offline())
    try:
        return load_builtin_checks().checks[check_id].run(ctx)
    finally:
        ctx.doc.close()


def test_a_font_embedded_in_full_is_not_subset(tmp_path: Path) -> None:
    path = tmp_path / "full-font.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_font(fontname="F0", fontbuffer=pymupdf.Font("tiro").buffer)
    page.insert_text((72, 72), "A fully embedded font", fontname="F0", fontsize=10)
    doc.save(path)
    doc.close()
    finding = _pdf_check(path, "font_embedding")[0]
    assert finding.severity is Severity.ERROR
    assert "embedded but not subset" in finding.message


def test_type3_fonts_warn_without_failing_embedding() -> None:
    from types import SimpleNamespace

    class FakePage:
        def get_fonts(self, *, full: bool):
            return [(7, "n/a", "Type3", "MatplotlibGlyphs")]

    class FakeRawDocument:
        def __getitem__(self, index: int):
            return FakePage()

    profile = load_profile("icassp")
    parsed = SimpleNamespace(doc=FakeRawDocument(), pages=[SimpleNamespace(number=1)])
    ctx = CheckContext(doc=parsed, profile=profile, track=profile.track("main"),
                       settings=Settings.offline())
    checks = load_builtin_checks().checks
    assert checks["type3_fonts"].run(ctx)[0].severity is Severity.WARNING
    assert checks["font_embedding"].run(ctx)[0].severity is Severity.SKIPPED


def test_a_file_over_five_megabytes_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "large.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Large file", fontsize=10)
    doc.embfile_add("data.bin", os.urandom(5_300_000))  # incompressible
    doc.save(path)
    doc.close()
    finding = _pdf_check(path, "pdf_health")[0]
    assert finding.severity is Severity.ERROR
    assert any("MB" in e.detail for e in finding.evidence)
