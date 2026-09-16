"""Synthetic PDFs, so the tests do not depend on any file outside the repo."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

A4 = (595.0, 842.0)
LEFT_COL = (71.0, 290.0)
RIGHT_COL = (306.0, 525.0)
TOP = 90.0
BOTTOM = 770.0

_LOREM = (
    "Language models are increasingly used as agents that decide which information a person "
    "sees. This paper studies how that mediation changes what content creators are rewarded "
    "for producing, following earlier work (Doe and Roe, 2021). We describe the setting, the "
    "method, and the evaluation we ran. "
)


def _column(page: pymupdf.Page, rect: tuple[float, float], text: str, size: float = 11.0,
            top: float = TOP, bold: bool = False) -> None:
    font = "tibo" if bold else "tiro"          # Times bold / Times roman
    page.insert_textbox(
        pymupdf.Rect(rect[0], top, rect[1], BOTTOM), text,
        fontname=font, fontsize=size, align=pymupdf.TEXT_ALIGN_LEFT,
    )


def build_paper(
    path: Path,
    *,
    pages: int = 9,
    limitations_page: int | None = 9,
    references_page: int | None = 9,
    appendix_page: int | None = None,
    page_size: tuple[float, float] = A4,
    body_size: float = 11.0,
    author_line: str = "Anonymous ACL submission",
    extra_first_page: str = "",
) -> Path:
    """A minimal but geometrically honest two-column ACL-style paper."""
    doc = pymupdf.open()
    for number in range(1, pages + 1):
        page = doc.new_page(width=page_size[0], height=page_size[1])
        top = TOP
        if number == 1:
            page.insert_textbox(
                pymupdf.Rect(71, 60, 525, 110),
                "A Study Of Something Specific\n" + author_line + "\n" + extra_first_page,
                fontname="tibo", fontsize=13, align=pymupdf.TEXT_ALIGN_CENTER,
            )
            top = 150.0
            _column(page, LEFT_COL, "1 Introduction", size=12.0, top=top, bold=True)
            top += 20

        body = _LOREM * 5
        if number == limitations_page:
            _column(page, LEFT_COL, "Limitations", size=12.0, top=top, bold=True)
            _column(page, LEFT_COL,
                    "Our study assumes a fixed candidate list and only English data. "
                    "We did not measure how the approach behaves under distribution shift, "
                    "and results come from a single run per configuration. ",
                    top=top + 18)
            top = TOP
        if number == references_page:
            _column(page, RIGHT_COL, "References", size=12.0, top=TOP, bold=True)
            _column(page, RIGHT_COL,
                    "Jane Doe and John Roe. 2021. A paper about things. In Proceedings. ",
                    top=TOP + 18)
        if number == appendix_page:
            _column(page, LEFT_COL, "A Appendix", size=12.0, top=TOP, bold=True)
            _column(page, LEFT_COL, body, size=body_size, top=TOP + 18)
            _column(page, RIGHT_COL, body, size=body_size, top=TOP)
            continue
        if number not in (limitations_page, references_page):
            _column(page, LEFT_COL, body, size=body_size, top=top)
            _column(page, RIGHT_COL, body, size=body_size, top=TOP)

    doc.set_metadata({"producer": "pdfTeX-1.40.26", "creator": "LaTeX with hyperref"})
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def clean_paper(tmp_path: Path) -> Path:
    """8 content pages, Limitations then References on page 9, appendix after."""
    return build_paper(tmp_path / "clean.pdf", pages=12, limitations_page=9,
                       references_page=9, appendix_page=10)


@pytest.fixture
def overlong_paper(tmp_path: Path) -> Path:
    """Content runs to page 11 before any unlimited section begins."""
    return build_paper(tmp_path / "overlong.pdf", pages=13, limitations_page=12,
                       references_page=12, appendix_page=13)


@pytest.fixture
def no_limitations_paper(tmp_path: Path) -> Path:
    return build_paper(tmp_path / "nolim.pdf", pages=9, limitations_page=None, references_page=9)


@pytest.fixture
def letter_paper(tmp_path: Path) -> Path:
    return build_paper(tmp_path / "letter.pdf", pages=9, page_size=(612.0, 792.0))


@pytest.fixture
def real_paper() -> Path:
    """The developer's local sample, when it happens to be there."""
    path = Path("/Users/davidchan/Downloads/_EMNLP2026__Clickbait.pdf")
    if not path.is_file():
        pytest.skip("local sample PDF not available")
    return path


# ---------------------------------------------------------------------------
# Small-caps headings and single-column papers (the ICLR / NeurIPS template)
# ---------------------------------------------------------------------------


def _small_caps(page: pymupdf.Page, x: float, y: float, word: str, big: float, small: float) -> float:
    """Set ``word`` the way pdfTeX emits \\textsc: a full-size initial, then
    the rest as smaller capitals. Returns the x position after the word."""
    page.insert_text((x, y), word[0], fontname="tiro", fontsize=big)
    x += pymupdf.get_text_length(word[0], fontname="tiro", fontsize=big)
    page.insert_text((x, y), word[1:].upper(), fontname="tiro", fontsize=small)
    return x + pymupdf.get_text_length(word[1:].upper(), fontname="tiro", fontsize=small)


def build_small_caps_paper(path: Path) -> Path:
    """Two US-Letter pages, single column at x=108, 10pt body, headings in
    small caps as the ICLR style file sets them. A stack of centred display
    equations shares a left edge, which must not become a second column."""
    body = (
        "Activation steering constructs a direction from examples of a behavior. "
        "We find that sequences sampled at random can replace those examples. "
    ) * 6
    doc = pymupdf.open()
    for number in (1, 2):
        page = doc.new_page(width=612, height=792)
        for i in range(40):
            page.insert_text((73, 100 + i * 12), f"{number * 100 + i:03d}", fontname="tiro", fontsize=8)
        page.insert_text((108, 60), "Under review as a conference paper at ICLR 2027",
                         fontname="tiro", fontsize=9)
        y = 100.0
        if number == 1:
            x = 108.0
            for word in ("Driverless", "Steering"):
                x = _small_caps(page, x, y, word, 17.2, 13.8) + 5
            y += 40
            page.insert_text((108, y), "Anonymous authors", fontname="tibo", fontsize=10)
            y += 30
            _small_caps(page, 280, y, "Abstract", 12.0, 9.6)
            y += 16
            page.insert_textbox(pymupdf.Rect(140, y, 470, y + 60),
                                "We study whether random sequences can stand in for examples. "
                                "The abstract sets out the claim in two sentences.",
                                fontname="tiro", fontsize=10)
            y += 70
            page.insert_text((108, y), "1", fontname="tiro", fontsize=12)
            _small_caps(page, 127, y, "Introduction", 12.0, 9.6)
            y += 16
            page.insert_textbox(pymupdf.Rect(108, y, 504, y + 200), body, fontname="tiro", fontsize=10)
            y += 210
            # 2.3 EVALUATION: 10pt small caps, no larger than the body, not bold.
            page.insert_text((108, y), "1.1", fontname="tiro", fontsize=10)
            x = 130.0
            for word in ("Evaluation", "Of", "Steering"):
                x = _small_caps(page, x, y, word, 10.0, 8.0) + 4
            y += 16
            # A three-level number pushes its label past the usual indent.
            page.insert_text((108, y), "1.1.1", fontname="tiro", fontsize=10)
            _small_caps(page, 142.3, y, "Setup", 10.0, 8.0)
            y += 16
            page.insert_textbox(pymupdf.Rect(108, y, 504, y + 200), body, fontname="tiro", fontsize=10)
            y += 210
            for i in range(4):
                page.insert_text((278, y + i * 14), f"v = x + {i}", fontname="tiro", fontsize=10)
        else:
            page.insert_textbox(pymupdf.Rect(108, y, 504, y + 300), body, fontname="tiro", fontsize=10)
            y += 320
            _small_caps(page, 108, y, "References", 12.0, 9.6)
            y += 16
            page.insert_text((108, y), "Jane Doe. A paper about things. CoRR, 2021.",
                             fontname="tiro", fontsize=10)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def small_caps_paper(tmp_path: Path) -> Path:
    return build_small_caps_paper(tmp_path / "small_caps.pdf")
