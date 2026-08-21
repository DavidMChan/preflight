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
