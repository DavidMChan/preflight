"""Layout analysis: columns, headings, fonts, and reading order."""

from __future__ import annotations

from pathlib import Path

from preflight.document import Document


def test_parses_page_geometry(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        assert doc.page_count == 12
        assert round(doc.pages[0].width) == 595
        assert round(doc.pages[0].height) == 842


def test_detects_two_column_bands(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        assert len(doc.column_bands) == 2
        left, right = doc.column_bands
        assert 65 < left[0] < 80
        assert 300 < right[0] < 315
        assert right[0] > left[1]           # there is a real gutter


def test_body_font_size_is_the_mode(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        assert doc.body_font_size == 11.0


def test_finds_structural_headings(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        limitations = doc.find_heading(["Limitations"])
        references = doc.find_heading(["References", "Bibliography"])
        appendix = doc.find_heading(["Appendix", "Appendices"])
        assert limitations is not None and limitations.page == 9
        assert references is not None and references.page == 9
        assert appendix is not None and appendix.page == 10
        # Reading order puts Limitations before References.
        assert limitations.order < references.order < appendix.order


def test_heading_lookup_is_case_and_punctuation_insensitive(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        assert doc.find_heading(["limitations"]) is not None
        assert doc.find_heading(["LIMITATIONS!"]) is not None
        assert doc.find_heading(["Nonexistent Section"]) is None


def test_text_between_headings_is_scoped(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        limitations = doc.find_heading(["Limitations"])
        references = doc.find_heading(["References"])
        assert limitations is not None and references is not None
        body = doc.text_between(limitations, references)
        assert "fixed candidate list" in body
        assert "Proceedings" not in body     # the bibliography is not swept in


def test_column_fit_is_high_for_a_two_column_page(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        assert doc.column_fit(doc.pages[1]) > 0.9


def test_inside_graphic_is_false_without_artwork(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        page = doc.pages[1]
        span = page.spans[0]
        assert not page.inside_graphic(span.bbox)


def test_metadata_and_no_invisible_text(clean_paper: Path) -> None:
    with Document(clean_paper) as doc:
        assert "pdfTeX" in (doc.metadata.get("producer") or "")
        assert not [s for s in doc.spans if s.invisible]


def test_real_paper_structure(real_paper: Path) -> None:
    """The developer sample, when present: a 23-page anonymous ACL submission."""
    with Document(real_paper) as doc:
        assert doc.page_count == 23
        assert doc.body_font_size == 11.0
        assert len(doc.column_bands) == 2
        assert doc.find_heading(["Limitations"]) is not None
        assert doc.find_heading(["References"]) is not None
