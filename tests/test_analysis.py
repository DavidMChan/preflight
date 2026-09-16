"""Sentence, caption and context extraction."""

from __future__ import annotations

from pathlib import Path

from preflight.analysis import (
    Sentence,
    body_text,
    clean_text,
    full_text,
    numeric_tokens,
    paper_context,
    sentences,
)
from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.profile import load_profile


def _ctx(path: Path) -> CheckContext:
    profile = load_profile("arr")
    return CheckContext(doc=Document(path), profile=profile, track=profile.track("long"),
                        settings=Settings.offline())


def test_clean_text_drops_line_numbers_and_rejoins_hyphens(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        assert clean_text(ctx, "001\nHello there\n002\nworld") == "Hello there\nworld"
        assert clean_text(ctx, "hyphen-\nated word") == "hyphenated word"
    finally:
        ctx.doc.close()


def test_body_text_stops_at_the_bibliography(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        body = body_text(ctx)
        assert "Language models are increasingly used" in body
        # The synthetic paper's bibliography entry must not leak into body prose.
        assert "Proceedings" not in body
        assert len(body) < len(full_text(ctx))
    finally:
        ctx.doc.close()


def test_sentences_are_segmented_and_cached(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        found = sentences(ctx)
        assert found
        assert sentences(ctx) is found            # cached
        assert all(s.word_count >= 4 for s in found)
        assert all(s.index == i for i, s in enumerate(found))
        assert any("agents" in s.text for s in found)
    finally:
        ctx.doc.close()


def test_sentence_helpers() -> None:
    s = Sentence(text="However, the model fails; we do not know why.", index=0)
    assert s.opener == "however"
    assert s.has_internal_break
    assert s.word_count == 9
    assert not Sentence(text="A plain short sentence here.", index=1).has_internal_break


def test_abbreviations_do_not_split_sentences(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        found = sentences(ctx, "We follow prior work, e.g. Smith et al. and others, in this setup. "
                               "The second sentence is entirely separate from the first one.")
        assert len(found) == 2
        assert "e.g." in found[0].text
    finally:
        ctx.doc.close()


def test_paper_context_is_stable_and_untruncated(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        first = paper_context(ctx)
        assert paper_context(ctx) == first
        assert "BEGIN PAPER TEXT" in first and "END PAPER TEXT" in first
        # The default budget is generous, so a normal paper is never trimmed.
        assert "llm_context_truncated" not in ctx.shared
        assert "omitted for length" not in first
    finally:
        ctx.doc.close()


def test_paper_context_records_truncation(clean_paper: Path) -> None:
    """Trimming must be recorded, so the run can report it rather than hide it."""
    ctx = _ctx(clean_paper)
    try:
        ctx.profile.raw.setdefault("llm", {})["max_context_chars"] = 400
        text = paper_context(ctx)
        assert "omitted for length" in text
        record = ctx.shared["llm_context_truncated"]
        assert record["omitted_chars"] > 0
        assert record["kept_chars"] == 400
        assert record["total_chars"] > record["kept_chars"]
    finally:
        ctx.doc.close()


def test_numeric_tokens_only_matches_decimals() -> None:
    assert numeric_tokens("we report 92.15 and 3.4 but not 2024 or v2") == ["92.15", "3.4"]


def test_title_is_the_largest_type_not_the_first_line(small_caps_paper: Path) -> None:
    from preflight.analysis import title_text

    ctx = _ctx(small_caps_paper)
    # The running head and the line numbers come first on the page; the title
    # is the largest type, and its small capitals are read at the initial's size.
    assert title_text(ctx) == "DRIVERLESS STEERING"


def test_captions_do_not_absorb_margin_line_numbers(tmp_path: Path) -> None:
    import pymupdf

    from preflight.analysis import captions

    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((108, 200), "Figure 3: Layer sweep over the", fontname="tiro", fontsize=10)
    page.insert_text((73, 206), "1898", fontname="tiro", fontsize=8)    # margin ruler
    page.insert_text((108, 212), "development split.", fontname="tiro", fontsize=10)
    page.insert_text((73, 218), "1899", fontname="tiro", fontsize=8)
    path = tmp_path / "cap.pdf"
    doc.save(path)
    doc.close()
    caps = captions(_ctx(path))
    assert len(caps) == 1
    assert caps[0].text == "Layer sweep over the development split."
    assert "1898" not in caps[0].text and "1899" not in caps[0].text
