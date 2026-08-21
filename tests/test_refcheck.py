"""The reference checker: classification, matching, caching, and routing."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from preflight.refcheck.core import (
    Candidate,
    Kind,
    MatchPolicy,
    Reference,
    Status,
    author_overlap,
    classify,
    extract_url,
    title_key,
    title_similarity,
)
from preflight.refcheck.store import Store

# ---------------------------------------------------------------------------
# Classification: the thing that stops blog posts being reported as fake papers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Anthropic. Introducing Claude Haiku 4.5, 2025. https://www.anthropic.com/news/x", Kind.BLOG),
        ("T. Wolf et al. Transformers. https://github.com/huggingface/transformers, 2020.", Kind.SOFTWARE),
        ("Bespoke Labs. MiniCheck. https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B", Kind.MODEL),
        ("A. Vaswani et al. Attention is all you need. In Advances in NeurIPS, 2017.", Kind.PAPER),
        ("S. Bird. NLP with Python. O Reilly Media, 2009. ISBN 978-0596516499.", Kind.BOOK),
        ("R. Fielding. HTTP/1.1. RFC 2616, https://www.rfc-editor.org/rfc/rfc2616", Kind.STANDARD),
        ("J. Smith. A study of things. PhD thesis, Some University, 2020.", Kind.THESIS),
    ],
)
def test_classification(raw: str, expected: Kind) -> None:
    assert classify(raw) is expected


def test_identifiers_outrank_urls() -> None:
    """A published paper that also links a repo is still a paper."""
    raw = "A. Author. A real paper. In ACL, 2023. Code at https://github.com/a/b"
    assert classify(raw, doi="10.1000/xyz") is Kind.PAPER


def test_only_papers_are_expected_to_be_indexed() -> None:
    assert Kind.PAPER.expects_index and Kind.PREPRINT.expects_index
    assert not Kind.BLOG.expects_index
    assert not Kind.SOFTWARE.expects_index
    assert not Kind.MODEL.expects_index


def test_unindexed_is_not_suspicious() -> None:
    """The whole point: a blog post absent from CrossRef is not an accusation."""
    assert not Status.UNINDEXED.is_suspicious
    assert not Status.RESOLVED.is_suspicious
    assert Status.NOT_FOUND.is_suspicious
    assert Status.AUTHOR_MISMATCH.is_suspicious
    assert Status.UNRESOLVED.is_suspicious


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def test_scavenge_recovers_identifiers() -> None:
    ref = Reference(
        raw="J. Doe. A paper. arXiv:1706.03762, 2017. doi:10.5555/3295222 https://x.org/a",
        index=0,
    )
    ref.scavenge()
    assert ref.doi == "10.5555/3295222"
    assert ref.arxiv_id == "1706.03762"
    assert ref.year == 2017
    assert ref.url == "https://x.org/a"


def test_extract_url_trims_trailing_punctuation() -> None:
    assert extract_url("see https://example.com/a.") == "https://example.com/a"
    assert extract_url("no url here") is None


def test_title_key_ignores_stopwords_and_case() -> None:
    assert title_key("The Attention of a Model") == title_key("Attention Model")


def test_usability_gate() -> None:
    assert not Reference(raw="ibid.", index=0).is_usable
    assert Reference(raw="x", index=0, title="A Sufficiently Long Title").is_usable
    assert Reference(raw="x", index=0, doi="10.1/2").is_usable


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_similarity_ranges() -> None:
    assert title_similarity("Attention Is All You Need", "attention is all you need") == 1.0
    assert title_similarity("BERT: Pre-training of Deep Bidirectional Transformers",
                            "BERT: Pre-training of Deep Bidirectional Transformers for "
                            "Language Understanding") > 0.8
    assert title_similarity("Attention Is All You Need", "A Survey of Reinforcement Learning") < 0.4


def test_author_overlap_reports_unknown_rather_than_zero() -> None:
    """Absence of author data must not read as disagreement."""
    assert author_overlap([], ["Smith"]) == -1.0
    assert author_overlap(["Vaswani, Ashish"], ["Ashish Vaswani", "N. Shazeer"]) == 1.0
    assert author_overlap(["Smith"], ["Jones"]) == 0.0


def _ref(**kw) -> Reference:
    return Reference(raw="raw", index=0, **kw)


def test_policy_accepts_a_clean_match() -> None:
    policy = MatchPolicy()
    ref = _ref(title="Attention Is All You Need", authors=["Vaswani"], year=2017)
    cand = Candidate(source="openalex", title="Attention Is All You Need",
                     authors=["Ashish Vaswani"], year=2017)
    assert policy.judge(ref, cand)[0] == "accept"


def test_policy_reviews_a_same_words_different_paper() -> None:
    """"Is Attention All You Need?" is a real, different paper."""
    policy = MatchPolicy()
    ref = _ref(title="Attention Is All You Need", authors=["Vaswani", "Shazeer"], year=2017)
    cand = Candidate(source="crossref", title="Is Attention All You Need?",
                     authors=["Someone Else", "Another Person"], year=2021)
    assert policy.judge(ref, cand)[0] == "review"


def test_policy_rejects_an_unrelated_title() -> None:
    policy = MatchPolicy()
    ref = _ref(title="Attention Is All You Need")
    cand = Candidate(source="dblp", title="Cooking With Cast Iron")
    assert policy.judge(ref, cand)[0] == "reject"


def test_exact_identifier_always_accepts() -> None:
    policy = MatchPolicy()
    ref = _ref(title="Whatever the extractor produced", doi="10.1/2")
    cand = Candidate(source="crossref", title="The Real Title", exact_id=True)
    outcome, score, _ = policy.judge(ref, cand)
    assert outcome == "accept" and score == 1.0


def test_policy_reviews_a_year_that_is_far_off() -> None:
    policy = MatchPolicy()
    ref = _ref(title="A Distinctive Paper Title Here", authors=["Smith"], year=2015)
    cand = Candidate(source="openalex", title="A Distinctive Paper Title Here",
                     authors=["Smith"], year=2023)
    assert policy.judge(ref, cand)[0] == "review"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def test_store_round_trips(tmp_path: Path) -> None:
    with Store(tmp_path / "refs.db") as store:
        assert store.enabled
        store.put("paper:a title", {"status": "verified", "source": "openalex"}, positive=True)
        assert store.get("paper:a title") == {"status": "verified", "source": "openalex"}
        assert store.get("paper:missing") is None


def test_store_expires_negatives_sooner(tmp_path: Path) -> None:
    """"Not found" ages out quickly; indexes catch up."""
    with Store(tmp_path / "refs.db", positive_ttl=1000, negative_ttl=-1) as store:
        store.put("k", {"status": "not_found"}, positive=False)
        assert store.get("k") is None
    with Store(tmp_path / "refs2.db", positive_ttl=1000, negative_ttl=1000) as store:
        store.put("k", {"status": "verified"}, positive=True)
        assert store.get("k") is not None


def test_store_degrades_without_a_path() -> None:
    store = Store(None)
    assert not store.enabled
    store.put("k", {"a": 1}, positive=True)
    assert store.get("k") is None


# ---------------------------------------------------------------------------
# Engine routing, with no network
# ---------------------------------------------------------------------------

def test_engine_routes_and_reports(monkeypatch) -> None:
    from preflight.refcheck import engine

    async def fake_web(session, reference):
        from preflight.refcheck.websources import Resolution
        if "github.com/real" in (reference.url or ""):
            return Resolution(True, "repository exists", url=reference.url, source="github")
        return Resolution(False, "repository does not exist (404)", url=reference.url,
                          source="github")

    monkeypatch.setattr(engine, "resolve_web", fake_web)
    monkeypatch.setattr(engine, "_select_sources", lambda config: ())

    refs = [
        Reference(raw="a", index=0, url="https://github.com/real/repo", kind=Kind.SOFTWARE),
        Reference(raw="b", index=1, url="https://github.com/fake/repo", kind=Kind.SOFTWARE),
        Reference(raw="c", index=2, title="An Unindexed Blog Post Title", kind=Kind.BLOG),
        Reference(raw="d", index=3, title="A Paper Nobody Has", kind=Kind.PAPER),
    ]
    config = engine.RefCheckConfig(use_web_search=False, cache_path=None)
    report = asyncio.run(engine.verify(refs, config))

    by_index = {v.reference.index: v for v in report.verdicts}
    assert by_index[0].status is Status.RESOLVED
    assert by_index[1].status is Status.UNRESOLVED
    # A blog post with no URL is not indexed anywhere, and that is not suspicious.
    assert by_index[2].status is Status.UNINDEXED
    assert not by_index[2].is_suspicious
    # A paper that no database has is the one case worth reporting.
    assert by_index[3].status is Status.NOT_FOUND
    assert by_index[3].is_suspicious
    assert len(report.suspicious) == 2


def test_engine_skips_unparsable_entries() -> None:
    from preflight.refcheck import engine

    config = engine.RefCheckConfig(use_web_search=False, cache_path=None)
    report = asyncio.run(engine.verify([Reference(raw="ibid.", index=0)], config))
    assert report.verdicts[0].status is Status.UNPARSED
    assert not report.verdicts[0].is_suspicious


# ---------------------------------------------------------------------------
# Bibliography bounds, entry shape, and URL repair
# ---------------------------------------------------------------------------

def test_bibliography_stops_at_the_next_section(clean_paper: Path) -> None:
    """The region ends at the next heading, whatever it is called.

    Papers letter their appendix sections ("A Main Experiments"), so looking only
    for a heading called "Appendix" runs the bibliography to the end of the file
    and turns every table row into a citation.
    """
    from preflight.context import CheckContext, Settings
    from preflight.document import Document
    from preflight.profile import load_profile
    from preflight.refcheck.parse import bibliography_lines

    profile = load_profile("arr")
    ctx = CheckContext(doc=Document(clean_paper), profile=profile,
                       track=profile.track("long"), settings=Settings.offline())
    try:
        pages = {line.page for line in bibliography_lines(ctx)}
        # References are on page 9 and the appendix starts on page 10.
        assert pages and max(pages) <= 10
        assert ctx.doc.page_count == 12       # ...so the tail is genuinely excluded
    finally:
        ctx.doc.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Anthropic. 2026. Claude-sonnet-4.6. https://anthropic.com/news/x.", True),
        ("xAI. 2025. Grok-4.3. 2025-08-20-grok-4-model-card.pdf.", True),
        ("A Author, B Author, and C Author. 2024. A paper title. In Proceedings.", True),
        ("Liyan Tang and Greg Durrett. 2024a. Minicheck: fact-checking on documents.", True),
        ("ensure a fair comparison.", False),                    # trailing prose
        ("Out-of-Bounds Rate 0.10 0.07 0.06 0.05 0.04 0.03 0.02", False),   # table row
        ("Observation", False),                                  # a heading
    ],
)
def test_entry_shape_filter(text: str, expected: bool) -> None:
    from preflight.refcheck.parse import looks_like_entry

    assert looks_like_entry(text) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # PDF extraction breaks URLs across lines in several ways.
        ("https: //www.anthropic.com/news/x", "https://www.anthropic.com/news/x"),
        ("ht tps://example.com/a", "https://example.com/a"),
        ("https://develo pers.openai.com/x", "https://developers.openai.com/x"),
        ("www.example.com/a", "https://www.example.com/a"),
        ("https://ok.example.com/x.", "https://ok.example.com/x"),
        # ...and sometimes destroys them beyond use, which must not crash a run.
        ("/deepmind.google/models/model-cards/gemini-3-1-pro/", None),
        ("https://model", None),          # host cut short: checking it proves nothing
        ("not a url", None),
        ("", None),
    ],
)
def test_url_repair(raw: str, expected: str | None) -> None:
    from preflight.refcheck.core import normalise_url

    assert normalise_url(raw) == expected


def test_url_join_does_not_swallow_the_next_sentence() -> None:
    """A line can break exactly where a URL ends; the following prose is not path."""
    from preflight.refcheck.parse import _continues_url

    assert _continues_url("https://example.com/news/claude-haiku-4-5.", "Anthropic") is False
    assert _continues_url("https://example.com/bespokelabs/", "Bespoke-MiniCheck") is True
    assert _continues_url("https://develo", "pers.openai.com") is True
    assert _continues_url("Smith,", "2024.") is False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_duplicate_references_are_verified_once() -> None:
    from preflight.refcheck import engine

    calls: list[int] = []

    async def fake_web(session, reference):
        from preflight.refcheck.websources import Resolution
        calls.append(reference.index)
        return Resolution(True, "repository exists", url=reference.url, source="github")

    import pytest as _pytest
    monkey = _pytest.MonkeyPatch()
    monkey.setattr(engine, "resolve_web", fake_web)
    monkey.setattr(engine, "_select_sources", lambda config: ())
    try:
        refs = [
            Reference(raw="a", index=0, url="https://github.com/a/b", kind=Kind.SOFTWARE,
                      title="The Same Tool"),
            Reference(raw="b", index=1, url="https://github.com/a/b", kind=Kind.SOFTWARE,
                      title="The Same Tool"),
            Reference(raw="c", index=2, url="https://github.com/c/d", kind=Kind.SOFTWARE,
                      title="A Different Tool"),
        ]
        report = asyncio.run(engine.verify(refs, engine.RefCheckConfig(use_web_search=False,
                                                                      cache_path=None)))
    finally:
        monkey.undo()

    # Two unique works, so two lookups — but a verdict for all three entries.
    assert len(calls) == 2
    assert report.unique == 2
    assert len(report.verdicts) == 3
    assert [v.reference.index for v in report.verdicts] == [0, 1, 2]
    assert all(v.status is Status.RESOLVED for v in report.verdicts)
    assert len(report.duplicates) == 1
    assert report.duplicates[0][1] == [0, 1]


def test_dedup_key_prefers_identifiers() -> None:
    from preflight.refcheck.engine import _dedup_key

    # A preprint and its published version share a title, so they collapse.
    a = Reference(raw="a", index=0, title="Attention Is All You Need")
    b = Reference(raw="b", index=1, title="attention is all you need!")
    assert _dedup_key(a) == _dedup_key(b)
    # A DOI is definitive and outranks the title.
    c = Reference(raw="c", index=2, title="Same Title", doi="10.1/x")
    d = Reference(raw="d", index=3, title="Same Title", doi="10.2/y")
    assert _dedup_key(c) != _dedup_key(d)
