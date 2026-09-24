"""Reference details: identifiers, pages, years, venues, authors, versions.

The cases are the ones a reviewer caught on a real submission that the checker
passed: a DOI belonging to a different paper, wrong pages, invented given names,
a misspelled surname, swapped authors, and a preprint cited in place of its
ACL version.
"""

from __future__ import annotations

import asyncio
import gzip
import time
from pathlib import Path

import httpx
import pytest

from preflight.refcheck.anthology import Anthology, anthology_id, build_index, latex_to_text
from preflight.refcheck.arxiv import parse_feed
from preflight.refcheck.batch import Batcher
from preflight.refcheck.compare import (
    cites_preprint,
    compare,
    compare_authors,
    given_compatible,
    split_name,
    venue_tags,
    venues_compatible,
)
from preflight.refcheck.core import (
    Candidate,
    Discrepancy,
    Kind,
    Reference,
    Status,
    find_pages,
    normalise_doi,
    parse_page_range,
)
from preflight.refcheck.dblp import build_query, parse_results

ROLELLM_AUTHORS = [
    "Noah Wang", "Z.Y. Peng", "Haoran Que", "Jiaheng Liu", "Wangchunshu Zhou", "Yuhan Wu",
    "Hongcheng Guo", "Ruitong Gan", "Zehao Ni", "Jian Yang", "Man Zhang", "Zhaoxiang Zhang",
]
CONVLAB_AUTHORS = [
    "Qi Zhu", "Zheng Zhang", "Yan Fang", "Xiang Li", "Ryuichi Takanobu", "Jinchao Li",
    "Baolin Peng", "Jianfeng Gao", "Xiaoyan Zhu", "Minlie Huang",
]


def _acl(**kw) -> Candidate:
    return Candidate(source="acl_anthology", preprint=False, **kw)


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("19--35", ("19", "35")),
        ("14743–14777", ("14743", "14777")),
        ("1877-901", ("1877", "1901")),          # abbreviated last page
        ("e1002", ("e1002", None)),
        ("not pages", None),
    ],
)
def test_page_ranges(text: str, expected: tuple | None) -> None:
    assert parse_page_range(text) == expected


def test_pages_are_found_in_a_citation() -> None:
    assert find_pages("In Proc. of X, pages 28–40, Malta, 2024.") == "28–40"
    assert find_pages("Nature, 521(7553):436–444, 2015.") == "436–444"
    assert find_pages("A paper with no pages. 2024.") is None


def test_doi_normalisation() -> None:
    assert normalise_doi("https://doi.org/10.18653/V1/2024.X.") == "10.18653/v1/2024.x"
    assert normalise_doi("doi: 10.1000/abc") == "10.1000/abc"
    assert normalise_doi("not a doi") is None


def test_doi_hyphen_at_a_line_break_is_kept() -> None:
    """A DOI broken after "findings-" must not lose its hyphen to de-hyphenation."""
    from preflight.refcheck.parse import _continues_url

    assert _continues_url("10.18653/v1/2024.findings-", "emnlp.218.")
    assert not _continues_url("10.18653/v1/2024.findings-emnlp.218.", "https://aclanthology.org/x")


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------

def test_name_forms() -> None:
    assert split_name("Wang, Noah").surname == "wang"
    assert split_name("Ivan Sekulić").surname == "sekulic"
    assert split_name("Wei Wang 0001").surname == "wang"
    assert split_name("Smith, Jr., John").given == ("john",)
    assert split_name("ZY Peng").given == ("z", "y")
    assert split_name("et al.") is None


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Zhongyuan Peng", "Z.Y. Peng", True),       # initials
        ("Yu-Min Tseng", "Yumin Tseng", True),       # hyphenation
        ("Zekun M. Wang", "Zekun Moore Wang", True),  # middle name
        ("Yufei Wu", "Yuhan Wu", False),             # a different first name
        ("Tiezheng Guo", "Hongcheng Guo", False),
    ],
)
def test_given_names(a: str, b: str, expected: bool) -> None:
    assert given_compatible(split_name(a).given, split_name(b).given) is expected


def test_invented_given_names_are_reported() -> None:
    cited = ["Zekun Moore Wang", "Zhongyuan Peng", "Haoran Que", "Jiaheng Liu", "Wangchunshu Zhou",
             "Yufei Wu", "Tiezheng Guo", "Bifan Gan", "Ziyuan Ni", "Man Yang", "et al."]
    arxiv = ["Zekun Moore Wang", "Zhongyuan Peng", *ROLELLM_AUTHORS[2:]]
    problems = compare_authors(cited, [ROLELLM_AUTHORS, arxiv], truncated=True)
    text = "; ".join(problems)
    for wrong, right in [("Yufei Wu", "Yuhan Wu"), ("Tiezheng Guo", "Hongcheng Guo"),
                         ("Bifan Gan", "Ruitong Gan"), ("Ziyuan Ni", "Zehao Ni"),
                         ("Man Yang", "Jian Yang")]:
        assert f"{wrong} should be {right}" in text
    # "Zekun Moore Wang" is how the arXiv record spells "Noah Wang": not a problem.
    assert "Zekun" not in text
    assert len(problems) == 5


def test_a_wrong_surname_and_swapped_authors_are_reported() -> None:
    cited = ["Qi Zhu", "Zheng Zhang", "Yan Fang", "Xiang Li", "Ryuichi Takanobu", "Baolin Peng",
             "Jinchao Li", "Jianfeng Gao", "Xiaoyan Zhao", "Minlie Huang"]
    problems = compare_authors(cited, [CONVLAB_AUTHORS], truncated=False)
    text = "; ".join(problems)
    assert "Xiaoyan Zhao is not an author (the record has Xiaoyan Zhu)" in text
    assert "the record lists Jinchao Li before Baolin Peng" in text


def test_correct_author_lists_pass() -> None:
    assert compare_authors(CONVLAB_AUTHORS, [CONVLAB_AUTHORS], truncated=False) == []
    assert compare_authors(["Qi Zhu", "Zheng Zhang", "et al."], [CONVLAB_AUTHORS], truncated=True) == []
    assert compare_authors(["Zhu, Qi", "Zhang, Zheng", "Fang, Yan", "Li, Xiang", "Takanobu, Ryuichi",
                            "Li, Jinchao", "Peng, Baolin", "Gao, Jianfeng", "Zhu, Xiaoyan",
                            "Huang, Minlie"], [CONVLAB_AUTHORS], truncated=False) == []
    # A corporate author is not compared against a list of people.
    assert compare_authors(["OpenAI"], [["Josh Achiam", "Steven Adler"]], truncated=False) == []
    assert compare_authors(["Gemini Team"], [["Rohan Anil"]], truncated=False) == []
    # Family-name-first order is the same person.
    assert compare_authors(["Zhu Qi", "Zhang Zheng"], [["Qi Zhu", "Zheng Zhang"]], truncated=False) == []


def test_a_silently_shortened_author_list_is_reported() -> None:
    problems = compare_authors(CONVLAB_AUTHORS[:3], [CONVLAB_AUTHORS], truncated=False)
    assert problems == ["the citation lists 3 of the record's 10 authors without 'et al.'"]


# ---------------------------------------------------------------------------
# Venues and versions
# ---------------------------------------------------------------------------

def test_venue_tags() -> None:
    assert venue_tags("Findings of the Association for Computational Linguistics: EMNLP 2024") == {
        "findings", "emnlp"}
    assert venue_tags("Proceedings of the 61st Annual Meeting of the Association for Computational "
                      "Linguistics (Volume 4: Student Research Workshop)") == {"acl", "workshop"}
    assert venue_tags("ACL (student)") == {"acl", "workshop"}
    assert venue_tags("Advances in Neural Information Processing Systems") == {"neurips"}
    assert venues_compatible({"neurips"}, venue_tags("NIPS"))
    assert not venues_compatible({"emnlp"}, {"acl"})


def test_cites_preprint() -> None:
    assert cites_preprint(Reference(raw="G. Simmons. Moral mimicry: Large language models "
                                        "produce moral rationalizations, 2022.", index=0))
    assert cites_preprint(Reference(raw="x", index=0, venue="arXiv preprint arXiv:2310.00746"))
    assert not cites_preprint(Reference(raw="x", index=0, venue="Findings of ACL 2024"))
    assert not cites_preprint(Reference(raw="A. B. A title. In Proceedings of ACL, 2023.", index=0))


def _report(ref: Reference, records: list[Candidate]) -> dict[str, str]:
    return {d.field: d.detail for d in compare(ref, records)}


def test_wrong_pages_are_reported() -> None:
    ref = Reference(raw="x", index=0, title="Reliable LLM-based User Simulator", pages="28–40",
                    year=2024, venue="Proceedings of the 1st Workshop on SCI-CHAT 2024")
    record = _acl(title="Reliable LLM-based User Simulator", pages="19-35", year=2024,
                  venue="Proceedings of the 1st Workshop on Simulating Conversational Intelligence")
    assert _report(ref, [record]) == {"pages": "pages 28–40; the record has 19-35"}


def test_matching_details_report_nothing() -> None:
    ref = Reference(raw="x", index=0, title="T", pages="142--149", year=2020,
                    venue="Proceedings of the 58th Annual Meeting of the ACL: System Demonstrations",
                    authors=CONVLAB_AUTHORS)
    record = _acl(title="T", pages="142-149", year=2020, authors=CONVLAB_AUTHORS,
                  venue="Proceedings of the 58th Annual Meeting of the Association for "
                        "Computational Linguistics: System Demonstrations")
    assert compare(ref, [record]) == []


def test_a_preprint_with_a_published_version_is_reported() -> None:
    ref = Reference(raw="Gabriel Simmons. Moral mimicry: Large language models produce moral "
                        "rationalizations tailored to political identity, 2022.", index=0,
                    title="Moral mimicry", year=2022, authors=["Gabriel Simmons"])
    arxiv = Candidate(source="arxiv", title="Moral Mimicry", year=2022, preprint=True,
                      authors=["Gabriel Simmons"])
    acl = _acl(title="Moral Mimicry", year=2023, pages="282-297", doi="10.18653/v1/2023.acl-srw.40",
               authors=["Gabriel Simmons"], venue="Proceedings of the 61st Annual Meeting of the "
               "Association for Computational Linguistics (Volume 4: Student Research Workshop)")
    found = _report(ref, [arxiv, acl])
    assert set(found) == {"version"}
    assert "cited as a work with no venue, but it was published" in found["version"]
    assert "pp. 282-297" in found["version"] and "10.18653/v1/2023.acl-srw.40" in found["version"]


def test_arxiv_journal_ref_counts_as_a_published_version() -> None:
    ref = Reference(raw="x", index=0, title="T", venue="arXiv preprint arXiv:1234.5678", year=2020)
    arxiv = Candidate(source="arxiv", title="T", year=2020, preprint=True,
                      journal_ref="NeurIPS 2021")
    assert "NeurIPS 2021" in _report(ref, [arxiv])["version"]


def test_findings_cited_as_the_main_conference_is_reported() -> None:
    ref = Reference(raw="x", index=0, title="T", venue="Proceedings of ACL 2024", year=2024)
    record = _acl(title="T", year=2024,
                  venue="Findings of the Association for Computational Linguistics: ACL 2024")
    assert "venue" in _report(ref, [record])


def test_a_different_conference_is_reported() -> None:
    ref = Reference(raw="x", index=0, title="T", venue="Proceedings of EMNLP 2023", year=2023)
    record = _acl(title="T", year=2023, venue="Proceedings of the 61st Annual Meeting of the "
                                              "Association for Computational Linguistics")
    assert "venue" in _report(ref, [record])


def test_a_cited_venue_backed_only_by_a_preprint_is_unconfirmed() -> None:
    ref = Reference(raw="x", index=0, title="T", venue="ICLR 2024", year=2024)
    arxiv = Candidate(source="arxiv", title="T", year=2023, preprint=True)
    assert "could not be confirmed" in _report(ref, [arxiv])["venue"]


def test_wrong_year_is_reported_but_online_first_is_not() -> None:
    ref = Reference(raw="x", index=0, title="T", venue="Journal of Things", year=2019)
    record = Candidate(source="crossref", title="T", venue="Journal of Things", year=2020,
                       years=(2020, 2019))
    assert "year" not in _report(ref, [record])
    ref.year = 2017
    assert "year" in _report(ref, [record])


def test_books_are_not_asked_for_a_published_version() -> None:
    ref = Reference(raw="S. Bird. NLP with Python, 2009.", index=0, title="NLP with Python",
                    kind=Kind.BOOK)
    record = Candidate(source="crossref", title="NLP with Python", venue="O'Reilly", year=2009)
    assert "version" not in _report(ref, [record])


# ---------------------------------------------------------------------------
# Sources: ACL Anthology, arXiv, DBLP, publisher pages
# ---------------------------------------------------------------------------

BIB = r"""% generated
@proceedings{acl-2020-d,
    title = "Proceedings of the 58th Annual Meeting of the ACL: System Demonstrations",
    year = "2020",
    url = "https://aclanthology.org/2020.acl-demos.0/"
}
@inproceedings{zhu-etal-2020-convlab,
    title = "{C}onv{L}ab-2: An Open-Source Toolkit for Building, Evaluating, and Diagnosing Dialogue Systems",
    author = "Zhu, Qi  and
      Zhang, Zheng  and
      Sekuli{\'c}, Ivan  and
      Zhu, Xiaoyan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations",
    month = jul,
    year = "2020",
    url = "https://aclanthology.org/2020.acl-demos.19/",
    doi = "10.18653/v1/2020.acl-demos.19",
    pages = "142--149"
}
"""


def test_anthology_index_from_its_bibtex(tmp_path: Path) -> None:
    index = Anthology.from_bibtex(BIB)
    record = index.by_id("2020.acl-demos.19")
    assert record is not None and record.exact_id
    assert record.title.startswith("ConvLab-2: An Open-Source Toolkit")
    assert record.authors == ["Qi Zhu", "Zheng Zhang", "Ivan Sekulić", "Xiaoyan Zhu"]
    assert record.pages == "142-149" and record.doi == "10.18653/v1/2020.acl-demos.19"
    assert not record.is_preprint
    # A title search survives case, hyphenation and spacing differences.
    found = index.by_title("ConvLab-2: An open-source toolkit for building, evaluating, and "
                           "diagnosing dialogue sys- tems")
    assert [r.record_id for r in found] == ["2020.acl-demos.19"]
    # Proceedings volumes are not papers.
    assert index.by_id("2020.acl-demos.0") is None
    # The on-disk build produces the same index.
    count = build_index(gzip.compress(BIB.encode()), tmp_path / "acl.db")
    assert count == 1


def test_anthology_ids() -> None:
    assert anthology_id(doi="10.18653/v1/2024.findings-acl.878") == "2024.findings-acl.878"
    assert anthology_id(url="https://aclanthology.org/2023.acl-srw.40.pdf") == "2023.acl-srw.40"
    assert anthology_id(url="https://aclanthology.org/P19-1001/") == "p19-1001"
    assert anthology_id(doi="10.1000/xyz") is None


def test_latex_escapes() -> None:
    assert latex_to_text(r"Sekuli{\'c}") == "Sekulić"
    assert latex_to_text(r"Chrupa{\l}a") == "Chrupała"
    assert latex_to_text(r"{\v{S}}najder") == "Šnajder"
    assert latex_to_text(r"M{\"u}ller") == "Müller"


FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2209.12106v2</id>
    <updated>2023-06-01T00:00:00Z</updated>
    <published>2022-09-25T00:00:00Z</published>
    <title>Moral Mimicry: Large Language Models Produce Moral
      Rationalizations Tailored to Political Identity</title>
    <author><name>Gabriel Simmons</name></author>
    <arxiv:journal_ref>ACL SRW 2023</arxiv:journal_ref>
  </entry>
</feed>"""


def test_arxiv_feed() -> None:
    [record] = parse_feed(FEED)
    assert record.record_id == "2209.12106"
    assert record.title.endswith("Tailored to Political Identity")
    assert record.year == 2022 and record.all_years == {2022, 2023}
    assert record.is_preprint and record.journal_ref == "ACL SRW 2023"
    assert parse_feed("<not xml") == []


def test_dblp_query_and_results() -> None:
    query = build_query(['a "quoted" title'])
    assert '"a \\"quoted\\" title"' in query and '"a \\"quoted\\" title."' in query
    data = {"results": {"bindings": [
        {"key": {"value": "convlab-2: x."}, "pub": {"value": "https://dblp.org/rec/conf/acl/Z20"},
         "title": {"value": "ConvLab-2: X."}, "venue": {"value": "ACL (demo)"},
         "pages": {"value": "142-149"}, "year": {"value": "2020"},
         "doi": {"value": "https://doi.org/10.18653/V1/2020.ACL-DEMOS.19"},
         "ord": {"value": str(ordinal)}, "name": {"value": name}}
        for ordinal, name in [(2, "Zheng Zhang 0001"), (1, "Qi Zhu")]
    ] + [{"key": {"value": "convlab-2: x."}, "pub": {"value": "https://dblp.org/rec/journals/corr/a"},
          "title": {"value": "ConvLab-2: X."}, "venue": {"value": "CoRR"}}]}}
    results = parse_results(data)["convlab-2: x"]
    published = next(r for r in results if not r.is_preprint)
    assert published.authors == ["Qi Zhu", "Zheng Zhang"]
    assert published.doi == "10.18653/v1/2020.acl-demos.19" and published.title == "ConvLab-2: X"
    assert any(r.is_preprint for r in results)


def test_publisher_page_metadata() -> None:
    from preflight.refcheck.sources import parse_citation_meta

    html = """<html><head>
      <meta name="citation_title" content="Attention is All you Need">
      <meta name="citation_author" content="Vaswani, Ashish">
      <meta content="Shazeer, Noam" name="citation_author">
      <meta name="citation_publication_date" content="2017">
      <meta name="citation_conference_title" content="Advances in Neural Information Processing Systems">
      <meta name="citation_firstpage" content="5998"><meta name="citation_lastpage" content="6008">
    </head></html>"""
    record = parse_citation_meta(html, "https://proceedings.neurips.cc/paper/2017/hash/x.html")
    assert record is not None
    assert record.authors == ["Vaswani, Ashish", "Shazeer, Noam"]
    assert record.pages == "5998-6008" and record.year == 2017
    assert record.source == "proceedings.neurips.cc"
    assert parse_citation_meta("<html></html>", "https://x") is None


def test_batcher_coalesces_and_reports_failure_as_unknown() -> None:
    calls: list[list[str]] = []

    async def fetch(kind: str, keys: list[str]) -> dict[str, list[Candidate]]:
        calls.append(keys)
        if "boom" in keys:
            raise RuntimeError("host down")
        return {k: [Candidate(source="x", title=k)] for k in keys if k != "missing"}

    async def main() -> None:
        batcher = Batcher(fetch, interval=0.0, size=10, window=0.01)
        a, b, c = await asyncio.gather(batcher.get("t", "a"), batcher.get("t", "b"),
                                       batcher.get("t", "missing"))
        assert [x.title for x in a] == ["a"] and [x.title for x in b] == ["b"] and c == []
        assert calls == [["a", "b", "missing"]]          # one request for all three
        with pytest.raises(LookupError):
            await batcher.get("t", "boom")
        batcher.close()

    asyncio.run(main())


def test_robots_txt_is_honoured_for_publisher_pages() -> None:
    from preflight.refcheck.sources import Session, landing_page

    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /paper/\n")
        return httpx.Response(200, text='<meta name="citation_title" content="T">')

    async def main() -> list[Candidate]:
        async with Session() as session:
            session._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            return await landing_page(session, "https://proceedings.neurips.cc/paper/2017/x.html")

    assert asyncio.run(main()) == []
    assert requested == ["/robots.txt"]            # the page itself was never requested


def _throttled_session(monkeypatch, statuses: list[int], headers: dict[str, str] | None = None,
                       **get_kwargs):
    """Serve ``statuses`` in order from one host; return (response, session, sleeps)."""
    from preflight.refcheck import sources

    sleeps: list[float] = []

    async def no_wait(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(sources.asyncio, "sleep", no_wait)
    queue = list(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        status = queue.pop(0) if queue else 200
        return httpx.Response(status, headers=headers if status == 429 else None, json={})

    async def main():
        async with sources.Session() as session:
            session._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            response = await session.get("https://api.example.org/works", rate=100.0, **get_kwargs)
            return response, session

    response, session = asyncio.run(main())
    return response, session, sleeps


def test_a_throttled_request_is_retried_until_it_is_answered(monkeypatch) -> None:
    response, session, sleeps = _throttled_session(monkeypatch, [429, 429, 503, 200])
    assert response is not None and response.status_code == 200
    assert session.throttled["api.example.org"] == 3
    assert "api.example.org" not in session.errors
    assert len(sleeps) == 3 and sleeps[2] >= sleeps[0] / 2   # exponential, with jitter


def test_retry_after_is_honoured_as_seconds_or_a_date(monkeypatch) -> None:
    from email.utils import formatdate

    from preflight.refcheck.sources import _retry_after

    _, _, sleeps = _throttled_session(monkeypatch, [429, 200], headers={"Retry-After": "7"})
    assert 7.0 <= sleeps[0] <= 8.0
    dated = httpx.Response(429, headers={"Retry-After": formatdate(time.time() + 30, usegmt=True)})
    assert 25.0 <= _retry_after(dated) <= 31.0
    assert _retry_after(httpx.Response(429, headers={"Retry-After": "soon"})) is None


def test_a_persistent_throttle_gives_up_as_unknown(monkeypatch) -> None:
    from preflight.refcheck.sources import THROTTLE_RETRIES

    response, session, sleeps = _throttled_session(monkeypatch, [429] * 20)
    assert response is None
    assert len(sleeps) == THROTTLE_RETRIES
    assert session.errors["api.example.org"] == "rate limited (429)"
    assert "api.example.org" not in session.disabled          # busy, not gone


def test_a_host_asking_for_a_long_wait_is_dropped_at_once(monkeypatch) -> None:
    response, session, sleeps = _throttled_session(monkeypatch, [429, 200],
                                                   headers={"Retry-After": "3600"})
    assert response is None and sleeps == []
    assert "api.example.org" in session.disabled
    assert "60 minutes" in session.errors["api.example.org"]


def test_only_consecutive_refusals_count_against_a_host(monkeypatch) -> None:
    """A burst of 429s that ends in an answer resets the count toward abandoning the host."""
    from preflight.refcheck import sources

    async def no_wait(delay: float) -> None:
        return None

    monkeypatch.setattr(sources.asyncio, "sleep", no_wait)
    queue = [429, 429, 200, 429, 429, 200]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(queue.pop(0) if queue else 200, json={})

    async def main():
        async with sources.Session() as session:
            session.throttle_budget = 3
            session._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            answers = [await session.get("https://api.example.org/w", rate=100.0) for _ in range(2)]
            return answers, session

    answers, session = asyncio.run(main())
    assert [a.status_code for a in answers] == [200, 200]
    assert session.throttled["api.example.org"] == 4 and not session.disabled


def test_the_model_client_retries_rate_limits() -> None:
    from preflight.llm.client import AsyncLLMClient

    client = AsyncLLMClient(api_key="sk-test")
    assert client._client().max_retries == client.max_retries == 5


# ---------------------------------------------------------------------------
# The engine: positive confirmation only
# ---------------------------------------------------------------------------

def _engine_run(monkeypatch, references, *, sources=(), resolve=None, client=None, live=None):
    from preflight.refcheck import engine
    from preflight.refcheck.sources import DoiLookup, SourceSpec

    async def fake_resolve(session, doi):
        return resolve(doi) if resolve else DoiLookup(None)

    async def fake_liveness(session, url):
        return live(url) if live else None     # never the network: nothing could be read

    monkeypatch.setattr(engine, "resolve_doi", fake_resolve)
    monkeypatch.setattr(engine, "liveness", fake_liveness)
    specs = tuple(SourceSpec(name, fetch) for name, fetch in sources)
    monkeypatch.setattr(engine, "_select_sources", lambda config: specs)
    config = engine.RefCheckConfig(use_web_search=client is not None, cache_path=None,
                                   use_anthology=False, use_arxiv=False, use_dblp=False)
    return asyncio.run(engine.verify(references, config, client))


def test_a_doi_pointing_elsewhere_is_reported_with_the_right_one(monkeypatch) -> None:
    from preflight.refcheck.sources import DoiLookup

    title = "Two tales of persona in LLMs: A survey of role-playing and personalization"
    ref = Reference(raw=f"[50] Y. Tseng et al. {title}. Findings of EMNLP 2024, pages 3769–3791. "
                        "doi: 10.18653/v1/2024.findings-emnlp.218.", index=0, title=title,
                    authors=["Yu-Min Tseng"], year=2024, pages="3769–3791",
                    venue="Findings of the Association for Computational Linguistics: EMNLP 2024",
                    doi="10.18653/v1/2024.findings-emnlp.218", kind=Kind.PAPER)
    wrong = Candidate(source="crossref", exact_id=True, title="Scaling Behavior for Large Language "
                      "Models regarding Numeral Systems", authors=["Zhejian Zhou"], year=2024)
    right = _acl(title="Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization",
                 authors=["Yu-Min Tseng"], year=2024, pages="16612-16631",
                 doi="10.18653/v1/2024.findings-emnlp.969",
                 venue="Findings of the Association for Computational Linguistics: EMNLP 2024")

    async def acl(session, reference):
        return [right]

    report = _engine_run(monkeypatch, [ref], sources=[("acl_anthology", acl)],
                         resolve=lambda doi: DoiLookup(True, wrong))
    [verdict] = report.verdicts
    assert verdict.status is Status.DETAILS_MISMATCH
    found = {d.field: d.detail for d in verdict.discrepancies}
    assert "belongs to a different work" in found["doi"] and "Scaling Behavior" in found["doi"]
    assert "this work's DOI is 10.18653/v1/2024.findings-emnlp.969" in found["doi"]
    assert found["pages"] == "pages 3769–3791; the record has 16612-16631"
    assert report.miscited == [verdict] and not verdict.is_suspicious


def test_an_unregistered_doi_is_reported(monkeypatch) -> None:
    from preflight.refcheck.sources import DoiLookup

    ref = Reference(raw="A. B. A real paper title here. In ACL, 2020. doi:10.1/nope", index=0,
                    title="A real paper title here", doi="10.1/nope", kind=Kind.PAPER)

    async def crossref(session, reference):
        return [Candidate(source="crossref", title="A Real Paper Title Here", year=2020)]

    report = _engine_run(monkeypatch, [ref], sources=[("crossref", crossref)],
                         resolve=lambda doi: DoiLookup(False))
    [verdict] = report.verdicts
    assert verdict.status is Status.DETAILS_MISMATCH
    assert "is not registered" in verdict.discrepancies[0].detail


class _FakeSearch:
    """A web search that claims to have found the work at a given URL."""

    available = True

    def __init__(self, url: str | None) -> None:
        self.url = url

    async def web_search(self, *args, **kwargs):
        return ({"found": True, "confidence": "high", "matched_title": "A Paper",
                 "record_url": self.url, "doi": None, "arxiv_id": None,
                 "note": "found it"}, [self.url] if self.url else [])


def test_web_search_alone_does_not_confirm_a_paper(monkeypatch) -> None:
    """The model saying "found" is a lead. With nothing behind it, the paper is unconfirmed."""
    ref = Reference(raw="A. B. A paper nobody indexes at all. In ICML, 2023.", index=0,
                    title="A paper nobody indexes at all", kind=Kind.PAPER)
    report = _engine_run(monkeypatch, [ref], client=_FakeSearch("https://example.com/paper"))
    [verdict] = report.verdicts
    assert verdict.status is Status.UNCONFIRMED and verdict.is_suspicious
    assert "no key source confirms it" in verdict.note
    assert "Nothing could be read at https://example.com/paper" in verdict.note


def _paper_nobody_indexes() -> Reference:
    return Reference(raw="A. B. A paper nobody indexes at all. In ICML, 2023.", index=0,
                     title="A paper nobody indexes at all", authors=["A. B."], kind=Kind.PAPER)


def test_a_web_search_that_reaches_a_live_page_is_web_only(monkeypatch) -> None:
    from preflight.refcheck.websources import Resolution

    report = _engine_run(monkeypatch, [_paper_nobody_indexes()],
                         client=_FakeSearch("https://lab.example.edu/papers/x.pdf"),
                         live=lambda url: Resolution(True, "URL resolves", url=url, source="url"))
    [verdict] = report.verdicts
    assert verdict.status is Status.WEB_ONLY and verdict.is_suspicious
    assert verdict.url == "https://lab.example.edu/papers/x.pdf"
    assert "located it at https://lab.example.edu/papers/x.pdf" in verdict.note


@pytest.mark.parametrize("case", ["no page", "blocked page", "different work"])
def test_a_web_search_that_confirms_nothing_stays_unconfirmed(monkeypatch, case: str) -> None:
    """A bare "found", a page that refuses us, or a record of another work are not a trace."""
    from preflight.refcheck import engine
    from preflight.refcheck.websources import Resolution

    other = Candidate(source="proceedings.mlr.press", title="An Entirely Different Paper",
                      authors=["C. D."], year=2021)

    async def fake_confirm(session, url, reference):
        return [other] if case == "different work" else []

    monkeypatch.setattr(engine, "confirm_url", fake_confirm)
    url = None if case == "no page" else "https://proceedings.mlr.press/v139/x"
    blocked = Resolution(True, "URL exists but blocked automated access (403)", url=url, blocked=True)
    report = _engine_run(monkeypatch, [_paper_nobody_indexes()], client=_FakeSearch(url),
                         live=lambda u: blocked if case == "blocked page" else
                         Resolution(True, "URL resolves", url=u))
    [verdict] = report.verdicts
    assert verdict.status is Status.UNCONFIRMED, verdict.note
    expected = {"no page": "gave no page or identifier",
                "blocked page": "Nothing could be read",
                "different work": "'An Entirely Different Paper'"}[case]
    assert expected in verdict.note


def test_a_web_search_lead_is_confirmed_at_its_source(monkeypatch) -> None:
    from preflight.refcheck import engine

    ref = Reference(raw="A. Smith. A paper only its publisher lists. In ICML, 2023.", index=0,
                    title="A paper only its publisher lists", authors=["A. Smith"], year=2023,
                    venue="ICML", kind=Kind.PAPER)
    record = Candidate(source="proceedings.mlr.press", title="A Paper Only Its Publisher Lists",
                       authors=["Alice Smith"], year=2023, exact_id=True,
                       venue="Proceedings of the 40th International Conference on Machine Learning")

    async def fake_confirm(session, url, reference):
        return [record] if "mlr.press" in url else []

    monkeypatch.setattr(engine, "confirm_url", fake_confirm)
    report = _engine_run(monkeypatch, [ref], client=_FakeSearch("https://proceedings.mlr.press/v202/x"))
    [verdict] = report.verdicts
    assert verdict.status is Status.VERIFIED
    assert "found by web search and confirmed" in verdict.note


def test_the_cache_is_keyed_on_the_citation_text() -> None:
    from preflight.refcheck.engine import _cache_key

    a = Reference(raw="[3] X. Zhao. A title. 2020.", index=2, kind=Kind.PAPER)
    b = Reference(raw="[7]  X.  Zhao. A title.  2020.", index=6, kind=Kind.PAPER)
    c = Reference(raw="[3] X. Zhu. A title. 2020.", index=2, kind=Kind.PAPER)
    assert _cache_key(a) == _cache_key(b)          # renumbering and reflowing: same entry
    assert _cache_key(a) != _cache_key(c)          # an edited entry is checked again


def test_discrepancies_survive_the_cache(tmp_path: Path) -> None:
    from preflight.refcheck.core import Verdict
    from preflight.refcheck.engine import _from_cache, _remember
    from preflight.refcheck.store import Store

    ref = Reference(raw="x", index=0)
    verdict = Verdict(reference=ref, status=Status.DETAILS_MISMATCH,
                      discrepancies=[Discrepancy("pages", "pages 1-2; the record has 3-4", "dblp")])
    with Store(tmp_path / "c.db") as store:
        _remember(store, "k", verdict)
        restored = _from_cache(ref, store.get("k"))
    assert restored.status is Status.DETAILS_MISMATCH
    assert restored.discrepancies == verdict.discrepancies


def test_a_parse_filed_under_the_wrong_entry_is_discarded() -> None:
    """The model once returned [81]'s fields for [82]; they must not stick."""
    from preflight.refcheck.parse import _apply

    ref = Reference(raw="[82] Qi Zhu, Zheng Zhang. ConvLab-2: An open-source toolkit for building, "
                        "evaluating, and diagnosing dialogue systems. In ACL, pages 71–78, 2020.",
                    index=81)
    _apply(ref, {"title": "The pragmatic mind of machines: Tracing the emergence of pragmatic "
                          "competence", "authors": ["Kefan Yu"], "year": 2026})
    assert ref.title is None and ref.authors == []
    _apply(ref, {"title": "ConvLab-2: An open-source toolkit for building, evaluating, and "
                          "diagnosing dialogue systems", "authors": ["Qi Zhu", "Zheng Zhang"],
                 "year": 2020, "pages": "71–78"})
    assert ref.title.startswith("ConvLab-2") and ref.pages == "71–78"


def test_published_limits_are_adopted() -> None:
    from preflight.refcheck.sources import Session

    async def main() -> Session:
        async with Session() as session:
            session._learn("api.crossref.org", httpx.Response(200, headers={
                "x-rate-limit-limit": "1", "x-rate-limit-interval": "1s", "x-concurrency-limit": "1"}))
            session._learn("api2.openreview.net", httpx.Response(200, headers={
                "ratelimit-policy": "20;w=60"}))
            return session

    session = asyncio.run(main())
    assert session.limiter("api.crossref.org", 8.0).per_second <= 0.9
    assert session.slots("api.crossref.org").limit == 1
    assert session.limiter("api2.openreview.net", 2.0).per_second <= 0.31


def test_a_row_set_in_two_fonts_is_read_left_to_right() -> None:
    """An italic title a hair above its row must stay inside its own entry."""
    from preflight.document import Line, Span
    from preflight.refcheck.parse import _rows

    def line(text: str, x0: float, y0: float, x1: float) -> Line:
        box = (x0, y0, x1, y0 + 10.1)
        return Line(spans=[Span(text, box, "Times", 10.0, 0, 0, 14)], bbox=box, page=14)

    reading_order = [
        line("alignment via in-context learning, 2024.", 129.6, 167.8, 287.4),
        line("Intention, Plans, and Practical Reason.", 224.7, 187.0, 390.2),   # italic, 0.1pt higher
        line("[62] Michael E. Bratman.", 108.0, 187.1, 216.4),
        line("Harvard University Press,", 398.5, 187.1, 505.2),
        line("Cambridge, MA, 1987.", 129.6, 198.1, 221.3),
    ]
    assert [x.text for x in _rows(reading_order, [108.0])] == [
        "alignment via in-context learning, 2024.",
        "[62] Michael E. Bratman.",
        "Intention, Plans, and Practical Reason.",
        "Harvard University Press,",
        "Cambridge, MA, 1987.",
    ]


def test_nicknames_and_detached_accents_are_the_same_person() -> None:
    assert compare_authors(["Thomas Henighan", "Milica Gaši´c"],
                           [["Tom Henighan", "Milica Gašić"]], truncated=False) == []
    # ...but a different person with the same surname is still reported.
    assert compare_authors(["Todd Henighan"], [["Tom Henighan"]], truncated=False) == [
        "Todd Henighan should be Tom Henighan"]


def test_list_conjunctions_and_truncation_markers_are_not_part_of_a_name() -> None:
    """A list split on commas keeps "and" on its last name; that is the same person."""
    assert split_name("and Madian Khabsa").given == ("madian",)
    assert split_name("& Jane Doe").raw == "Jane Doe"
    assert split_name("Hugo Touvron and others").surname == "touvron"
    assert split_name("Jane Doe et al.").surname == "doe"
    assert split_name("and others") is None
    assert split_name("Anderson Smith").given == ("anderson",)
    llama_guard = ["Hakan Inan", "Kartikeya Upasani", "Jianfeng Chi", "Madian Khabsa"]
    assert compare_authors(["Hakan Inan", "Kartikeya Upasani", "Jianfeng Chi", "and Madian Khabsa"],
                           [llama_guard], truncated=False) == []
    assert compare_authors(["Nina Panickssery", "and Alexander Matt Turner"],
                           [["Nina Panickssery", "Alexander Turner"]], truncated=False) == []
    assert compare_authors(["Hakan Inan and others"], [llama_guard], truncated=True) == []
    # ...but a wrong name after the conjunction is still reported, without the "and".
    assert compare_authors(["Hakan Inan", "and Martin Khabsa"], [["Hakan Inan", "Madian Khabsa"]],
                           truncated=False) == ["Martin Khabsa should be Madian Khabsa"]


def test_a_cited_venue_is_looked_for_when_only_the_preprint_is_found(monkeypatch) -> None:
    """Title changed between arXiv and NeurIPS: the search finds it, the publisher confirms it."""
    from preflight.refcheck import engine

    ref = Reference(raw="N. Stiennon et al. Learning to summarize from human feedback. In Advances "
                        "in Neural Information Processing Systems, pages 3008–3021, 2020.", index=0,
                    title="Learning to summarize from human feedback", authors=["Nisan Stiennon"],
                    venue="Advances in Neural Information Processing Systems", year=2020,
                    pages="3008–3021", kind=Kind.PAPER)
    preprint = Candidate(source="arxiv", title="Learning to summarize from human feedback",
                         authors=["Nisan Stiennon"], year=2020, preprint=True)
    published = Candidate(source="proceedings.neurips.cc", exact_id=True,
                          title="Learning to summarize with human feedback", authors=["Nisan Stiennon"],
                          year=2020, pages="3008-3021",
                          venue="Advances in Neural Information Processing Systems")

    async def arxiv(session, reference):
        return [preprint]

    async def fake_confirm(session, url, reference):
        return [published]

    monkeypatch.setattr(engine, "confirm_url", fake_confirm)
    report = _engine_run(monkeypatch, [ref], sources=[("arxiv", arxiv)],
                         client=_FakeSearch("https://proceedings.neurips.cc/paper/2020/x"))
    [verdict] = report.verdicts
    assert verdict.status is Status.VERIFIED, verdict.discrepancies
    assert report.searched == 1


def test_a_cited_venue_that_cannot_be_found_is_still_reported(monkeypatch) -> None:
    ref = Reference(raw="A. B. Some paper. In ICLR, 2026.", index=0, title="Some paper about things",
                    venue="ICLR 2026", year=2026, kind=Kind.PAPER)
    preprint = Candidate(source="arxiv", title="Some paper about things", year=2025, preprint=True)

    async def arxiv(session, reference):
        return [preprint]

    report = _engine_run(monkeypatch, [ref], sources=[("arxiv", arxiv)],
                         client=_FakeSearch("https://arxiv.org/abs/2501.00001"))
    [verdict] = report.verdicts
    assert verdict.status is Status.DETAILS_MISMATCH
    assert "could not be confirmed" in verdict.discrepancies[0].detail


def test_dblp_author_query_and_keys() -> None:
    from preflight.refcheck.dblp import author_key, build_author_query

    query = build_author_query([author_key("Nisan  Stiennon", 2020)])
    for year in (2019, 2020, 2021):
        assert f'"Nisan Stiennon" "{year}"^^xsd:gYear' in query
    data = {"results": {"bindings": [
        {"key": {"value": "Nisan Stiennon|2020"}, "pub": {"value": "https://dblp.org/rec/conf/nips/S20"},
         "title": {"value": "Learning to summarize with human feedback."}, "venue": {"value": "NeurIPS"},
         "year": {"value": "2020"}}]}}
    [record] = parse_results(data)["Nisan Stiennon|2020"]
    assert record.venue == "NeurIPS" and not record.is_preprint


def test_a_retitled_published_version_is_found_by_its_author(monkeypatch) -> None:
    """No title lookup links "from human feedback" (arXiv) to "with human feedback" (NeurIPS)."""
    from preflight.refcheck import engine

    ref = Reference(raw="N. Stiennon et al. Learning to summarize from human feedback. In Advances "
                        "in Neural Information Processing Systems, pages 3008–3021, 2020.", index=0,
                    title="Learning to summarize from human feedback", authors=["Nisan Stiennon"],
                    venue="Advances in Neural Information Processing Systems", year=2020,
                    pages="3008–3021", kind=Kind.PAPER)
    preprint = Candidate(source="arxiv", title="Learning to summarize from human feedback",
                         authors=["Nisan Stiennon"], year=2020, preprint=True)
    published = Candidate(source="dblp", title="Learning to summarize with human feedback",
                          authors=["Nisan Stiennon"], year=2020, venue="NeurIPS", pages="3008-3021",
                          preprint=False)
    asked: list[tuple[str, int]] = []

    class FakeDblp:
        async def by_author(self, name, year):
            asked.append((name, year))
            return [published]

        def close(self):
            pass

    async def arxiv(session, reference):
        return [preprint]

    original = engine._start_batched_sources

    def attach(session, config, pending):
        original(session, config, pending)
        session.dblp = FakeDblp()

    monkeypatch.setattr(engine, "_start_batched_sources", attach)
    report = _engine_run(monkeypatch, [ref], sources=[("arxiv", arxiv)])
    [verdict] = report.verdicts
    assert asked == [("Nisan Stiennon", 2020)]
    assert verdict.status is Status.VERIFIED, verdict.discrepancies


def test_a_book_needs_a_key_source_too(monkeypatch) -> None:
    """A book no source confirms is reported, not waved through as "not indexed"."""
    ref = Reference(raw="M. E. Bratman. Intention, Plans, and Practical Reason. Harvard University "
                        "Press, 1987.", index=0, title="Intention, Plans, and Practical Reason",
                    kind=Kind.BOOK)
    report = _engine_run(monkeypatch, [ref])
    assert report.verdicts[0].status is Status.NOT_FOUND


def test_a_link_to_a_key_source_is_confirmed_there_not_by_liveness(monkeypatch) -> None:
    """The parser called it a technical report, but it links to arXiv: check the arXiv record."""
    from preflight.refcheck import engine

    ref = Reference(raw="B. Emi and M. Spero. Technical report on the Pangram AI-generated text "
                        "classifier, 2024. https://arxiv.org/abs/2402.14873", index=0,
                    title="Technical report on the Pangram AI-generated text classifier",
                    url="https://arxiv.org/abs/2402.14873", kind=Kind.STANDARD)
    touched: list[str] = []

    async def fake_web(session, reference):
        touched.append(reference.url)
        from preflight.refcheck.websources import Resolution
        return Resolution(True, "URL resolves", url=reference.url, source="url")

    async def arxiv(session, reference):
        return [Candidate(source="arxiv", title="Technical Report on the Pangram AI-Generated Text "
                          "Classifier", authors=["Bradley Emi", "Max Spero"], year=2024, preprint=True)]

    monkeypatch.setattr(engine, "resolve_web", fake_web)
    report = _engine_run(monkeypatch, [ref], sources=[("arxiv", arxiv)])
    assert touched == []
    assert report.verdicts[0].status is Status.VERIFIED


def test_an_entry_of_unknown_kind_is_not_excused(monkeypatch) -> None:
    """[43] parsed as "unknown"; an unconfirmed unknown entry may be a paper, so it is reported."""
    ref = Reference(raw="G. Simmons. Moral mimicry: Large language models produce moral "
                        "rationalizations tailored to political identity, 2022.", index=0,
                    title="Moral mimicry: Large language models produce moral rationalizations",
                    kind=Kind.UNKNOWN)
    report = _engine_run(monkeypatch, [ref])
    assert report.verdicts[0].status is Status.NOT_FOUND and report.verdicts[0].is_suspicious


def _findings(clean_paper: Path, verdicts: list) -> dict:
    """Run the three reference checks over a prepared report."""
    from preflight.checks.references import (
        check_reference_details,
        check_reference_verification,
        check_reference_versions,
    )
    from preflight.context import CheckContext, Settings
    from preflight.document import Document
    from preflight.profile import load_profile
    from preflight.refcheck.engine import RefCheckReport

    profile = load_profile("iclr")
    ctx = CheckContext(doc=Document(clean_paper), profile=profile, track=profile.track(None),
                       settings=Settings())
    ctx.shared["refcheck_report"] = {"entries": ["x"] * len(verdicts),
                                     "references": [v.reference for v in verdicts],
                                     "report": RefCheckReport(verdicts=verdicts)}
    found: dict = {}
    try:
        for check in (check_reference_verification, check_reference_details, check_reference_versions):
            result = asyncio.run(check(ctx))
            results = result if isinstance(result, list) else [result]
            found[check.__name__] = results[0]
            found.update({f.check_id: f for f in results})
        return found
    finally:
        ctx.doc.close()


def test_wrong_details_and_unconfirmed_papers_are_errors(clean_paper: Path) -> None:
    from preflight.models import Severity
    from preflight.refcheck.core import Verdict

    wrong = Verdict(Reference(raw="[21] x", index=0, title="RoleLLM"), Status.DETAILS_MISMATCH,
                    discrepancies=[Discrepancy("doi", "DOI 10.1/x belongs to a different work"),
                                   Discrepancy("version", "cited as a preprint, but it was published")])
    missing = Verdict(Reference(raw="[3] y", index=1, title="A paper nobody has"), Status.UNCONFIRMED)
    found = _findings(clean_paper, [wrong, missing])
    assert found["check_reference_verification"].severity is Severity.ERROR
    details = found["check_reference_details"]
    assert details.severity is Severity.ERROR
    # The version note is not repeated among the errors...
    assert "version" not in details.evidence[0].detail
    assert details.evidence[0].quote == "[21] RoleLLM"
    # ...it is its own finding, and a warning.
    assert found["check_reference_versions"].severity is Severity.WARNING


def test_a_reference_found_only_by_web_search_is_a_warning(clean_paper: Path) -> None:
    from preflight.models import Severity
    from preflight.refcheck.core import Verdict

    traced = Verdict(Reference(raw="[7] z", index=0, title="A lab report"), Status.WEB_ONLY,
                     source="web_search", note="a web search located it at https://x.edu/r.pdf")
    found = _findings(clean_paper, [traced])
    assert found["reference_verification"].severity is Severity.PASS
    web = found["reference_verification.web_only"]
    assert web.severity is Severity.WARNING
    assert "[found only by web search]" in web.evidence[0].detail

    # Beside a reference nobody can find, each keeps its own severity.
    missing = Verdict(Reference(raw="[3] y", index=1, title="A paper nobody has"), Status.NOT_FOUND)
    found = _findings(clean_paper, [traced, missing])
    assert found["reference_verification"].severity is Severity.ERROR
    assert "1 of 2 reference(s)" in found["reference_verification"].message
    assert found["reference_verification.web_only"].severity is Severity.WARNING


def test_a_preprint_with_a_published_version_is_only_a_warning(clean_paper: Path) -> None:
    from preflight.models import Severity
    from preflight.refcheck.core import Verdict

    preprint = Verdict(Reference(raw="[43] x", index=0, title="Moral mimicry"), Status.DETAILS_MISMATCH,
                       discrepancies=[Discrepancy("version", "cited with no venue, but it was published")])
    found = _findings(clean_paper, [preprint])
    assert found["check_reference_verification"].severity is Severity.PASS
    assert found["check_reference_details"].severity is Severity.PASS
    assert found["check_reference_versions"].severity is Severity.WARNING
