"""The verification pipeline.

Four tiers, and a reference leaves as soon as one of them answers:

1. **Cache.** Same references recur across drafts.
2. **Bulk databases, fully concurrent.** Every applicable source for every
   reference is issued at once under per-host rate limits. Nothing waits on a
   database it did not need.
3. **Route by type.** A repository, a model card or a vendor blog post is
   verified against the thing it actually is, never against CrossRef.
4. **Live web search.** For what remains, a model searches the web under domain
   filters and returns a citation. This is the tier a database-only checker has
   no answer for, and it is exactly where such checkers spend their time and
   produce their false accusations.

Only after all four does a reference get reported as unverifiable.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..llm.client import AsyncLLMClient, LLMError
from .core import Candidate, MatchPolicy, Reference, Status, Verdict
from .sources import ACADEMIC_SOURCES, Session, SourceSpec
from .store import Store
from .websources import needs_web_route, resolve_web

SEARCH_DOMAINS = [
    "arxiv.org", "aclanthology.org", "openreview.net", "dl.acm.org", "ieeexplore.ieee.org",
    "link.springer.com", "sciencedirect.com", "nature.com", "science.org", "jmlr.org",
    "proceedings.mlr.press", "papers.nips.cc", "openalex.org", "semanticscholar.org",
    "doi.org", "crossref.org", "dblp.org", "pubmed.ncbi.nlm.nih.gov", "biorxiv.org",
    "github.com", "huggingface.co", "pypi.org", "rfc-editor.org", "ietf.org", "w3.org",
    "openai.com", "anthropic.com", "deepmind.google", "ai.meta.com", "blog.google",
    "microsoft.com", "nvidia.com", "wikipedia.org", "scholar.google.com",
]

SEARCH_SYSTEM = (
    "You establish whether a cited work actually exists. You search the web and report only what "
    "the results show. You never assume a work exists because its title sounds plausible, and you "
    "never declare a work fake merely because a search did not surface it. If the search results "
    "do not settle the question, say so and set `found` to false with low confidence.\n\n"
    "You are advising the paper's own authors before they submit, so a false alarm wastes their "
    "time on a citation that was fine. Report a problem only when you can point at what is wrong."
)

_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "matched_title": {"type": ["string", "null"]},
        "venue": {"type": ["string", "null"]},
        "year": {"type": ["integer", "null"]},
        "authors_match": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "note": {"type": "string"},
    },
    "required": ["found", "confidence", "matched_title", "venue", "year", "authors_match", "note"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class RefCheckConfig:
    """Everything tunable about a run."""

    max_references: int = 0            # 0 = all
    concurrency: int = 16              # references verified at once
    search_concurrency: int = 6        # live web searches at once
    timeout: float = 6.0               # per HTTP request
    total_timeout: float = 120.0       # wall clock for the whole run
    use_llm_parse: bool = True
    use_web_search: bool = True
    enabled_sources: tuple[str, ...] = ()   # empty = all non-optional
    mailto: str | None = None
    semantic_scholar_key: str | None = None
    github_token: str | None = None
    cache_path: str | None = None
    policy: MatchPolicy = field(default_factory=MatchPolicy)


@dataclass(slots=True)
class RefCheckReport:
    verdicts: list[Verdict] = field(default_factory=list)
    entries: int = 0
    parsed: int = 0
    elapsed: float = 0.0
    truncated: bool = False
    cache_hits: int = 0
    unique: int = 0
    duplicates: list[tuple[str, list[int]]] = field(default_factory=list)
    source_errors: dict[str, str] = field(default_factory=dict)
    throttled: dict[str, int] = field(default_factory=dict)
    searched: int = 0

    @property
    def suspicious(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.is_suspicious]

    def count(self, status: Status) -> int:
        return sum(1 for v in self.verdicts if v.status is status)


def _dedup_key(reference: Reference) -> str:
    """What makes two bibliography entries the same work.

    A DOI or arXiv id is definitive. Otherwise the normalised title, which also
    collapses a preprint and its published version — the same work cited twice.
    """
    if reference.doi:
        return f"doi:{reference.doi.lower()}"
    if reference.arxiv_id:
        return f"arxiv:{reference.arxiv_id}"
    key = reference.key
    return f"title:{key}" if key else f"raw:{reference.index}"


def _expand(verdicts: list[Verdict], groups: dict[str, list[Reference]]) -> list[Verdict]:
    """Give every original reference the verdict earned by its representative."""
    from dataclasses import replace as _replace

    out: list[Verdict] = []
    for verdict, group in zip(verdicts, groups.values(), strict=False):
        out.append(verdict)
        for duplicate in group[1:]:
            out.append(_replace(verdict, reference=duplicate))
    return sorted(out, key=lambda v: v.reference.index)


def _select_sources(config: RefCheckConfig) -> tuple[SourceSpec, ...]:
    if config.enabled_sources:
        wanted = set(config.enabled_sources)
        return tuple(s for s in ACADEMIC_SOURCES if s.name in wanted)
    return tuple(s for s in ACADEMIC_SOURCES if not s.optional)


async def verify(
    references: list[Reference],
    config: RefCheckConfig,
    client: AsyncLLMClient | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> RefCheckReport:
    """Verify a list of parsed references."""
    import time

    started = time.monotonic()
    report = RefCheckReport(entries=len(references))
    if config.max_references:
        references = references[: config.max_references]
        report.truncated = report.entries > len(references)
    report.parsed = len(references)
    if not references:
        return report

    # Deduplicate before doing any work. A bibliography can cite the same work
    # twice — a preprint and its proceedings version, or a plain repeat — and
    # looking it up twice costs twice and reports it twice.
    groups: dict[str, list[Reference]] = {}
    for reference in references:
        reference.classify_kind()
        groups.setdefault(_dedup_key(reference), []).append(reference)
    representatives = [group[0] for group in groups.values()]
    report.unique = len(representatives)
    report.duplicates = [
        (group[0].label, [r.index for r in group]) for group in groups.values() if len(group) > 1
    ]

    sources = _select_sources(config)
    store = Store(config.cache_path if config.cache_path is not None else None)
    limiter = asyncio.Semaphore(config.concurrency)
    search_limiter = asyncio.Semaphore(config.search_concurrency)
    done = 0

    def tick(label: str) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, len(representatives), label)

    try:
        async with Session(
            timeout=config.timeout,
            mailto=config.mailto,
            semantic_scholar_key=config.semantic_scholar_key,
            github_token=config.github_token,
        ) as session:

            async def one(reference: Reference) -> Verdict:
                async with limiter:
                    verdict = await _verify_one(
                        reference, session, sources, config, client, store, search_limiter, report
                    )
                tick(reference.label[:40])
                return verdict

            try:
                # Only the representatives are looked up; duplicates inherit the verdict.
                verdicts = await asyncio.wait_for(
                    asyncio.gather(*(one(r) for r in representatives)),
                    timeout=config.total_timeout,
                )
                report.verdicts = _expand(list(verdicts), groups)
            except TimeoutError:
                report.verdicts = [
                    Verdict(reference=r, status=Status.ERROR, note="the run exceeded its time budget")
                    for r in references
                ]
            report.source_errors = dict(session.errors)
            report.throttled = dict(session.throttled)
    finally:
        report.cache_hits = store.hits
        store.close()

    report.elapsed = time.monotonic() - started
    return report


async def _verify_one(
    reference: Reference,
    session: Session,
    sources: tuple[SourceSpec, ...],
    config: RefCheckConfig,
    client: AsyncLLMClient | None,
    store: Store,
    search_limiter: asyncio.Semaphore,
    report: RefCheckReport,
) -> Verdict:
    reference.classify_kind()

    if not reference.is_usable:
        return Verdict(reference=reference, status=Status.UNPARSED,
                       note="not enough of this entry could be parsed to look it up")

    cache_key = f"{reference.kind.value}:{reference.key}"
    cached = store.get(cache_key)
    if cached:
        return _from_cache(reference, cached)

    # Tier 3 first for things that are not papers: it is both faster and correct.
    if needs_web_route(reference) and reference.url:
        resolution = await resolve_web(session, reference)
        if resolution is not None and resolution.ok:
            verdict = Verdict(
                reference=reference, status=Status.RESOLVED, source=resolution.source,
                matched_title=resolution.title, score=1.0, note=resolution.detail,
                url=resolution.url, checked_sources=(resolution.source,),
            )
            _remember(store, cache_key, verdict, positive=True)
            return verdict
        if resolution is not None and not resolution.ok:
            verdict = Verdict(
                reference=reference, status=Status.UNRESOLVED, source=resolution.source,
                note=resolution.detail, url=resolution.url, checked_sources=(resolution.source,),
            )
            _remember(store, cache_key, verdict, positive=False)
            return verdict

    # Tier 2: the fast databases first, all at once. Only if they cannot settle
    # the reference do we pay for the rate-limited ones, so a 0.5 req/s source
    # never serialises the whole bibliography behind it.
    candidates: list[Candidate] = []
    checked: list[str] = []
    best: Candidate | None = None
    outcome, score, authors = "reject", 0.0, -1.0

    for tier in (1, 2):
        applicable = [s for s in sources if s.tier == tier and s.applies(reference)]
        if not applicable:
            continue
        results = await asyncio.gather(
            *(s.fetch(session, reference) for s in applicable), return_exceptions=True
        )
        for spec, result in zip(applicable, results, strict=False):
            checked.append(spec.name)
            if isinstance(result, list):
                candidates.extend(result)
        best, outcome, score, authors = _best(reference, candidates, config.policy)
        if outcome == "accept":
            break
    if outcome == "accept" and best is not None:
        verdict = Verdict(
            reference=reference, status=Status.VERIFIED, source=best.source,
            matched_title=best.title, score=score, author_score=authors,
            note=f"matched in {best.source}", url=best.url, checked_sources=tuple(checked),
        )
        _remember(store, cache_key, verdict, positive=True)
        return verdict

    # Tier 4: ask the live web, which is the only tier that can clear a real but
    # unindexed citation.
    if config.use_web_search and client is not None and client.available:
        async with search_limiter:
            report.searched += 1
            searched = await _web_search(reference, best, client)
        if searched is not None:
            _remember(store, cache_key, searched, positive=searched.status in
                      {Status.VERIFIED, Status.RESOLVED})
            return searched

    if outcome == "review" and best is not None:
        verdict = Verdict(
            reference=reference, status=Status.AUTHOR_MISMATCH, source=best.source,
            matched_title=best.title, score=score, author_score=authors,
            note="a close title exists but the authors or year do not line up",
            url=best.url, checked_sources=tuple(checked),
        )
        _remember(store, cache_key, verdict, positive=False)
        return verdict

    if not reference.kind.expects_index:
        # A blog post or repository that no database indexes is not suspicious;
        # it was never going to be in one.
        return Verdict(
            reference=reference, status=Status.UNINDEXED, checked_sources=tuple(checked),
            note=f"cited as {reference.kind.value}; bibliographic databases do not index these",
        )

    verdict = Verdict(
        reference=reference, status=Status.NOT_FOUND, score=score,
        checked_sources=tuple(checked),
        note=f"no match in {len(checked)} database(s)" + ("" if not checked else ""),
    )
    _remember(store, cache_key, verdict, positive=False)
    return verdict


def _best(
    reference: Reference, candidates: list[Candidate], policy: MatchPolicy
) -> tuple[Candidate | None, str, float, float]:
    """Pick the candidate that best explains the reference."""
    best: tuple[Candidate | None, str, float, float] = (None, "reject", 0.0, -1.0)
    rank = {"accept": 0, "review": 1, "reject": 2}
    for candidate in candidates:
        outcome, score, authors = policy.judge(reference, candidate)
        current = (candidate, outcome, score, authors)
        if best[0] is None:
            best = current
            continue
        # Prefer a better outcome, then a better title score, then author agreement.
        if (rank[outcome], -score, -authors) < (rank[best[1]], -best[2], -best[3]):
            best = current
    return best


async def _web_search(
    reference: Reference, near: Candidate | None, client: AsyncLLMClient
) -> Verdict | None:
    """Ask the model to search the live web for this citation."""
    described = [f"Title: {reference.title or '(unparsed)'}"]
    if reference.authors:
        described.append(f"Authors: {', '.join(reference.authors[:6])}")
    if reference.year:
        described.append(f"Year: {reference.year}")
    if reference.venue:
        described.append(f"Venue as cited: {reference.venue}")
    if reference.url:
        described.append(f"URL as cited: {reference.url}")
    if near is not None:
        described.append(
            f"A database returned a similar but not-conclusive record: {near.title!r} "
            f"({near.year or 'year unknown'})."
        )

    prompt = (
        "Determine whether the work cited below actually exists. Search for it.\n\n"
        + "\n".join(described)
        + f"\n\nFull citation as printed:\n{reference.raw[:600]}\n\n"
        "Set `found` true if the search results show a real work matching this citation. A blog "
        "post, a repository, a model card, a standard or a technical report counts as real — it "
        "does not have to be a peer-reviewed paper, and it does not have to be indexed anywhere.\n\n"
        "`found` and `note` must agree. If your note describes locating the work — naming its "
        "arXiv id, its venue, its page — then `found` is true. Report a difference in authors or "
        "venue through `authors_match` and `note`, not by setting `found` to false: a citation "
        "with the wrong author order is a real work cited imperfectly, not a missing work.\n\n"
        "Set `authors_match` to `no` ONLY when the cited authors are substantially different "
        "PEOPLE from the real ones — a citation attributed to the wrong researchers. It is not a "
        "mismatch when the citation lists fewer authors than the real work, abbreviates with "
        "'et al.', gives initials, orders names differently, or cites a corporate author "
        "('Anthropic', 'Gemini Team') for a work with many individual authors. Large technical "
        "reports routinely have dozens of authors and are routinely cited by their first few or "
        "by their lab, and reporting that as a mismatch is a false accusation. When in doubt use "
        "`unknown`.\n\n"
        "Set `found` false only when the search genuinely did not surface the work."
    )
    try:
        data, citations = await client.web_search(
            SEARCH_SYSTEM, prompt, schema=_SEARCH_SCHEMA, name="reference_check",
            allowed_domains=SEARCH_DOMAINS, context_size="low",
        )
    except LLMError as exc:
        return Verdict(reference=reference, status=Status.ERROR, note=str(exc)[:200])

    note = str(data.get("note", ""))[:300]
    url = citations[0] if citations else None
    if data.get("found"):
        if str(data.get("authors_match")) == "no":
            return Verdict(
                reference=reference, status=Status.AUTHOR_MISMATCH, source="web_search",
                matched_title=data.get("matched_title"), score=1.0,
                note=f"found on the web, but the authors differ. {note}", url=url,
                checked_sources=("web_search",),
            )
        status = Status.VERIFIED if reference.kind.is_academic else Status.RESOLVED
        return Verdict(
            reference=reference, status=status, source="web_search",
            matched_title=data.get("matched_title"), score=1.0,
            note=f"confirmed by web search ({data.get('confidence')} confidence). {note}",
            url=url, checked_sources=("web_search",),
        )

    if not reference.kind.expects_index:
        return Verdict(
            reference=reference, status=Status.UNRESOLVED, source="web_search",
            note=f"not found by web search either. {note}", url=url,
            checked_sources=("web_search",),
        )
    return Verdict(
        reference=reference, status=Status.NOT_FOUND, source="web_search",
        note=f"no database match, and web search did not find it. {note}",
        checked_sources=("web_search",),
    )


def _remember(store: Store, key: str, verdict: Verdict, positive: bool) -> None:
    store.put(key, {
        "status": verdict.status.value,
        "source": verdict.source,
        "matched_title": verdict.matched_title,
        "score": verdict.score,
        "author_score": verdict.author_score,
        "note": verdict.note,
        "url": verdict.url,
        "checked_sources": list(verdict.checked_sources),
    }, positive=positive)


def _from_cache(reference: Reference, data: dict[str, Any]) -> Verdict:
    try:
        status = Status(str(data.get("status")))
    except ValueError:
        status = Status.ERROR
    return Verdict(
        reference=reference,
        status=status,
        source=data.get("source"),
        matched_title=data.get("matched_title"),
        score=float(data.get("score") or 0.0),
        author_score=float(data.get("author_score") if data.get("author_score") is not None else -1.0),
        note=str(data.get("note") or "") + " (cached)",
        url=data.get("url"),
        checked_sources=tuple(data.get("checked_sources") or ()),
    )
