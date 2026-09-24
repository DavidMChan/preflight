"""The verification pipeline.

A paper passes only on positive confirmation: a record from a key source — the
DOI registry, the ACL Anthology, arXiv, DBLP, CrossRef, OpenAlex, OpenReview,
or a publisher's own landing page — whose title and authors match the citation.
Then every field that record contradicts is reported: a DOI that belongs to a
different paper, wrong pages, a misspelled or missing author, a preprint cited
in place of its published version.

1. **Cache.** Keyed on the citation's own text, so editing an entry re-checks it.
2. **Identifiers.** Every DOI, arXiv id and ACL Anthology link the citation
   prints is resolved, and must lead to the work the citation names.
3. **Key sources.** Tier 1 for every reference; tier 2 when tier 1 did not
   confirm it, or confirmed a different version than the one cited. arXiv and
   DBLP are batched and prefetched, so they cost a few requests per run.
4. **Web search, as discovery.** A model searches the web for what is left,
   but its answer is only a lead: the URL or identifier it returns is fetched
   from the key source and matched like any other record. A work the web
   search reports but no key source confirms is reported as unconfirmed.

A blog post, repository or model card is checked against what it actually is
(see :mod:`.websources`) rather than against a bibliographic index.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Any

from ..llm.client import AsyncLLMClient, LLMError
from .anthology import Anthology, anthology_id
from .arxiv import Arxiv
from .compare import cites_preprint, compare, compare_authors
from .core import Candidate, Discrepancy, MatchPolicy, Reference, Status, Verdict, title_agrees
from .dblp import Dblp
from .sources import ACADEMIC_SOURCES, Session, SourceSpec, confirm_url, resolve_doi, scholarly_url
from .store import Store
from .websources import liveness, needs_web_route, resolve_web

#: Bump when the verdict logic changes, so verdicts reached by older logic are
#: not served from the cache.
CACHE_VERSION = "v6"

SEARCH_DOMAINS = [
    "arxiv.org", "aclanthology.org", "openreview.net", "dl.acm.org", "ieeexplore.ieee.org",
    "link.springer.com", "sciencedirect.com", "nature.com", "science.org", "jmlr.org",
    "proceedings.mlr.press", "papers.nips.cc", "proceedings.neurips.cc", "openaccess.thecvf.com",
    "ojs.aaai.org", "ijcai.org", "openalex.org", "semanticscholar.org", "doi.org", "crossref.org",
    "dblp.org", "pubmed.ncbi.nlm.nih.gov", "biorxiv.org", "github.com", "huggingface.co",
    "pypi.org", "rfc-editor.org", "ietf.org", "w3.org", "openai.com", "anthropic.com",
    "deepmind.google", "ai.meta.com", "blog.google", "microsoft.com", "nvidia.com",
    "wikipedia.org", "scholar.google.com",
]

SEARCH_SYSTEM = (
    "You locate the official record of a cited work: its DOI, its arXiv id, or the URL of the page "
    "where its publisher, proceedings or repository lists it. You search the web and report only "
    "what the results show. You never assume a work exists because its title sounds plausible. "
    "What you return is checked against the source you name, so a record URL that does not list "
    "this exact work is worse than none."
)

_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "matched_title": {"type": ["string", "null"]},
        "record_url": {"type": ["string", "null"]},
        "doi": {"type": ["string", "null"]},
        "arxiv_id": {"type": ["string", "null"]},
        "note": {"type": "string"},
    },
    "required": ["found", "confidence", "matched_title", "record_url", "doi", "arxiv_id", "note"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class RefCheckConfig:
    """Everything tunable about a run."""

    max_references: int = 0            # 0 = all
    concurrency: int = 16              # references verified at once
    search_concurrency: int = 6        # live web searches at once
    timeout: float = 6.0               # per HTTP request
    total_timeout: float = 150.0       # wall clock for the whole run
    use_llm_parse: bool = True
    use_web_search: bool = True
    use_anthology: bool = True
    use_arxiv: bool = True
    use_dblp: bool = True
    enabled_sources: tuple[str, ...] = ()   # empty = all non-optional
    mailto: str | None = None
    semantic_scholar_key: str | None = None
    github_token: str | None = None
    openalex_key: str | None = None
    cache_path: str | None = None
    anthology_path: str | None = None
    anthology_max_age_days: float = 7.0
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
    timed_out: int = 0

    @property
    def suspicious(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.is_suspicious]

    @property
    def miscited(self) -> list[Verdict]:
        """References whose details a key source contradicts."""
        return [v for v in self.verdicts if v.discrepancies]

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


def _cache_key(reference: Reference) -> str:
    """The citation's own text, minus its label: any edit to the entry re-checks it."""
    text = re.sub(r"^\s*\[\d{1,4}\]\s*", "", reference.raw)
    digest = hashlib.sha1(re.sub(r"\s+", "", text.lower()).encode()).hexdigest()[:24]
    return f"{CACHE_VERSION}:{digest}"


def _expand(verdicts: list[Verdict], groups: dict[str, list[Reference]]) -> list[Verdict]:
    """Give every original reference the verdict earned by its representative."""
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


@dataclass(slots=True)
class _Run:
    """What every reference's verification shares."""

    session: Session
    sources: tuple[SourceSpec, ...]
    config: RefCheckConfig
    client: AsyncLLMClient | None
    search_limiter: asyncio.Semaphore
    report: RefCheckReport


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

    store = Store(config.cache_path if config.cache_path is not None else None)
    limiter = asyncio.Semaphore(config.concurrency)
    done = 0

    def tick(label: str) -> None:
        nonlocal done
        done += 1
        if progress is not None:
            progress(done, len(representatives), label)

    verdicts: dict[int, Verdict] = {}
    pending: list[tuple[int, Reference]] = []
    for position, reference in enumerate(representatives):
        cached = store.get(_cache_key(reference)) if reference.is_usable else None
        if cached:
            verdicts[position] = _from_cache(reference, cached)
            tick(reference.label[:40])
        else:
            pending.append((position, reference))

    try:
        async with Session(
            timeout=config.timeout,
            mailto=config.mailto,
            semantic_scholar_key=config.semantic_scholar_key,
            github_token=config.github_token,
            openalex_key=config.openalex_key,
        ) as session:
            run = _Run(session, _select_sources(config), config, client,
                       asyncio.Semaphore(config.search_concurrency), report)
            _start_batched_sources(session, config, [r for _, r in pending])

            async def one(reference: Reference) -> Verdict:
                async with limiter:
                    verdict = await _verify_one(reference, run)
                if verdict.status not in (Status.ERROR, Status.UNPARSED):
                    _remember(store, _cache_key(reference), verdict)
                tick(reference.label[:40])
                return verdict

            tasks = {asyncio.create_task(one(r)): position for position, r in pending}
            if tasks:
                finished, unfinished = await asyncio.wait(tasks, timeout=config.total_timeout)
                for task in unfinished:
                    task.cancel()
                for task in finished:
                    position = tasks[task]
                    try:
                        verdicts[position] = task.result()
                    except Exception as exc:          # one reference must not sink the run
                        verdicts[position] = Verdict(reference=representatives[position],
                                                     status=Status.ERROR, note=str(exc)[:200])
                report.timed_out = len(unfinished)
                for task in unfinished:
                    position = tasks[task]
                    verdicts[position] = Verdict(
                        reference=representatives[position], status=Status.ERROR,
                        note="not finished within the run's time budget")
            report.source_errors = dict(session.errors)
            if session.anthology is not None and session.anthology.error:
                report.source_errors["aclanthology.org"] = session.anthology.error
            report.throttled = dict(session.throttled)
            if session.anthology is not None:
                session.anthology.close()
    finally:
        report.cache_hits = store.hits
        store.close()

    ordered = [verdicts[position] for position in range(len(representatives))]
    report.verdicts = _expand(ordered, groups)
    report.elapsed = time.monotonic() - started
    return report


def _start_batched_sources(session: Session, config: RefCheckConfig, pending: list[Reference]) -> None:
    """Attach the batched sources and start them on every title at once.

    DBLP costs the same for one title as for a hundred, and arXiv asks for three
    seconds between requests, so the whole bibliography goes out up front and
    each reference later picks up its share.
    """
    titles = [r.title for r in pending if r.title and r.is_usable
              and (r.kind.is_academic or not r.url)]
    wanted = {s.name for s in _select_sources(config)}
    if config.use_anthology and "acl_anthology" in wanted:
        path = config.anthology_path
        session.anthology = Anthology(path, config.anthology_max_age_days)
        # Start the (at most weekly) download now, alongside everything else.
        session.background.append(asyncio.create_task(session.anthology.ready(session)))
    if config.use_arxiv and "arxiv" in wanted:
        session.arxiv = Arxiv(session)
        session.arxiv.prefetch(titles)
    if config.use_dblp and "dblp" in wanted:
        session.dblp = Dblp(session)
        session.dblp.prefetch(titles)


async def _identifiers(reference: Reference, run: _Run,
                       checked: list[str]) -> tuple[list[Candidate], list[Discrepancy]]:
    """Resolve every identifier the citation prints. Each must lead to this work."""
    session, threshold = run.session, run.config.policy.review_title
    found: list[Candidate] = []
    problems: list[Discrepancy] = []

    if reference.doi:
        checked.append("doi")
        lookup = await resolve_doi(session, reference.doi)
        record = lookup.record
        if record is not None and title_agrees(reference, record.title, threshold):
            found.append(record)
        elif record is not None:
            problems.append(Discrepancy(
                "doi", f"DOI {reference.doi} belongs to a different work: {_cite(record)}",
                source=record.source))
        elif lookup.exists is False:
            problems.append(Discrepancy(
                "doi", f"DOI {reference.doi} is not registered; doi.org has no record of it",
                source="doi.org"))

    if reference.arxiv_id and session.arxiv is not None:
        checked.append("arxiv_id")
        try:
            records: list[Candidate] | None = await session.arxiv.by_id(reference.arxiv_id)
        except LookupError:
            records = None
        if records:
            if title_agrees(reference, records[0].title, threshold):
                found.extend(records)
            else:
                problems.append(Discrepancy(
                    "arxiv", f"arXiv:{reference.arxiv_id} is a different paper: {_cite(records[0])}",
                    source="arxiv"))
        elif records is not None:
            problems.append(Discrepancy(
                "arxiv", f"arXiv:{reference.arxiv_id} does not exist", source="arxiv"))

    ident = anthology_id(url=reference.url)
    if ident and ident != anthology_id(doi=reference.doi) and session.anthology is not None \
            and await session.anthology.ready(session):
        checked.append("acl_anthology_id")
        record = await session.anthology.fetch(session, ident)
        if record is not None and title_agrees(reference, record.title, threshold):
            found.append(record)
        elif record is not None:
            problems.append(Discrepancy(
                "url", f"the ACL Anthology link ({ident}) is a different paper: {_cite(record)}",
                source="acl_anthology"))
    return found, problems


def _cite(record: Candidate) -> str:
    lead = record.authors[0] if record.authors else "unknown authors"
    more = " et al." if len(record.authors) > 1 else ""
    return f"“{record.title}” ({lead}{more}, {record.year or 'n.d.'})"


_OPENREVIEW_VENUES = re.compile(r"(?i)\b(?:iclr|neurips|nips|tmlr|colm|icml|openreview)\b|"
                                r"learning representations|neural information processing|"
                                r"conference on language modeling|transactions on machine learning")


def _needs_more(reference: Reference, accepted: list[Candidate]) -> bool:
    """Whether tier 2 could still add something tier 1 did not."""
    if not accepted:
        return True
    published = [r for r in accepted if not r.is_preprint]
    if cites_preprint(reference) or reference.venue:
        if not published:
            return True           # the cited venue, or a published version, is still unconfirmed
    if reference.pages and not any(r.pages for r in published):
        return True
    return bool(compare_authors(reference.authors, [r.authors for r in accepted if r.authors],
                                truncated=reference.truncated_authors))


def _can_search(run: _Run) -> bool:
    return run.config.use_web_search and run.client is not None and run.client.available


def _unpublished(reference: Reference, accepted: list[Candidate]) -> bool:
    """Only preprints were found: either the cited venue, or a published version, is unknown."""
    return all(r.is_preprint for r in accepted) and (
        _venue_unconfirmed(reference, accepted) or cites_preprint(reference))


def _venue_unconfirmed(reference: Reference, accepted: list[Candidate]) -> bool:
    """A citation names a venue, but every record found is a preprint."""
    return bool(reference.venue) and not cites_preprint(reference) and all(
        r.is_preprint for r in accepted)


def _relevant(spec: SourceSpec, reference: Reference, accepted: list[Candidate]) -> bool:
    """OpenReview allows twenty requests a minute, so it is asked only what it can answer:
    a paper nothing else confirmed, or one cited at a venue it hosts and not yet confirmed there."""
    if spec.name == "openreview" and accepted:
        return bool(_OPENREVIEW_VENUES.search(reference.venue or "")) and all(
            r.is_preprint for r in accepted)
    return True


async def _verify_one(reference: Reference, run: _Run) -> Verdict:
    reference.classify_kind()
    policy = run.config.policy

    if not reference.is_usable:
        return Verdict(reference=reference, status=Status.UNPARSED,
                       note="not enough of this entry could be parsed to look it up")

    # A repository, model card or blog post is checked against what it is. A
    # link to arXiv, the ACL Anthology, OpenReview or a DOI is a paper, however
    # the parser classed it, and is confirmed at that source instead.
    if needs_web_route(reference) and reference.url and not scholarly_url(reference.url):
        resolution = await resolve_web(run.session, reference)
        if resolution is not None:
            return Verdict(
                reference=reference, status=Status.RESOLVED if resolution.ok else Status.UNRESOLVED,
                source=resolution.source, matched_title=resolution.title,
                score=1.0 if resolution.ok else 0.0, note=resolution.detail, url=resolution.url,
                checked_sources=(resolution.source,),
            )

    checked: list[str] = []
    candidates, problems = await _identifiers(reference, run, checked)
    accepted = [c for c in candidates if policy.judge(reference, c)[0] == "accept"]
    best: Candidate | None = None
    outcome, score, authors = "reject", 0.0, -1.0

    for tier in (1, 2):
        if tier == 2 and not _needs_more(reference, accepted):
            break
        applicable = [s for s in run.sources if s.tier == tier and s.applies(reference)
                      and _relevant(s, reference, accepted)]
        if not applicable:
            continue
        results = await asyncio.gather(
            *(s.fetch(run.session, reference) for s in applicable), return_exceptions=True
        )
        for spec, result in zip(applicable, results, strict=False):
            checked.append(spec.name if isinstance(result, list) else f"{spec.name} (failed)")
            if isinstance(result, list):
                candidates.extend(result)
        accepted = [c for c in candidates if policy.judge(reference, c)[0] == "accept"]
        best, outcome, score, authors = _best(reference, candidates, policy)

    if accepted and _unpublished(reference, accepted) and run.session.dblp is not None:
        # The published version may carry a different title. DBLP lists it
        # under the same first author, in about the same year.
        first = next((r.authors[0] for r in accepted if r.authors), None) or \
            (reference.authors[0] if reference.authors else None)
        year = reference.year or next((r.year for r in accepted if r.year), None)
        if first and year:
            checked.append("dblp_author")
            try:
                found = await run.session.dblp.by_author(first, year)
            except LookupError:
                found = []
            accepted += [c for c in found if policy.judge(reference, c)[0] == "accept"]
    if accepted and _venue_unconfirmed(reference, accepted) and _can_search(run):
        # The work is real, but only its preprint was found and the citation
        # names a venue. Ask where the published version is, and confirm it.
        async with run.search_limiter:
            run.report.searched += 1
            venue_lead = await _web_search(reference, None, run.client, published_in=reference.venue)
        if not venue_lead.error:
            checked.append("web_search")
            found = await _confirm(reference, venue_lead, run)
            accepted += [c for c in found if policy.judge(reference, c)[0] == "accept"]
    if accepted:
        return _conclude(reference, accepted, problems, checked, policy)

    # Discovery: a web search may know where the record is. What it returns is
    # a lead, confirmed against the key source it names before it counts.
    lead: _Lead | None = None
    if _can_search(run):
        async with run.search_limiter:
            run.report.searched += 1
            lead = await _web_search(reference, best, run.client)
        if lead.error:
            return Verdict(reference=reference, status=Status.ERROR, note=lead.error,
                           checked_sources=tuple(checked), discrepancies=problems)
        confirmed = await _confirm(reference, lead, run)
        checked.append("web_search")
        accepted = [c for c in confirmed if policy.judge(reference, c)[0] == "accept"]
        if accepted:
            return _conclude(reference, accepted, problems, checked, policy, via_search=True)
        if lead.found and reference.kind.has_web_route:
            # A blog post, repository or standard: the page existing is the confirmation.
            # Anything that may be a paper, including an entry of unknown kind, needs a key source.
            for url in lead.urls[:2]:
                resolution = await liveness(run.session, url)
                if resolution is not None and resolution.ok:
                    return Verdict(
                        reference=reference, status=Status.RESOLVED, source="web_search",
                        matched_title=lead.title, score=1.0, url=resolution.url,
                        note=f"located by web search; {resolution.detail}. {lead.note}",
                        checked_sources=tuple(checked), discrepancies=problems)
        if lead.found:
            where = f" at {lead.urls[0]}" if lead.urls else ""
            return Verdict(
                reference=reference, status=Status.UNCONFIRMED, source="web_search",
                matched_title=lead.title, url=lead.urls[0] if lead.urls else None,
                note=f"web search reports it{where}, but no key source confirms it. {lead.note}",
                checked_sources=tuple(checked), discrepancies=problems)

    if outcome == "review" and best is not None:
        return Verdict(
            reference=reference, status=Status.AUTHOR_MISMATCH, source=best.source,
            matched_title=best.title, score=score, author_score=authors,
            note="a close title exists but the authors or year do not line up",
            url=best.url, checked_sources=tuple(checked), discrepancies=problems)

    if reference.kind.has_web_route and not problems:
        # A blog post or repository that no database indexes is not suspicious;
        # it was never going to be in one. An entry of unknown kind may be a
        # paper, so it is not excused this way.
        return Verdict(
            reference=reference, status=Status.UNINDEXED, checked_sources=tuple(checked),
            note=f"cited as {reference.kind.value}; bibliographic databases do not index these")

    note = f"no key source has it (checked {', '.join(checked) or 'nothing'})"
    if lead is not None:
        note += f"; web search did not find it either. {lead.note}"
    return Verdict(reference=reference, status=Status.NOT_FOUND, score=score,
                   checked_sources=tuple(checked), note=note, discrepancies=problems)


def _conclude(reference: Reference, accepted: list[Candidate], problems: list[Discrepancy],
              checked: list[str], policy: MatchPolicy, via_search: bool = False) -> Verdict:
    """The verdict for a confirmed work: verified, or confirmed with discrepancies."""
    records: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for record in accepted:
        key = (record.source, record.record_id or record.doi or record.title.lower())
        if key not in seen:
            seen.add(key)
            records.append(record)

    discrepancies = list(problems)
    preprint = cites_preprint(reference)
    correct_doi = next((r.doi for r in records if r.doi and r.is_preprint == preprint), None)
    for index, problem in enumerate(discrepancies):
        if problem.field == "doi" and correct_doi:
            discrepancies[index] = _replace(problem, detail=f"{problem.detail}; this work's DOI is "
                                                            f"{correct_doi}")
    discrepancies.extend(compare(reference, records))

    primary = max(records, key=lambda r: (r.exact_id, not r.is_preprint, bool(r.pages)))
    _, score, authors = policy.judge(reference, primary)
    sources = sorted({r.source for r in records})
    how = "found by web search and confirmed" if via_search else "confirmed"
    return Verdict(
        reference=reference,
        status=Status.DETAILS_MISMATCH if discrepancies else Status.VERIFIED,
        source=primary.source, matched_title=primary.title, score=score, author_score=authors,
        note=f"{how} in {', '.join(sources)}", url=primary.url,
        checked_sources=tuple(checked), discrepancies=discrepancies,
    )


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


@dataclass(slots=True)
class _Lead:
    """What a web search reported. A lead, not a verdict."""

    found: bool = False
    title: str | None = None
    urls: list[str] = field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    note: str = ""
    error: str | None = None


async def _web_search(reference: Reference, near: Candidate | None, client: AsyncLLMClient,
                      published_in: str | None = None) -> _Lead:
    """Ask the model where the record of this citation is."""
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

    task = "Find the official record of the work cited below. Search for it."
    if published_in:
        task = (f"The work cited below exists as a preprint. The citation says it was published in "
                f"{published_in}. Find the record of that published version — the proceedings, "
                "journal or OpenReview page — if it exists. If it was not published there, say so "
                "and set `found` false.")
    prompt = (
        task + "\n\n"
        + "\n".join(described)
        + f"\n\nFull citation as printed:\n{reference.raw[:600]}\n\n"
        "Set `found` true only if the search results show a real work matching this citation. "
        "Then say where its record is: `doi` if it has one, `arxiv_id` if it is on arXiv, and "
        "`record_url` for the page that lists it — the publisher or proceedings page, the ACL "
        "Anthology or OpenReview page, or, for a blog post, repository, model card or standard, "
        "the page itself. Prefer the published version's record over a preprint. Copy identifiers "
        "exactly as the results show them; never construct one. Leave a field null when the "
        "results do not show it.\n\n"
        "Set `found` false when the search did not surface the work."
    )
    try:
        data, citations = await client.web_search(
            SEARCH_SYSTEM, prompt, schema=_SEARCH_SCHEMA, name="reference_check",
            allowed_domains=SEARCH_DOMAINS, context_size="low",
        )
    except LLMError as exc:
        return _Lead(error=str(exc)[:200])

    urls = [u for u in [data.get("record_url"), *citations] if isinstance(u, str) and u.startswith("http")]
    urls = [re.sub(r"[?&]utm_source=[^&]*", "", u) for u in dict.fromkeys(urls)]
    return _Lead(
        found=bool(data.get("found")),
        title=data.get("matched_title"),
        urls=urls,
        doi=data.get("doi") or None,
        arxiv_id=data.get("arxiv_id") or None,
        note=str(data.get("note", ""))[:300],
    )


async def _confirm(reference: Reference, lead: _Lead, run: _Run) -> list[Candidate]:
    """Fetch the records a web-search lead points at, from the key sources themselves."""
    session = run.session
    found: list[Candidate] = []
    if lead.doi:
        lookup = await resolve_doi(session, lead.doi)
        if lookup.record is not None:
            found.append(_replace(lookup.record, exact_id=False))
    if lead.arxiv_id and session.arxiv is not None:
        try:
            found.extend(_replace(c, exact_id=False) for c in await session.arxiv.by_id(lead.arxiv_id))
        except LookupError:
            pass
    for url in lead.urls[:4]:
        try:
            records = await confirm_url(session, url, reference)
        except LookupError:
            continue
        # Found by a search, not printed in the citation: judged on title and
        # authors like any search result.
        found.extend(_replace(c, exact_id=False) for c in records)
    return found


def _remember(store: Store, key: str, verdict: Verdict) -> None:
    positive = verdict.status in {Status.VERIFIED, Status.RESOLVED, Status.DETAILS_MISMATCH}
    store.put(key, {
        "status": verdict.status.value,
        "source": verdict.source,
        "matched_title": verdict.matched_title,
        "score": verdict.score,
        "author_score": verdict.author_score,
        "note": verdict.note,
        "url": verdict.url,
        "checked_sources": list(verdict.checked_sources),
        "discrepancies": [d.to_dict() for d in verdict.discrepancies],
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
        discrepancies=[Discrepancy(str(d.get("field")), str(d.get("detail")), d.get("source"))
                       for d in data.get("discrepancies") or [] if isinstance(d, dict)],
    )
