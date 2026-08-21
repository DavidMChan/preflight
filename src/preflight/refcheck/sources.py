"""Asynchronous lookups against bibliographic databases and the open web.

The design difference from a fallthrough checker: every source for every
reference is issued concurrently under a per-host rate limiter, with a short
timeout, and the first confident match wins. Nothing waits on a database it did
not need, and one slow host cannot hold up the run.

arXiv's search API is deliberately absent from the fast path — it enforces a
three-second gap between requests, which is precisely the thing that makes a
serial checker take minutes. arXiv preprints are indexed by OpenAlex anyway, and
a direct arXiv id lookup is still done when the reference carries one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .core import Candidate, Kind, Reference

USER_AGENT = "preflight/0.1 (academic pre-flight checker)"


class RateLimiter:
    """An async token bucket, one per host, that slows down when refused.

    A fixed rate is a guess about what a service will tolerate. When the guess is
    wrong the choice is to keep being refused, or to adapt — so a throttle halves
    the rate and a clean run slowly earns it back. That keeps a host in play
    instead of forcing every reference it would have answered into a much more
    expensive tier.
    """

    def __init__(self, per_second: float, burst: int | None = None) -> None:
        self.initial = max(per_second, 0.01)
        self.floor = max(per_second / 8.0, 0.2)
        self.per_second = max(per_second, 0.01)
        self.capacity = burst if burst is not None else max(1, int(per_second))
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    def penalise(self) -> None:
        """Called on a 429/503: halve the rate, down to a floor."""
        self.per_second = max(self.floor, self.per_second / 2.0)

    def reward(self) -> None:
        """Called on a clean response: drift back toward the original rate."""
        if self.per_second < self.initial:
            self.per_second = min(self.initial, self.per_second * 1.1)

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._updated) * self.per_second)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.per_second)


@dataclass(slots=True)
class SourceSpec:
    """One lookup route: what it is called, how fast it may be called, what it covers."""

    name: str
    fetch: Callable[[Session, Reference], Awaitable[list[Candidate]]]
    rate: float = 8.0
    kinds: tuple[Kind, ...] = ()      # empty means "any academic kind"
    optional: bool = False
    # 1 = fast and generously rate-limited, tried for every reference.
    # 2 = slow or tightly rate-limited, tried only for references tier 1 could
    # not settle. Firing a 0.5 req/s source at every reference serialises the
    # whole run behind it, which is the failure this checker exists to fix.
    tier: int = 1

    def applies(self, reference: Reference) -> bool:
        if self.kinds:
            return reference.kind in self.kinds
        return reference.kind.is_academic or reference.kind is Kind.UNKNOWN


class Session:
    """Shared HTTP client plus per-host rate limiters."""

    def __init__(
        self,
        timeout: float = 6.0,
        mailto: str | None = None,
        semantic_scholar_key: str | None = None,
        github_token: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.mailto = mailto
        self.semantic_scholar_key = semantic_scholar_key
        self.github_token = github_token
        self._client: httpx.AsyncClient | None = None
        self._limiters: dict[str, RateLimiter] = {}
        self.errors: dict[str, str] = {}
        self.throttled: dict[str, int] = {}
        self.disabled: set[str] = set()
        # Hosts adapt their rate when refused; a host is only abandoned if it
        # refuses persistently, which means it is not going to answer today.
        self.throttle_budget = 25

    async def __aenter__(self) -> Session:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def limiter(self, host: str, rate: float) -> RateLimiter:
        if host not in self._limiters:
            self._limiters[host] = RateLimiter(rate)
        return self._limiters[host]

    async def get(self, url: str, *, rate: float = 8.0, headers: dict[str, str] | None = None,
                  method: str = "GET", retries: int = 1) -> httpx.Response | None:
        """One request, rate limited per host, with a short backoff on a throttle.

        A 429 that is not retried is worse than a slow request: the reference
        falls through to the expensive tiers and can end up reported as missing
        when all that actually happened was a throttle.
        """
        if self._client is None:  # pragma: no cover - misuse
            raise RuntimeError("Session must be used as an async context manager")
        host = urlparse(url).netloc
        if host in self.disabled:
            return None
        for attempt in range(retries + 1):
            await self.limiter(host, rate).acquire()
            try:
                response = await self._client.request(method, url, headers=headers)
            except (TimeoutError, httpx.HTTPError, ValueError, UnicodeError) as exc:
                if attempt < retries:
                    await asyncio.sleep(0.3)
                    continue
                self.errors[host] = f"{type(exc).__name__}: {exc}"
                return None
            if response.status_code in (429, 503):
                self.throttled[host] = self.throttled.get(host, 0) + 1
                self.limiter(host, rate).penalise()
                if self.throttled[host] >= self.throttle_budget:
                    self.disabled.add(host)
                    self.errors[host] = (
                        f"throttling ({response.status_code}); dropped after "
                        f"{self.throttled[host]} refusals"
                    )
                    return None
                if attempt < retries:
                    await asyncio.sleep(min(_retry_after(response) or 0.8, 1.5))
                    continue
                self.errors[host] = f"rate limited ({response.status_code})"
                return None
            self.limiter(host, rate).reward()
            return response
        return None

    async def json(self, url: str, *, rate: float = 8.0,
                   headers: dict[str, str] | None = None) -> Any | None:
        response = await self.get(url, rate=rate, headers=headers)
        if response is None or response.status_code >= 400:
            return None
        try:
            return response.json()
        except ValueError:
            return None


def _retry_after(response: httpx.Response) -> float | None:
    """Honour a server's own backoff hint when it gives one."""
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _year(value: Any) -> int | None:
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1500 < year < 2100 else None


# ---------------------------------------------------------------------------
# Academic sources
# ---------------------------------------------------------------------------


async def crossref_doi(session: Session, ref: Reference) -> list[Candidate]:
    """Resolve a DOI directly. Definitive, and one request."""
    if not ref.doi:
        return []
    data = await session.json(f"https://api.crossref.org/works/{quote(ref.doi, safe='')}", rate=20)
    item = (data or {}).get("message")
    if not isinstance(item, dict):
        return []
    return [_crossref_candidate(item, exact=True)]


def _crossref_candidate(item: dict[str, Any], exact: bool = False) -> Candidate:
    titles = item.get("title") or []
    authors = [
        " ".join(x for x in (a.get("given"), a.get("family")) if x)
        for a in (item.get("author") or []) if isinstance(a, dict)
    ]
    issued = ((item.get("issued") or {}).get("date-parts") or [[None]])[0]
    container = item.get("container-title") or []
    return Candidate(
        source="crossref",
        title=titles[0] if titles else "",
        authors=[a for a in authors if a],
        year=_year(issued[0]) if issued else None,
        venue=container[0] if container else None,
        doi=item.get("DOI"),
        url=item.get("URL"),
        exact_id=exact,
    )


async def crossref_search(session: Session, ref: Reference) -> list[Candidate]:
    query = ref.title or ref.raw[:200]
    url = (
        "https://api.crossref.org/works?rows=4&select=title,author,issued,DOI,container-title,URL"
        f"&query.bibliographic={quote(query)}"
    )
    if session.mailto:
        url += f"&mailto={quote(session.mailto)}"
    data = await session.json(url, rate=12)
    items = ((data or {}).get("message") or {}).get("items") or []
    return [_crossref_candidate(i) for i in items if isinstance(i, dict)]


async def openalex(session: Session, ref: Reference) -> list[Candidate]:
    """OpenAlex: broad coverage, fast, and generous limits. The workhorse."""
    mail = f"&mailto={quote(session.mailto)}" if session.mailto else ""
    if ref.doi:
        url = f"https://api.openalex.org/works/https://doi.org/{quote(ref.doi, safe='')}?{mail.lstrip('&')}"
        data = await session.json(url, rate=10)
        if isinstance(data, dict) and data.get("id"):
            return [_openalex_candidate(data, exact=True)]
    query = ref.title or ref.raw[:200]
    url = (
        "https://api.openalex.org/works?per-page=4&select=id,doi,title,publication_year,"
        f"authorships,primary_location&filter=title.search:{quote(query)}{mail}"
    )
    data = await session.json(url, rate=10)
    results = (data or {}).get("results") or []
    return [_openalex_candidate(r) for r in results if isinstance(r, dict)]


def _openalex_candidate(item: dict[str, Any], exact: bool = False) -> Candidate:
    authors = [
        ((a.get("author") or {}).get("display_name") or "")
        for a in (item.get("authorships") or []) if isinstance(a, dict)
    ]
    location = item.get("primary_location") or {}
    source = (location.get("source") or {}) if isinstance(location, dict) else {}
    return Candidate(
        source="openalex",
        title=item.get("title") or "",
        authors=[a for a in authors if a],
        year=_year(item.get("publication_year")),
        venue=source.get("display_name"),
        doi=(item.get("doi") or "").replace("https://doi.org/", "") or None,
        url=item.get("id"),
        exact_id=exact,
    )


async def dblp(session: Session, ref: Reference) -> list[Candidate]:
    """DBLP: authoritative for computer science, but rate-limits hard."""
    query = ref.title or ref.raw[:150]
    url = f"https://dblp.org/search/publ/api?q={quote(query)}&format=json&h=4"
    data = await session.json(url, rate=2)
    hits = (((data or {}).get("result") or {}).get("hits") or {}).get("hit") or []
    out: list[Candidate] = []
    for hit in hits:
        info = (hit or {}).get("info") or {}
        raw_authors = ((info.get("authors") or {}).get("author")) or []
        if isinstance(raw_authors, dict):
            raw_authors = [raw_authors]
        authors = [a.get("text", "") if isinstance(a, dict) else str(a) for a in raw_authors]
        out.append(
            Candidate(
                source="dblp",
                title=info.get("title", "").rstrip("."),
                authors=[a for a in authors if a],
                year=_year(info.get("year")),
                venue=info.get("venue"),
                doi=info.get("doi"),
                url=info.get("ee") or info.get("url"),
            )
        )
    return out


async def semantic_scholar(session: Session, ref: Reference) -> list[Candidate]:
    """Semantic Scholar: good coverage, but throttles hard without a key."""
    query = ref.title or ref.raw[:150]
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote(query)}&limit=4&fields=title,authors,year,venue,externalIds,url"
    )
    headers = {"x-api-key": session.semantic_scholar_key} if session.semantic_scholar_key else None
    data = await session.json(url, rate=1 if not session.semantic_scholar_key else 8, headers=headers)
    out: list[Candidate] = []
    for item in (data or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        external = item.get("externalIds") or {}
        out.append(
            Candidate(
                source="semantic_scholar",
                title=item.get("title") or "",
                authors=[(a or {}).get("name", "") for a in (item.get("authors") or [])],
                year=_year(item.get("year")),
                venue=item.get("venue"),
                doi=external.get("DOI"),
                url=item.get("url"),
            )
        )
    return out


async def arxiv_by_id(session: Session, ref: Reference) -> list[Candidate]:
    """A direct arXiv id lookup only — never a search, which is what is slow."""
    if not ref.arxiv_id:
        return []
    data = await session.get(
        f"https://export.arxiv.org/api/query?id_list={quote(ref.arxiv_id)}", rate=0.5
    )
    if data is None or data.status_code >= 400 or "<entry>" not in data.text:
        return []
    body = data.text
    title = _between(body, "<title>", "</title>", skip=1)
    authors = _all_between(body, "<name>", "</name>")
    published = _between(body, "<published>", "</published>")
    return [
        Candidate(
            source="arxiv",
            title=" ".join((title or "").split()),
            authors=authors,
            year=_year(published),
            venue="arXiv",
            url=f"https://arxiv.org/abs/{ref.arxiv_id}",
            exact_id=True,
        )
    ]


def _between(text: str, start: str, end: str, skip: int = 0) -> str | None:
    pos = 0
    for _ in range(skip + 1):
        pos = text.find(start, pos)
        if pos < 0:
            return None
        pos += len(start)
    stop = text.find(end, pos)
    return text[pos:stop] if stop > 0 else None


def _all_between(text: str, start: str, end: str) -> list[str]:
    out: list[str] = []
    pos = 0
    while True:
        pos = text.find(start, pos)
        if pos < 0:
            break
        pos += len(start)
        stop = text.find(end, pos)
        if stop < 0:
            break
        out.append(" ".join(text[pos:stop].split()))
        pos = stop
    return out


ACADEMIC_SOURCES: tuple[SourceSpec, ...] = (
    # Tier 1 — the default path. High limits, broad coverage, and between them
    # they resolve the overwhelming majority of a bibliography. OpenAlex indexes
    # arXiv, so preprints are covered here too.
    SourceSpec("crossref_doi", crossref_doi, rate=10.0, tier=1),
    SourceSpec("openalex", openalex, rate=8.0, tier=1),
    SourceSpec("crossref", crossref_search, rate=8.0, tier=1),
    # Tier 2 — off by default. Each of these is rate-limited to the point where
    # firing it at a whole bibliography serialises the run: arXiv asks for a
    # three-second gap, and Semantic Scholar throttles hard without a key. They
    # add little over tier 1 plus the web-search tier, so they are opt-in via
    # `refcheck.sources`. Adding them buys coverage at a real cost in wall clock.
    SourceSpec("dblp", dblp, rate=3.0, tier=2, optional=True),
    SourceSpec("arxiv_id", arxiv_by_id, rate=0.5, tier=2, optional=True),
    SourceSpec("semantic_scholar", semantic_scholar, rate=1.0, tier=2, optional=True),
)
