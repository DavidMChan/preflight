"""Asynchronous lookups against the key sources a citation is confirmed by.

Every source for every reference is issued concurrently under a per-host rate
limiter, so nothing waits on a host it did not need and one slow host cannot
hold up the run. Hosts that ask for requests to be spaced out — arXiv and
DBLP — are reached through batchers (see :mod:`.batch`) that put many lookups
in one request instead.

Every request identifies the tool and a contact address in its User-Agent, and
honours the rate each host asks for. Pages fetched from a publisher's site,
rather than an API, are fetched only where that site's robots.txt allows it,
and no faster than its crawl delay.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
import urllib.robotparser
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urlparse

import httpx

from .anthology import anthology_id
from .core import Candidate, Kind, Reference, normalise_doi

if TYPE_CHECKING:
    from .anthology import Anthology
    from .arxiv import Arxiv
    from .dblp import Dblp

TOOL = "preflight/0.2 (+https://github.com/DavidMChan/preflight)"

#: Retries a throttled request gets, and the most it will sleep in total for
#: them. A reference whose lookup gives up on a 429 falls through to the next
#: tier, and can end up reported as missing when the host was only busy.
THROTTLE_RETRIES = 4
THROTTLE_WAIT_BUDGET = 90.0
ROBOTS_TOKEN = "preflight"


def user_agent(mailto: str | None = None) -> str:
    """Who is asking, and how to reach them — what every API's etiquette asks for."""
    return f"{TOOL[:-1]}; mailto:{mailto})" if mailto else TOOL


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
        self.floor = min(self.initial, max(per_second / 8.0, 0.2))
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

    def lower(self, per_second: float) -> None:
        """The host published its limit: never exceed it again this run."""
        if per_second < self.initial:
            self.initial = self.per_second = max(per_second, 0.01)
            self.floor = min(self.floor, self.initial)
            self.capacity = 1
            self._tokens = min(self._tokens, 1.0)


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


class Slots:
    """How many requests a host may have in flight at once. The host can lower it."""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self.active = 0
        self._condition = asyncio.Condition()

    async def __aenter__(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def __aexit__(self, *exc: object) -> None:
        async with self._condition:
            self.active -= 1
            self._condition.notify_all()


@dataclass(slots=True)
class SourceSpec:
    """One lookup route: what it is called, how fast it may be called, what it covers."""

    name: str
    fetch: Callable[[Session, Reference], Awaitable[list[Candidate]]]
    rate: float = 8.0
    kinds: tuple[Kind, ...] = ()      # empty means "anything that may be a paper"
    optional: bool = False
    # 1 = consulted for every reference. 2 = consulted only when tier 1 could
    # not confirm the reference, or confirmed a different version of it.
    tier: int = 1

    def applies(self, reference: Reference) -> bool:
        if self.kinds:
            return reference.kind in self.kinds
        # An entry the parser called a blog post, report or web page may well be
        # a paper if it carries no URL, or links to arXiv, the ACL Anthology,
        # OpenReview or a DOI. Only a key source can say it is not.
        return (reference.kind.is_academic or reference.kind is Kind.UNKNOWN
                or not reference.url or scholarly_url(reference.url))


class Session:
    """Shared HTTP client, per-host rate limiters, robots.txt, and the batched sources."""

    def __init__(
        self,
        timeout: float = 6.0,
        mailto: str | None = None,
        semantic_scholar_key: str | None = None,
        github_token: str | None = None,
        openalex_key: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.mailto = mailto
        self.semantic_scholar_key = semantic_scholar_key
        self.github_token = github_token
        self.openalex_key = openalex_key
        self._client: httpx.AsyncClient | None = None
        self._limiters: dict[str, RateLimiter] = {}
        self._slots: dict[str, Slots] = {}
        self._robots: dict[str, tuple[urllib.robotparser.RobotFileParser | None, float | None]] = {}
        self._robots_lock = asyncio.Lock()
        self.errors: dict[str, str] = {}
        self.throttled: dict[str, int] = {}
        self.disabled: set[str] = set()
        # Hosts adapt their rate when refused; a host is only abandoned if it
        # refuses persistently -- this many times in a row, with no answer in
        # between -- which means it is not going to answer today. A burst of
        # 429s from concurrent lookups, followed by answers, is just a busy host.
        self.throttle_budget = 25
        self._refusals: dict[str, int] = {}
        self.anthology: Anthology | None = None
        self.arxiv: Arxiv | None = None
        self.dblp: Dblp | None = None
        self.background: list[asyncio.Task[Any]] = []

    async def __aenter__(self) -> Session:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            follow_redirects=True,
            headers={"User-Agent": user_agent(self.mailto)},
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        for batched in (self.arxiv, self.dblp):
            if batched is not None:
                batched.close()
        for task in self.background:
            task.cancel()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def limiter(self, host: str, rate: float) -> RateLimiter:
        if host not in self._limiters:
            self._limiters[host] = RateLimiter(rate)
        return self._limiters[host]

    def slots(self, host: str, limit: int = 6) -> Slots:
        if host not in self._slots:
            self._slots[host] = Slots(limit)
        return self._slots[host]

    def _learn(self, host: str, response: httpx.Response) -> None:
        """Adopt the limits a host publishes in its response headers.

        CrossRef sends ``x-rate-limit-limit``/``-interval`` and
        ``x-concurrency-limit`` (one request at a time for anonymous clients);
        OpenReview sends ``ratelimit-policy: 20;w=60``. Staying inside them is
        both politer and faster than being throttled.
        """
        headers = response.headers
        rate: float | None = None
        with suppress(ValueError, ZeroDivisionError):
            if headers.get("x-rate-limit-limit") and headers.get("x-rate-limit-interval"):
                interval = float(headers["x-rate-limit-interval"].strip().rstrip("s") or 1)
                rate = float(headers["x-rate-limit-limit"]) / interval
            elif headers.get("ratelimit-policy"):
                match = re.match(r"\s*(\d+)\s*;\s*w=(\d+)", headers["ratelimit-policy"])
                if match:
                    rate = int(match.group(1)) / int(match.group(2))
        if rate:
            self.limiter(host, rate).lower(rate * 0.9)
        with suppress(ValueError):
            concurrency = int(headers.get("x-concurrency-limit") or 0)
            if concurrency:
                self.slots(host).limit = concurrency

    async def allowed(self, url: str) -> tuple[bool, float | None]:
        """Whether robots.txt lets this tool fetch ``url``, and the host's crawl delay.

        Used for pages, not APIs: an API's terms of use govern it instead. A
        robots.txt that cannot be read at all counts as a refusal.
        """
        parts = urlparse(url)
        host = parts.netloc
        async with self._robots_lock:
            if host not in self._robots:
                parser: urllib.robotparser.RobotFileParser | None = urllib.robotparser.RobotFileParser()
                response = await self.get(f"{parts.scheme}://{host}/robots.txt", rate=1.0, retries=0)
                if response is None or response.status_code >= 500:
                    parser = None
                elif response.status_code in (401, 403):
                    parser.disallow_all = True
                elif response.status_code >= 400:
                    parser.allow_all = True
                else:
                    parser.parse(response.text.splitlines())
                delay = parser.crawl_delay(ROBOTS_TOKEN) if parser is not None else None
                self._robots[host] = (parser, float(delay) if delay else None)
        parser, delay = self._robots[host]
        if parser is None:
            return False, None
        return parser.can_fetch(ROBOTS_TOKEN, url), delay

    async def get(self, url: str, *, rate: float = 8.0, headers: dict[str, str] | None = None,
                  method: str = "GET", retries: int = 1, timeout: float | None = None,
                  data: dict[str, str] | None = None, concurrency: int = 6,
                  throttle: tuple[int, ...] = (429, 503),
                  throttle_retries: int = THROTTLE_RETRIES) -> httpx.Response | None:
        """One request, rate limited per host, retried with backoff when throttled.

        A 429 that is not retried is worse than a slow request: the reference
        falls through to the expensive tiers and can end up reported as missing
        when all that actually happened was a throttle. So a throttle gets its
        own retries, separate from ``retries`` for transport errors, waiting as
        long as the host asks or else backing off exponentially with jitter, so
        that lookups refused together do not all return together. A host that
        asks us to come back more than a minute later is taken out of the run.
        """
        if self._client is None:  # pragma: no cover - misuse
            raise RuntimeError("Session must be used as an async context manager")
        host = urlparse(url).netloc
        if host in self.disabled:
            return None
        limiter = self.limiter(host, rate)
        failures = throttles = 0
        waited = 0.0
        while True:
            async with self.slots(host, concurrency):
                await limiter.acquire()
                try:
                    response = await self._client.request(
                        method, url, headers=headers, data=data,
                        timeout=httpx.Timeout(timeout) if timeout else httpx.USE_CLIENT_DEFAULT,
                    )
                except (TimeoutError, httpx.HTTPError, ValueError, UnicodeError) as exc:
                    if failures < retries:
                        failures += 1
                        await asyncio.sleep(0.3)
                        continue
                    self.errors[host] = f"{type(exc).__name__}: {exc}"
                    return None
            self._learn(host, response)
            if response.status_code not in throttle:
                self._refusals[host] = 0
                limiter.reward()
                return response

            self.throttled[host] = self.throttled.get(host, 0) + 1
            self._refusals[host] = self._refusals.get(host, 0) + 1
            limiter.penalise()
            wait = _retry_after(response)
            if (wait is not None and wait > 60) or self._refusals[host] >= self.throttle_budget:
                self.disabled.add(host)
                self.errors[host] = _refusal(host, response, self._refusals[host])
                return None
            pause = _backoff(throttles, wait, limiter.per_second)
            if throttles >= throttle_retries or waited + pause > THROTTLE_WAIT_BUDGET:
                self.errors[host] = f"rate limited ({response.status_code})"
                return None
            throttles += 1
            waited += pause
            await asyncio.sleep(pause)

    async def json(self, url: str, *, rate: float = 8.0, headers: dict[str, str] | None = None,
                   concurrency: int = 6) -> Any | None:
        response = await self.get(url, rate=rate, headers=headers, concurrency=concurrency)
        if response is not None and response.status_code == 403 and "Challenge" in response.text[:300]:
            host = urlparse(url).netloc
            self.errors[host] = "this endpoint now requires a browser challenge; not attempted further"
            self.disabled.add(host)
            return None
        if response is None or response.status_code >= 400:
            return None
        if "html" in response.headers.get("content-type", ""):
            host = urlparse(url).netloc
            self.errors[host] = "returned a web page instead of data (a bot challenge?)"
            return None
        try:
            return response.json()
        except ValueError:
            return None

    async def download(self, url: str, timeout: float = 60.0) -> bytes | None:
        response = await self.get(url, rate=1.0, retries=1, timeout=timeout)
        if response is None or response.status_code >= 400:
            return None
        return response.content


def _retry_after(response: httpx.Response) -> float | None:
    """Honour a server's own backoff hint: seconds, or an HTTP date to wait until."""
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, when.timestamp() - time.time())


def _backoff(attempt: int, retry_after: float | None, per_second: float) -> float:
    """How long to wait before retrying a throttled request.

    The host's own Retry-After when it gives one, plus a little jitter; otherwise
    an exponential step from one interval at the (already lowered) rate, capped
    at 30 seconds, drawn from its upper half so refused lookups spread out.
    """
    if retry_after is not None:
        return retry_after + random.uniform(0.0, 1.0)
    step = min(30.0, max(1.0, 1.0 / per_second) * 2.0 ** attempt)
    return random.uniform(step / 2.0, step)


def _refusal(host: str, response: httpx.Response, count: int) -> str:
    if host == "api.openalex.org":
        return ("OpenAlex refused: its free daily budget for this network is used up. Set "
                "OPENALEX_API_KEY (keys are free) to give this tool its own budget")
    wait = _retry_after(response)
    if wait is not None and wait > 60:
        return f"throttling ({response.status_code}); the host asked to wait {wait / 60:.0f} minutes"
    return f"throttling ({response.status_code}); dropped after {count} refusals"


def _year(value: Any) -> int | None:
    try:
        year = int(str(value)[:4])
    except (TypeError, ValueError):
        return None
    return year if 1500 < year < 2100 else None


def _date_year(value: Any) -> int | None:
    """CrossRef/CSL ``{"date-parts": [[2024, 8]]}`` to 2024."""
    parts = (value or {}).get("date-parts") if isinstance(value, dict) else None
    return _year(parts[0][0]) if parts and parts[0] else None


# ---------------------------------------------------------------------------
# DOIs: every DOI a citation prints is resolved
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DoiLookup:
    """What the DOI registry says about a DOI.

    ``exists`` is True or False only when a registry answered; None means the
    lookups failed, and that is never reported as a missing DOI.
    """

    exists: bool | None
    record: Candidate | None = None


def _crossref_candidate(item: dict[str, Any], exact: bool = False, source: str = "crossref") -> Candidate:
    titles = item.get("title") or []
    title = titles[0] if isinstance(titles, list) and titles else titles if isinstance(titles, str) else ""
    # CrossRef keeps "A Method for Automatic Evaluation" apart from "BLEU".
    subtitles = item.get("subtitle") or []
    subtitle = subtitles[0] if isinstance(subtitles, list) and subtitles else subtitles
    if isinstance(subtitle, str) and subtitle.strip() and subtitle.lower() not in str(title).lower():
        title = f"{title}: {subtitle}"
    authors = [
        a.get("literal") or " ".join(x for x in (a.get("given"), a.get("family")) if x)
        for a in (item.get("author") or []) if isinstance(a, dict)
    ]
    container = item.get("container-title") or []
    venue = container[0] if isinstance(container, list) and container else container or None
    years = tuple(y for y in (_date_year(item.get(k)) for k in
                              ("published-print", "published-online", "published", "issued")) if y)
    doi = normalise_doi(item.get("DOI"))
    return Candidate(
        source=source,
        title=" ".join(str(title).split()),
        authors=[a for a in authors if a],
        year=_date_year(item.get("issued")) or (years[0] if years else None),
        years=years,
        venue=venue if isinstance(venue, str) else None,
        doi=doi,
        url=item.get("URL"),
        exact_id=exact,
        pages=item.get("page") or None,
        preprint=True if item.get("type") == "posted-content" else None,
        record_id=doi,
    )


async def resolve_doi(session: Session, doi: str) -> DoiLookup:
    """Resolve a DOI to the record it is registered for.

    ACL Anthology DOIs come from the local index. Everything else goes to
    CrossRef, then to doi.org content negotiation (which also covers DataCite,
    arXiv's registrar), then to the handle system, which says whether the DOI
    exists at all.
    """
    doi = normalise_doi(doi) or doi
    ident = anthology_id(doi=doi)
    if ident and session.anthology is not None and await session.anthology.ready(session):
        record = await session.anthology.fetch(session, ident)
        if record is not None:
            return DoiLookup(True, record)

    response = await session.get(f"https://api.crossref.org/works/{quote(doi, safe='/')}",
                                 **_crossref_limits(session))
    if response is not None and response.status_code == 200:
        try:
            item = response.json().get("message")
        except ValueError:
            item = None
        if isinstance(item, dict):
            return DoiLookup(True, _crossref_candidate(item, exact=True))

    response = await session.get(f"https://doi.org/{quote(doi, safe='/')}", rate=5,
                                 headers={"Accept": "application/vnd.citationstyles.csl+json"})
    if response is not None and response.status_code == 200:
        try:
            item = response.json()
        except ValueError:
            item = None
        if isinstance(item, dict) and item.get("title"):
            return DoiLookup(True, _crossref_candidate(item, exact=True, source="doi.org"))

    response = await session.get(f"https://doi.org/api/handles/{quote(doi, safe='/')}", rate=5)
    if response is not None:
        try:
            code = response.json().get("responseCode")
        except ValueError:
            code = None
        if code == 1:
            return DoiLookup(True)
        if code == 100:
            return DoiLookup(False)
    return DoiLookup(None)


# ---------------------------------------------------------------------------
# Title searches
# ---------------------------------------------------------------------------


def _crossref_limits(session: Session) -> dict[str, float | int]:
    """CrossRef's published limits: 1 request/s, one at a time, unless identified."""
    return {"rate": 3.0, "concurrency": 3} if session.mailto else {"rate": 1.0, "concurrency": 1}


async def crossref_search(session: Session, ref: Reference) -> list[Candidate]:
    query = ref.title or ref.raw[:200]
    url = (
        "https://api.crossref.org/works?rows=5&select=title,subtitle,author,issued,published-print,"
        "published-online,DOI,container-title,URL,page,type"
        f"&query.bibliographic={quote(query)}"
    )
    if ref.authors:
        url += f"&query.author={quote(' '.join(ref.authors[:3]))}"
    if session.mailto:
        url += f"&mailto={quote(session.mailto)}"
    data = await session.json(url, **_crossref_limits(session))
    items = ((data or {}).get("message") or {}).get("items") or []
    return [_crossref_candidate(i) for i in items if isinstance(i, dict)]


async def openalex(session: Session, ref: Reference) -> list[Candidate]:
    """OpenAlex: broad coverage. Needs an API key for more than a trickle of requests."""
    params = {
        "per-page": "4",
        "select": "id,doi,title,publication_year,authorships,primary_location,biblio,type",
        "filter": "title.search:" + re.sub(r"[,|]", " ", ref.title or ref.raw[:200]),
    }
    if session.openalex_key:
        params["api_key"] = session.openalex_key
    elif session.mailto:
        params["mailto"] = session.mailto
    data = await session.json(f"https://api.openalex.org/works?{urlencode(params)}", rate=8)
    results = (data or {}).get("results") or []
    return [_openalex_candidate(r) for r in results if isinstance(r, dict)]


def _openalex_candidate(item: dict[str, Any], exact: bool = False) -> Candidate:
    authors = [
        ((a.get("author") or {}).get("display_name") or "")
        for a in (item.get("authorships") or []) if isinstance(a, dict)
    ]
    location = item.get("primary_location") or {}
    source = (location.get("source") or {}) if isinstance(location, dict) else {}
    biblio = item.get("biblio") or {}
    first, last = biblio.get("first_page"), biblio.get("last_page")
    return Candidate(
        source="openalex",
        title=item.get("title") or "",
        authors=[a for a in authors if a],
        year=_year(item.get("publication_year")),
        venue=source.get("display_name"),
        doi=normalise_doi(item.get("doi")),
        url=item.get("id"),
        exact_id=exact,
        pages=f"{first}-{last}" if first and last else first or None,
        preprint=True if item.get("type") == "preprint" or source.get("type") == "repository" else None,
        record_id=item.get("id"),
    )


#: OpenReview publishes ``ratelimit-policy: 20;w=60``.
_OPENREVIEW_RATE = 0.3


def _openreview_candidate(note: dict[str, Any]) -> Candidate | None:
    content = note.get("content") or {}

    def value(key: str) -> Any:
        field = content.get(key)
        return field.get("value") if isinstance(field, dict) else field

    title = value("title")
    if not title:
        return None
    venue = str(value("venue") or "")
    venueid = str(value("venueid") or "")
    accepted = bool(venue) and not venue.lower().startswith("submitted to") and not any(
        word in venueid.lower() for word in ("withdrawn", "rejected", "archive", "submission"))
    stamp = note.get("pdate") or note.get("cdate") or note.get("tcdate")
    year = time.gmtime(stamp / 1000).tm_year if isinstance(stamp, (int, float)) else None
    return Candidate(
        source="openreview",
        title=" ".join(str(title).split()),
        authors=[str(a) for a in (value("authors") or [])],
        year=year,
        venue=venue + ("" if accepted else " (not accepted)") if venue else None,
        url=f"https://openreview.net/forum?id={note.get('forum') or note.get('id')}",
        preprint=not accepted,
        record_id=note.get("id"),
    )


async def openreview(session: Session, ref: Reference) -> list[Candidate]:
    """OpenReview: ICLR, NeurIPS, TMLR, COLM and ICML papers that have no DOI."""
    if not ref.title:
        return []
    # "prefix" matches the title as a phrase; "terms" ranks loosely related papers first.
    params = {"term": ref.title, "type": "prefix", "content": "all", "group": "all",
              "source": "forum", "limit": "5"}
    data = await session.json(f"https://api2.openreview.net/notes/search?{urlencode(params)}",
                              rate=_OPENREVIEW_RATE, concurrency=2)
    notes = (data or {}).get("notes") or []
    return [c for c in (_openreview_candidate(n) for n in notes if isinstance(n, dict)) if c]


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
                venue=item.get("venue") or None,
                doi=normalise_doi(external.get("DOI")),
                url=item.get("url"),
            )
        )
    return out


async def openlibrary(session: Session, ref: Reference) -> list[Candidate]:
    """Open Library, for books: no bibliographic index of papers lists them."""
    if not ref.title:
        return []
    params = {"title": ref.title, "limit": "4",
              "fields": "title,subtitle,author_name,first_publish_year,publish_year,publisher,key"}
    if ref.authors:
        # Open Library matches "Bratman" but not "Michael E. Bratman".
        first = ref.authors[0]
        params["author"] = first.split(",")[0].strip() if "," in first else first.split()[-1]
    data = await session.json(f"https://openlibrary.org/search.json?{urlencode(params)}",
                              rate=0.5, concurrency=1)
    out: list[Candidate] = []
    for doc in (data or {}).get("docs") or []:
        if not isinstance(doc, dict) or not doc.get("title"):
            continue
        title = doc["title"] + (f": {doc['subtitle']}" if doc.get("subtitle") else "")
        years = tuple(sorted({y for y in doc.get("publish_year") or [] if isinstance(y, int)}))
        out.append(Candidate(
            source="openlibrary", title=title, authors=list(doc.get("author_name") or []),
            year=_year(doc.get("first_publish_year")), years=years,
            venue=(doc.get("publisher") or [None])[0], preprint=False,
            url=f"https://openlibrary.org{doc.get('key', '')}", record_id=doc.get("key"),
        ))
    return out


async def acl_anthology(session: Session, ref: Reference) -> list[Candidate]:
    if not ref.title or session.anthology is None or not await session.anthology.ready(session):
        return []
    return session.anthology.by_title(ref.title)


async def arxiv_search(session: Session, ref: Reference) -> list[Candidate]:
    if not ref.title or session.arxiv is None:
        return []
    return await session.arxiv.search(ref.title)


async def dblp_search(session: Session, ref: Reference) -> list[Candidate]:
    if not ref.title or session.dblp is None:
        return []
    return await session.dblp.search(ref.title)


# ---------------------------------------------------------------------------
# Publisher pages: the machine-readable record a landing page carries
# ---------------------------------------------------------------------------

#: Hosts whose pages are the publisher's own record of a paper. A page elsewhere
#: carrying citation_* tags is not a key source, whatever it says.
SCHOLARLY_HOSTS = (
    "proceedings.neurips.cc", "papers.nips.cc", "papers.neurips.cc", "proceedings.mlr.press",
    "openaccess.thecvf.com", "www.ecva.net", "jmlr.org", "www.jmlr.org", "ojs.aaai.org",
    "www.ijcai.org", "www.isca-archive.org", "dl.acm.org", "ieeexplore.ieee.org",
    "link.springer.com", "www.nature.com", "www.science.org", "www.pnas.org", "journals.plos.org",
    "direct.mit.edu", "academic.oup.com", "onlinelibrary.wiley.com", "www.cambridge.org",
    "www.roboticsproceedings.org", "www.biorxiv.org", "www.medrxiv.org", "iopscience.iop.org",
    "www.mdpi.com", "www.frontiersin.org", "www.jair.org", "jair.org", "pubmed.ncbi.nlm.nih.gov",
)


#: Hosts that are key sources themselves, reached through their own APIs.
KEY_SOURCE_HOSTS = ("arxiv.org", "aclanthology.org", "openreview.net", "doi.org", "dx.doi.org")


def scholarly_url(url: str | None) -> bool:
    """Whether a URL points at a key source's record of a paper."""
    host = urlparse(url or "").netloc.lower()
    return bool(host) and (host in SCHOLARLY_HOSTS or any(
        host == h or host.endswith("." + h) for h in KEY_SOURCE_HOSTS))


class _MetaTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        values = {k.lower(): (v or "") for k, v in attrs}
        name = (values.get("name") or values.get("property") or "").lower()
        if name.startswith("citation_") and values.get("content"):
            self.tags.setdefault(name, []).append(" ".join(values["content"].split()))


def parse_citation_meta(html: str, url: str) -> Candidate | None:
    """The Highwire ``citation_*`` tags publishers put on a paper's landing page."""
    parser = _MetaTags()
    try:
        parser.feed(html[:500_000])
    except Exception:
        return None
    tags = parser.tags

    def first(*names: str) -> str | None:
        for name in names:
            if tags.get(name):
                return tags[name][0]
        return None

    title = first("citation_title")
    if not title:
        return None
    first_page, last_page = first("citation_firstpage"), first("citation_lastpage")
    venue = first("citation_conference_title", "citation_journal_title", "citation_inbook_title",
                  "citation_book_title", "citation_publisher")
    return Candidate(
        source=urlparse(url).netloc,
        title=title,
        authors=tags.get("citation_author", []),
        year=_year(first("citation_publication_date", "citation_date", "citation_year",
                         "citation_online_date")),
        venue=venue,
        doi=normalise_doi(first("citation_doi")),
        url=url,
        exact_id=True,
        pages=f"{first_page}-{last_page}" if first_page and last_page else first_page,
        record_id=url,
    )


async def landing_page(session: Session, url: str) -> list[Candidate]:
    host = urlparse(url).netloc.lower()
    if host not in SCHOLARLY_HOSTS:
        return []
    permitted, delay = await session.allowed(url)
    if not permitted:
        session.errors.setdefault(host, "robots.txt does not allow automated access to its pages")
        return []
    response = await session.get(url, rate=1.0 / delay if delay else 1.0, retries=0)
    if response is None or response.status_code >= 400:
        return []
    record = parse_citation_meta(response.text, str(response.url))
    return [record] if record else []


async def confirm_url(session: Session, url: str, reference: Reference) -> list[Candidate]:
    """The key-source record behind a URL, by whichever route the URL belongs to."""
    parts = urlparse(url)
    host = parts.netloc.lower()
    if host.endswith("arxiv.org") and session.arxiv is not None:
        return await session.arxiv.by_id(re.sub(r"^/(?:abs|pdf|html)/", "", parts.path).removesuffix(".pdf"))
    if host.endswith("aclanthology.org") and session.anthology is not None:
        ident = anthology_id(url=url)
        if ident and await session.anthology.ready(session):
            record = await session.anthology.fetch(session, ident)
            return [record] if record else []
        return []
    if host.endswith("openreview.net"):
        # Fetching a note by id now requires a browser challenge; the search API
        # does not. The lead says OpenReview has it, so search OpenReview.
        return await openreview(session, reference)
    if host in ("doi.org", "dx.doi.org"):
        lookup = await resolve_doi(session, parts.path.lstrip("/"))
        return [lookup.record] if lookup.record else []
    if host.endswith("dblp.org"):
        return []                   # robots.txt disallows it; the SPARQL route covers DBLP
    return await landing_page(session, url)


ACADEMIC_SOURCES: tuple[SourceSpec, ...] = (
    # Tier 1 — consulted for every reference. The ACL Anthology is a local
    # index; arXiv and DBLP were prefetched for the whole bibliography in a few
    # batched requests, so each reference only picks up its share.
    SourceSpec("acl_anthology", acl_anthology, tier=1),
    SourceSpec("arxiv", arxiv_search, tier=1),
    SourceSpec("dblp", dblp_search, tier=1),
    SourceSpec("openlibrary", openlibrary, rate=0.5, tier=1, kinds=(Kind.BOOK,)),
    # Tier 2 — for references tier 1 did not confirm, or confirmed as a
    # different version than the one cited. These are rate-limited per request:
    # CrossRef allows anonymous clients one request a second, OpenReview twenty
    # a minute, and OpenAlex a small shared budget without a key.
    SourceSpec("crossref", crossref_search, rate=1.0, tier=2),
    SourceSpec("openalex", openalex, rate=8.0, tier=2),
    SourceSpec("openreview", openreview, rate=_OPENREVIEW_RATE, tier=2),
    SourceSpec("semantic_scholar", semantic_scholar, rate=1.0, tier=2, optional=True),
)
