"""arXiv, through the export API that arXiv provides for programs.

``export.arxiv.org`` is arXiv's host for automated access. Its terms ask for no
more than one request every three seconds over a single connection, which the
batcher enforces. Many titles go into one query (``ti:"..." OR ti:"..."``) and
many ids into one ``id_list``, so a whole bibliography costs a handful of
requests rather than one per reference.
"""

from __future__ import annotations

import asyncio
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from .batch import Batcher
from .core import Candidate, normalise_title, title_similarity
from .sources import user_agent

if TYPE_CHECKING:
    from .sources import Session

API = "https://export.arxiv.org/api/query"
_NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_ID = re.compile(r"arxiv\.org/abs/([^\s?#]+?)(?:v\d+)?$", re.IGNORECASE)
_ARXIV_ID = re.compile(r"(?i)(?:arxiv[:\s]*)?((?:\d{4}\.\d{4,5})|(?:[a-z\-]+(?:\.[a-z]{2})?/\d{7}))(?:v\d+)?")


def normalise_arxiv_id(value: str | None) -> str | None:
    match = _ARXIV_ID.search(value or "")
    return match.group(1).lower() if match else None


def parse_feed(xml: str) -> list[Candidate]:
    """Candidates from an arXiv Atom feed."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []
    out: list[Candidate] = []
    for entry in root.findall("a:entry", _NS):
        title = " ".join((entry.findtext("a:title", "", _NS) or "").split())
        match = _ID.search(entry.findtext("a:id", "", _NS) or "")
        if not match or not title or title.lower() == "error":
            continue
        ident = match.group(1).lower()
        published = (entry.findtext("a:published", "", _NS) or "")[:4]
        updated = (entry.findtext("a:updated", "", _NS) or "")[:4]
        years = tuple(int(y) for y in {published, updated} if y.isdigit())
        out.append(Candidate(
            source="arxiv",
            title=title,
            authors=[" ".join((a.findtext("a:name", "", _NS) or "").split())
                     for a in entry.findall("a:author", _NS)],
            year=int(published) if published.isdigit() else None,
            years=years,
            venue="arXiv",
            doi=f"10.48550/arxiv.{ident}",
            url=f"https://arxiv.org/abs/{ident}",
            preprint=True,
            journal_ref=" ".join((entry.findtext("arxiv:journal_ref", "", _NS) or "").split()) or None,
            published_doi=(entry.findtext("arxiv:doi", "", _NS) or "").strip().lower() or None,
            record_id=ident,
        ))
    return out


def _phrase(title: str) -> str:
    return " ".join(normalise_title(title).split()[:30])


class Arxiv:
    """Batched arXiv lookups by id and by title."""

    def __init__(self, session: Session, interval: float = 3.0, batch: int = 20) -> None:
        self.session = session
        self.interval = interval
        self.batcher = Batcher(self._fetch, interval=interval, size=batch)

    async def _fetch(self, kind: str, keys: list[str]) -> dict[str, list[Candidate]]:
        if kind == "id":
            params = {"id_list": ",".join(keys), "max_results": str(len(keys))}
        else:
            query = " OR ".join(f'ti:"{key}"' for key in keys)
            params = {"search_query": query, "max_results": str(min(100, 3 * len(keys)))}
        found = parse_feed(await asyncio.to_thread(self._request, f"{API}?{urlencode(params)}"))
        if kind == "id":
            return {key: [c for c in found if c.record_id == key] for key in keys}
        return {key: [c for c in found if title_similarity(key, c.title) >= 0.72] for key in keys}

    def _request(self, url: str) -> str:
        """One request through the standard library, as arXiv's API manual does it.

        arXiv's front end answers 406 to httpx whatever its headers, while the
        same request from urllib or curl succeeds. The batcher already keeps
        requests apart and one at a time, so this runs in a worker thread.
        """
        request = urllib.request.Request(url, headers={"User-Agent": user_agent(self.session.mailto)})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.session.errors["export.arxiv.org"] = f"{type(exc).__name__}: {exc}"
            raise LookupError(f"arXiv API: {exc}") from exc

    async def by_id(self, arxiv_id: str) -> list[Candidate]:
        """The record for an arXiv id; empty if arXiv has no such paper."""
        ident = normalise_arxiv_id(arxiv_id)
        if not ident:
            return []
        return [replace(c, exact_id=True) for c in await self.batcher.get("id", ident)]

    async def search(self, title: str) -> list[Candidate]:
        phrase = _phrase(title)
        if len(phrase.split()) < 3:
            return []
        return [c for c in await self.batcher.get("title", phrase)
                if title_similarity(title, c.title) >= 0.72]

    def prefetch(self, titles: list[str]) -> None:
        for title in titles:
            phrase = _phrase(title)
            if len(phrase.split()) >= 3:
                self.batcher.submit("title", phrase)

    def close(self) -> None:
        self.batcher.close()
