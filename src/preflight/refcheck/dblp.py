"""DBLP, through its public SPARQL endpoint.

dblp.org's own search API now sits behind a bot challenge, and its robots.txt
disallows automated access. The SPARQL service at ``sparql.dblp.org`` is the
route DBLP offers for programs: its robots.txt allows ``/sparql`` with a
ten-second crawl delay, which the batcher enforces.

A title lookup scans the whole title index (about 25 seconds), so it costs the
same whether it asks for one title or a hundred. The whole bibliography
therefore goes in one query.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .batch import Batcher
from .core import Candidate, normalise_doi

if TYPE_CHECKING:
    from .sources import Session

ENDPOINT = "https://sparql.dblp.org/sparql"
CRAWL_DELAY = 10.0

# The subquery computes each title's lowercase form once and joins it against
# the wanted titles. Filtering ``LCASE(?title) = ?key`` instead compares every
# DBLP title with every key, which times out on a whole bibliography.
_QUERY = """PREFIX dblp: <https://dblp.org/rdf/schema#>
SELECT ?key ?pub ?title ?year ?venue ?pages ?doi ?ord ?name WHERE {
  { SELECT ?pub ?title ?key WHERE {
      ?pub dblp:title ?title .
      BIND(LCASE(STR(?title)) AS ?key)
      VALUES ?key { %s }
  } }
  OPTIONAL { ?pub dblp:yearOfPublication ?year }
  OPTIONAL { ?pub dblp:publishedIn ?venue }
  OPTIONAL { ?pub dblp:pagination ?pages }
  OPTIONAL { ?pub dblp:doi ?doi }
  OPTIONAL { ?pub dblp:hasSignature ?sig . ?sig dblp:signatureOrdinal ?ord .
             ?sig dblp:signatureDblpName ?name }
}"""


# A work's published version is often retitled ("Learning to summarize from
# human feedback" on arXiv, "... with human feedback" at NeurIPS), so no title
# lookup finds it. Its first author's papers from around the same year do.
_AUTHOR_QUERY = """PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
SELECT ?key ?pub ?title ?year ?venue ?pages ?doi ?ord ?name WHERE {
  { SELECT DISTINCT ?key ?pub WHERE {
      VALUES (?key ?person ?when) { %s }
      ?person_iri dblp:primaryCreatorName ?person .
      ?pub dblp:authoredBy ?person_iri ; dblp:yearOfPublication ?when .
  } }
  ?pub dblp:title ?title .
  OPTIONAL { ?pub dblp:yearOfPublication ?year }
  OPTIONAL { ?pub dblp:publishedIn ?venue }
  OPTIONAL { ?pub dblp:pagination ?pages }
  OPTIONAL { ?pub dblp:doi ?doi }
  OPTIONAL { ?pub dblp:hasSignature ?sig . ?sig dblp:signatureOrdinal ?ord .
             ?sig dblp:signatureDblpName ?name }
}"""


def author_key(name: str, year: int) -> str:
    return f"{' '.join(name.split())}|{year}"


def build_author_query(keys: list[str]) -> str:
    rows = []
    for key in keys:
        name, _, year = key.rpartition("|")
        for when in (int(year) - 1, int(year), int(year) + 1):
            rows.append(f'({_literal(key)} {_literal(name)} "{when}"^^xsd:gYear)')
    return _AUTHOR_QUERY % " ".join(rows)


def title_key(title: str) -> str:
    """DBLP titles end with a full stop; the key is the lowercased title without it."""
    return " ".join(title.split()).lower().rstrip(".")


def _literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_query(keys: list[str]) -> str:
    values = []
    for key in keys:
        values.append(_literal(key))
        if not key.endswith(("?", "!")):
            values.append(_literal(key + "."))
    return _QUERY % " ".join(values)


def parse_results(data: dict) -> dict[str, list[Candidate]]:
    """Group SPARQL rows (one per author) into one candidate per publication."""
    publications: dict[str, dict] = {}
    for row in (data.get("results") or {}).get("bindings") or []:
        value = {k: v.get("value", "") for k, v in row.items() if isinstance(v, dict)}
        pub = value.get("pub")
        if not pub:
            continue
        key = value.get("key", "")
        record = publications.setdefault(pub, {**value, "authors": {},
                                               "key": key if "|" in key else title_key(key)})
        if value.get("name") and value.get("ord", "").isdigit():
            record["authors"][int(value["ord"])] = re.sub(r"\s+\d{4}$", "", value["name"])
    out: dict[str, list[Candidate]] = {}
    for pub, record in publications.items():
        venue = record.get("venue") or None
        year = record.get("year", "")
        out.setdefault(record["key"], []).append(Candidate(
            source="dblp",
            title=(record.get("title") or "").rstrip("."),
            authors=[record["authors"][k] for k in sorted(record["authors"])],
            year=int(year) if year[:4].isdigit() else None,
            venue=venue,
            pages=record.get("pages") or None,
            doi=normalise_doi(record.get("doi")),
            url=pub,
            preprint=(venue or "").strip().lower() == "corr",
            record_id=pub.rsplit("/rec/", 1)[-1],
        ))
    return out


class Dblp:
    """Batched DBLP lookups over SPARQL: by title, and by first author and year."""

    def __init__(self, session: Session, interval: float = CRAWL_DELAY, batch: int = 100) -> None:
        self.session = session
        self.batcher = Batcher(self._fetch, interval=interval, size=batch, window=1.0)

    async def _fetch(self, kind: str, keys: list[str]) -> dict[str, list[Candidate]]:
        response = await self.session.get(
            ENDPOINT, method="POST", rate=1 / CRAWL_DELAY, retries=0, timeout=90.0, concurrency=1,
            data={"query": build_author_query(keys) if kind == "author" else build_query(keys)},
            headers={"Accept": "application/sparql-results+json"},
        )
        if response is None:
            raise LookupError("DBLP SPARQL: no response")
        if response.status_code >= 400:
            # The endpoint reports a query that ran out of time as a 429.
            detail = response.text[:200] if "timed out" in response.text else ""
            raise LookupError(f"DBLP SPARQL: HTTP {response.status_code} {detail}".strip())
        try:
            data = response.json()
        except ValueError as exc:
            raise LookupError("DBLP SPARQL returned something other than JSON") from exc
        if "results" not in data:
            raise LookupError(f"DBLP SPARQL: {str(data.get('exception', 'no results'))[:120]}")
        return parse_results(data)

    async def search(self, title: str) -> list[Candidate]:
        key = title_key(title)
        if len(key.split()) < 3:
            return []
        return await self.batcher.get("title", key)

    async def by_author(self, name: str, year: int) -> list[Candidate]:
        """Everything DBLP lists for an author (by their exact DBLP name) within a year."""
        if len(name.split()) < 2 or len(name.split()[0].strip(".")) < 2:
            return []                       # initials cannot name a DBLP person
        return await self.batcher.get("author", author_key(name, year))

    def prefetch(self, titles: list[str]) -> None:
        for title in titles:
            key = title_key(title)
            if len(key.split()) >= 3:
                self.batcher.submit("title", key)

    def close(self) -> None:
        self.batcher.close()
