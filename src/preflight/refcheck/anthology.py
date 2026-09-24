"""A local index of the ACL Anthology.

The Anthology is the record of every *CL venue — ACL, EMNLP, NAACL, EACL, AACL,
COLING, LREC, TACL, CL, Findings and the workshops — with page numbers, DOIs and
full author lists. It has no search API, but it publishes its whole bibliography
as one file (``anthology.bib.gz``, about 13 MB). That is downloaded at most once
a week and indexed into SQLite, so a lookup is a local query rather than a
request. A paper newer than the dump is fetched from its own ``.bib`` page.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from .core import Candidate, compact

if TYPE_CHECKING:
    from .sources import Session

DUMP_URL = "https://aclanthology.org/anthology.bib.gz"
BIB_URL = "https://aclanthology.org/{id}.bib"

_ID_FROM_DOI = re.compile(r"(?i)^10\.18653/v1/([a-z0-9][a-z0-9.\-]+)$")
_ID_FROM_URL = re.compile(r"(?i)aclanthology\.org/([a-z0-9]+[.\-][a-z0-9.\-]+?)(?:\.pdf|\.bib|/|$)")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id      TEXT PRIMARY KEY,
    compact TEXT NOT NULL,
    title   TEXT NOT NULL,
    authors TEXT NOT NULL,
    year    INTEGER,
    venue   TEXT,
    pages   TEXT,
    doi     TEXT,
    url     TEXT
);
CREATE INDEX IF NOT EXISTS papers_compact ON papers (compact);
"""


def anthology_id(doi: str | None = None, url: str | None = None) -> str | None:
    """The Anthology id a DOI (10.18653/v1/...) or an aclanthology.org URL names."""
    if doi:
        match = _ID_FROM_DOI.match(doi.strip())
        if match:
            return match.group(1).lower()
    if url:
        match = _ID_FROM_URL.search(url)
        if match:
            return match.group(1).rstrip(".").lower()
    return None


# ---------------------------------------------------------------------------
# BibTeX, as the Anthology writes it
# ---------------------------------------------------------------------------

_ACCENTS = {
    "'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303", "=": "\u0304",
    ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
    "r": "\u030a", "d": "\u0323", "b": "\u0331",
}
_LETTERS = {
    "i": "i", "j": "j", "o": "ø", "O": "Ø", "l": "ł", "L": "Ł", "ss": "ß", "ae": "æ", "AE": "Æ",
    "oe": "œ", "OE": "Œ", "aa": "å", "AA": "Å",
}
_ACCENT_RE = re.compile(r"\\([`'^\"~=.]|[uvHckrdb](?![a-zA-Z]))\s*(?:\{\s*(\\?[a-zA-Z])\s*\}|(\\?[a-zA-Z]))")
_LETTER_RE = re.compile(r"\\(ss|ae|AE|oe|OE|aa|AA|[ijoOlL])(?![a-zA-Z])")


def latex_to_text(text: str) -> str:
    """Turn the Anthology's LaTeX escapes into plain Unicode."""
    import unicodedata

    def accent(match: re.Match[str]) -> str:
        base = match.group(2) or match.group(3) or ""
        base = _LETTERS.get(base.lstrip("\\"), base.lstrip("\\"))
        return unicodedata.normalize("NFC", base + _ACCENTS.get(match.group(1), ""))

    text = _ACCENT_RE.sub(accent, text)
    text = _LETTER_RE.sub(lambda m: _LETTERS[m.group(1)], text)
    text = text.replace("--", "–").replace("\\&", "&").replace("~", " ")
    text = re.sub(r"\\[a-zA-Z]+\s*", "", text)
    return " ".join(text.replace("{", "").replace("}", "").split())


_FIELD = re.compile(r"^\s{4}(\w+)\s*=\s*(.*)$")


def parse_bibtex(text: str) -> Iterator[dict[str, str]]:
    """Entries from the Anthology's machine-written BibTeX, one dict per entry.

    The file is generated, so its layout is regular: a field starts on a line
    indented by four spaces, and continuation lines are indented further.
    """
    for chunk in text.split("\n@"):
        head, _, body = chunk.partition("\n")
        match = re.match(r"@?(\w+)\{([^,]*),", head.strip())
        if not match:
            continue
        entry: dict[str, str] = {"ENTRYTYPE": match.group(1).lower(), "ID": match.group(2)}
        name: str | None = None
        for line in body.split("\n"):
            field = _FIELD.match(line)
            if field:
                name = field.group(1).lower()
                entry[name] = field.group(2)
            elif name is not None and line.strip() not in {"", "}"}:
                entry[name] += " " + line.strip()
        for key, value in list(entry.items()):
            if key in {"ENTRYTYPE", "ID"}:
                continue
            value = value.strip().rstrip(",").strip()
            if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
            elif len(value) >= 2 and value[0] == "{" and value[-1] == "}":
                value = value[1:-1]
            entry[key] = value
        yield entry


def _person(name: str) -> str:
    """"Wang, Noah" to "Noah Wang"; "Smith, Jr., John" to "John Smith Jr."."""
    parts = [p.strip() for p in latex_to_text(name).split(",")]
    if len(parts) == 3:
        return f"{parts[2]} {parts[0]} {parts[1]}".strip()
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}".strip()
    return parts[0]


def _row(entry: dict[str, str]) -> tuple | None:
    if "author" not in entry or "title" not in entry:
        return None                   # a proceedings volume, not a paper
    url = entry.get("url", "")
    ident = anthology_id(url=url) or entry["ID"]
    title = latex_to_text(entry["title"])
    authors = [_person(a) for a in re.split(r"\s+and\s+", entry["author"]) if a.strip()]
    year = entry.get("year", "")
    return (
        ident, compact(title), title, json.dumps(authors),
        int(year) if year.isdigit() else None,
        latex_to_text(entry.get("booktitle") or entry.get("journal") or "") or None,
        (entry.get("pages") or "").replace("--", "-") or None,
        (entry.get("doi") or "").lower() or None,
        url or None,
    )


def build_index(dump: bytes, path: Path | str) -> int:
    """Index a downloaded dump into a fresh SQLite file. Returns the paper count."""
    text = gzip.decompress(dump).decode("utf-8", errors="replace")
    rows = [row for row in (_row(e) for e in parse_bibtex(text)) if row is not None]
    target = Path(path)
    temporary = target.with_suffix(".tmp")
    with suppress(FileNotFoundError):
        temporary.unlink()
    with sqlite3.connect(temporary) as conn:
        conn.executescript(_SCHEMA)
        conn.executemany("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.close()
    os.replace(temporary, target)
    return len(rows)


def _candidate(row: sqlite3.Row | tuple, exact: bool = False) -> Candidate:
    ident, _key, title, authors, year, venue, pages, doi, url = row
    return Candidate(
        source="acl_anthology", title=title, authors=json.loads(authors), year=year,
        venue=venue, pages=pages, doi=doi, url=url, exact_id=exact, preprint=False,
        record_id=ident,
    )


class Anthology:
    """The ACL Anthology, as a local index with a live fallback."""

    def __init__(self, path: Path | str | None, max_age_days: float = 7.0) -> None:
        self.path = Path(path).expanduser() if path else None
        self.max_age = max_age_days * 86400
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._tried = False
        self.error: str | None = None

    @classmethod
    def from_bibtex(cls, text: str) -> Anthology:
        """An in-memory index, for tests."""
        index = cls(None)
        index._conn = sqlite3.connect(":memory:")
        index._conn.executescript(_SCHEMA)
        rows = [row for row in (_row(e) for e in parse_bibtex(text)) if row is not None]
        index._conn.executemany("INSERT OR REPLACE INTO papers VALUES (?,?,?,?,?,?,?,?,?)", rows)
        index._tried = True
        return index

    @property
    def available(self) -> bool:
        return self._conn is not None

    async def ready(self, session: Session) -> bool:
        """Open the index, downloading and building it first if it is missing or stale."""
        if self._tried:
            return self.available
        async with self._lock:
            if self._tried:
                return self.available
            self._tried = True
            if self.path is None:
                return False
            stale = not self.path.exists() or time.time() - self.path.stat().st_mtime > self.max_age
            if stale:
                try:
                    dump = await session.download(DUMP_URL, timeout=60.0)
                    if dump:
                        self.path.parent.mkdir(parents=True, exist_ok=True)
                        await asyncio.to_thread(build_index, dump, self.path)
                except (OSError, sqlite3.Error, EOFError, gzip.BadGzipFile) as exc:
                    self.error = f"could not build the ACL Anthology index: {exc}"
            if self.path.exists():
                with suppress(sqlite3.Error):
                    self._conn = sqlite3.connect(self.path)
            elif self.error is None:
                self.error = "the ACL Anthology dump could not be downloaded"
            return self.available

    def by_id(self, ident: str) -> Candidate | None:
        if self._conn is None:
            return None
        with suppress(sqlite3.Error):
            row = self._conn.execute("SELECT * FROM papers WHERE id = ?", (ident.lower(),)).fetchone()
            if row:
                return _candidate(row, exact=True)
        return None

    def by_title(self, title: str) -> list[Candidate]:
        key = compact(latex_to_text(title))
        if self._conn is None or len(key) < 12:
            return []
        with suppress(sqlite3.Error):
            rows = self._conn.execute("SELECT * FROM papers WHERE compact = ?", (key,)).fetchall()
            return [_candidate(r) for r in rows]
        return []

    async def fetch(self, session: Session, ident: str) -> Candidate | None:
        """A paper by id, from the index or, if newer than the dump, from its own page."""
        found = self.by_id(ident)
        if found is not None:
            return found
        response = await session.get(BIB_URL.format(id=ident), rate=1.0)
        if response is None or response.status_code >= 400 or not response.text.lstrip().startswith("@"):
            return None
        for entry in parse_bibtex(response.text):
            row = _row(entry)
            if row is not None:
                return _candidate(row, exact=True)
        return None

    def close(self) -> None:
        if self._conn is not None:
            with suppress(sqlite3.Error):
                self._conn.close()
            self._conn = None
