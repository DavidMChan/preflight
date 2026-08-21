"""Reference records, title normalisation, and match scoring.

The matching rules live here rather than in the sources, so every database is
judged by the same standard and a verdict can explain itself.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")
_STOPWORDS = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "via", "using"}
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", re.IGNORECASE)
_ARXIV_RE = re.compile(r"\b(?:arxiv[:\s]*)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def normalise_title(title: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", title or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.lower())
    return _SPACE.sub(" ", text).strip()


def title_key(title: str) -> str:
    """A cache key that survives subtitle and stopword drift."""
    words = [w for w in normalise_title(title).split() if w not in _STOPWORDS]
    return " ".join(words)


def title_similarity(left: str, right: str) -> float:
    """0..1 similarity over normalised titles, blending order and token overlap."""
    a, b = normalise_title(left), normalise_title(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ratio = SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()) - _STOPWORDS, set(b.split()) - _STOPWORDS
    if not ta or not tb:
        return ratio
    jaccard = len(ta & tb) / len(ta | tb)
    # A contained title (subtitle dropped by one index) should still score well.
    containment = len(ta & tb) / min(len(ta), len(tb))
    return max(ratio, 0.5 * jaccard + 0.5 * containment)


def surname(name: str) -> str:
    """Best-effort last name, handling "Doe, Jane" and "Jane Doe"."""
    cleaned = _PUNCT.sub(" ", (name or "").strip())
    parts = [p for p in cleaned.split() if len(p) > 1]
    if not parts:
        return ""
    if "," in (name or ""):
        return normalise_title(name.split(",")[0])
    return normalise_title(parts[-1])


def author_overlap(left: list[str], right: list[str]) -> float:
    """Share of the shorter author list whose surnames appear in the other."""
    a = {s for s in (surname(x) for x in left) if s}
    b = {s for s in (surname(x) for x in right) if s}
    if not a or not b:
        return -1.0          # unknown, not zero: absence of data is not evidence
    return len(a & b) / min(len(a), len(b))


class Kind(StrEnum):
    """What sort of thing is being cited.

    This drives which verifier runs. Asking CrossRef about a GitHub repository
    and then reporting "not found in any bibliographic database" is the single
    most common way a reference checker manufactures a false accusation.
    """

    PAPER = "paper"              # published in a venue or journal
    PREPRINT = "preprint"        # arXiv and friends
    BOOK = "book"
    THESIS = "thesis"
    SOFTWARE = "software"        # a repository or package
    MODEL = "model"              # a model or dataset card
    BLOG = "blog"                # vendor announcements, blog posts, news
    STANDARD = "standard"        # RFCs, ISO, legislation, technical reports
    WEB = "web"                  # anything else with a URL
    UNKNOWN = "unknown"

    @property
    def is_academic(self) -> bool:
        return self in {Kind.PAPER, Kind.PREPRINT, Kind.BOOK, Kind.THESIS}

    @property
    def expects_index(self) -> bool:
        """Whether absence from a bibliographic database is meaningful at all."""
        return self in {Kind.PAPER, Kind.PREPRINT, Kind.THESIS}

    @property
    def has_web_route(self) -> bool:
        """Whether a dedicated verifier exists for this kind.

        Distinct from ``not expects_index``: an unrecognised entry is not
        automatically a web artifact, and treating it as one would let genuinely
        unparsable fragments through as though they had been checked.
        """
        return self in {Kind.SOFTWARE, Kind.MODEL, Kind.BLOG, Kind.WEB, Kind.STANDARD}


_URL_RE = re.compile(r"(?i)https?\s?:\s?/\s?/[^\s<>\")\]]+|\bwww\.[^\s<>\")\]]+")

_KIND_HOSTS: tuple[tuple[str, Kind], ...] = (
    ("github.com", Kind.SOFTWARE),
    ("gitlab.com", Kind.SOFTWARE),
    ("bitbucket.org", Kind.SOFTWARE),
    ("pypi.org", Kind.SOFTWARE),
    ("huggingface.co", Kind.MODEL),
    ("kaggle.com", Kind.MODEL),
    ("zenodo.org", Kind.MODEL),
    ("arxiv.org", Kind.PREPRINT),
    ("aclanthology.org", Kind.PAPER),
    ("openreview.net", Kind.PAPER),
    ("rfc-editor.org", Kind.STANDARD),
    ("ietf.org", Kind.STANDARD),
    ("iso.org", Kind.STANDARD),
    ("nist.gov", Kind.STANDARD),
)

_BLOG_HINTS = (
    "blog", "news", "announc", "introducing ", "release notes", "press release",
    "retrieved from", "accessed ", "[online]", "web page", "website",
)
_SOFTWARE_HINTS = ("github", "gitlab", "repository", "software", "python package", "library, version")
_THESIS_HINTS = ("phd thesis", "ph.d. thesis", "master's thesis", "masters thesis", "dissertation")
_BOOK_HINTS = ("press,", "publishers", "isbn", "edition,", "chapter ")
_PAPER_HINTS = (
    "in proceedings", "proceedings of", "journal of", "transactions on", "conference on",
    "in advances in", "acl", "emnlp", "naacl", "neurips", "icml", "iclr", "aaai", "cvpr",
    "volume", "pages ", "pp.",
)


def normalise_url(url: str | None) -> str | None:
    """Repair the URL damage that PDF text extraction reliably causes.

    Extractors break long URLs across lines and columns, so a citation arrives as
    ``https: //host/path`` or ``ht tps://host/path`` or with spaces sprinkled
    through the path. Left alone these produce a schemeless, unusable string that
    the HTTP client rejects outright, which is worse than having no URL at all.
    Anything that still is not an absolute http(s) URL is discarded.
    """
    if not url:
        return None
    text = url.strip().rstrip(".,;:)]}\'\"")
    text = re.sub(r"^h\s*t\s*t\s*p(s?)\s*:", r"http\1:", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)^(https?:)\s*/\s*/\s*", r"\1//", text)
    text = re.sub(r"\s+", "", text)
    if text.lower().startswith("www."):
        text = "https://" + text
    match = re.match(r"(?i)^https?://([^/\s]+)", text)
    if not match:
        return None
    host = match.group(1)
    # A host with no dot is a fragment the extractor cut short ("https://model").
    # Checking it would only produce a spurious dead-link report.
    if "." not in host.strip("."):
        return None
    return text


def extract_url(text: str) -> str | None:
    match = _URL_RE.search(text or "")
    if not match:
        return None
    return normalise_url(match.group(0))


def classify(raw: str, *, venue: str | None = None, doi: str | None = None,
             arxiv_id: str | None = None, url: str | None = None) -> Kind:
    """Best-effort type for a reference, from its text and identifiers.

    Deliberately conservative: a DOI or a proceedings marker outranks a URL,
    because plenty of properly published papers also link to a repository.
    """
    text = f"{raw} {venue or ''}".lower()
    if doi:
        return Kind.PAPER
    if arxiv_id:
        return Kind.PREPRINT

    href = (url or extract_url(raw) or "").lower()
    if href:
        for host, kind in _KIND_HOSTS:
            if host in href:
                return kind

    if any(h in text for h in _THESIS_HINTS):
        return Kind.THESIS
    if any(h in text for h in _PAPER_HINTS):
        return Kind.PAPER
    if any(h in text for h in _SOFTWARE_HINTS):
        return Kind.SOFTWARE
    if any(h in text for h in _BLOG_HINTS):
        return Kind.BLOG
    if any(h in text for h in _BOOK_HINTS):
        return Kind.BOOK
    if href:
        return Kind.WEB
    return Kind.UNKNOWN


class Status(StrEnum):
    """What we concluded about one reference."""

    VERIFIED = "verified"                # found in a bibliographic database
    RESOLVED = "resolved"                # a non-paper citation whose target exists
    AUTHOR_MISMATCH = "author_mismatch"
    NOT_FOUND = "not_found"
    UNRESOLVED = "unresolved"            # a non-paper citation we could not reach
    UNINDEXED = "unindexed"              # not a paper, and not expected to be indexed
    UNPARSED = "unparsed"
    ERROR = "error"

    @property
    def is_suspicious(self) -> bool:
        """Only statuses that a reader should actually go and check by hand."""
        return self in {Status.AUTHOR_MISMATCH, Status.NOT_FOUND, Status.UNRESOLVED}


@dataclass(slots=True)
class Reference:
    """One bibliography entry, as parsed from the PDF."""

    raw: str
    index: int
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    kind: Kind = Kind.UNKNOWN

    def classify_kind(self) -> Kind:
        if self.kind is Kind.UNKNOWN:
            self.kind = classify(self.raw, venue=self.venue, doi=self.doi,
                                 arxiv_id=self.arxiv_id, url=self.url)
        return self.kind

    @property
    def key(self) -> str:
        return title_key(self.title or self.raw[:120])

    @property
    def label(self) -> str:
        return self.title or " ".join(self.raw.split())[:90]

    @property
    def is_usable(self) -> bool:
        """Enough signal to look this up by some route.

        A repository or blog citation often has a terse title or none at all,
        but a URL is all its verifier needs — so requiring a title here would
        silently drop exactly the citations the type routing exists to handle.
        """
        if self.doi or self.arxiv_id:
            return True
        if self.kind.has_web_route:
            # A model card, repository or announcement is identified by its URL
            # or by a short product name ("Grok-4.3"), and the web-search tier
            # can settle either. Demanding a paper-shaped title here would drop
            # exactly the citations the type routing exists to handle.
            return bool(self.url or self.title)
        return bool(self.title and len(self.title.split()) >= 3)

    def scavenge(self) -> None:
        """Pull identifiers straight out of the raw text when parsing missed them."""
        if not self.doi:
            match = _DOI_RE.search(self.raw)
            if match:
                self.doi = match.group(0).rstrip(".,;")
        if not self.arxiv_id and "arxiv" in self.raw.lower():
            match = _ARXIV_RE.search(self.raw)
            if match:
                self.arxiv_id = match.group(1)
        self.url = normalise_url(self.url) or extract_url(self.raw)
        if not self.year:
            years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", self.raw)]
            if years:
                # Citations often carry an access date as well; the publication
                # year is the earlier one in practice.
                self.year = min(years)


@dataclass(slots=True)
class Candidate:
    """A record a database returned for a reference."""

    source: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    exact_id: bool = False       # matched by DOI/arXiv id rather than by search

    def score(self, reference: Reference) -> float:
        if self.exact_id:
            return 1.0
        return title_similarity(reference.title or reference.raw, self.title)


@dataclass(slots=True)
class Verdict:
    """The conclusion for one reference, with the evidence behind it."""

    reference: Reference
    status: Status
    source: str | None = None
    matched_title: str | None = None
    score: float = 0.0
    author_score: float = -1.0
    note: str = ""
    url: str | None = None
    checked_sources: tuple[str, ...] = ()

    @property
    def is_suspicious(self) -> bool:
        return self.status.is_suspicious

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.reference.index,
            "title": self.reference.label,
            "status": self.status.value,
            "source": self.source,
            "matched_title": self.matched_title,
            "score": round(self.score, 3),
            "author_score": round(self.author_score, 3),
            "note": self.note,
            "url": self.url,
        }


@dataclass(slots=True)
class MatchPolicy:
    """Thresholds that decide verified / mismatch / not found."""

    accept_title: float = 0.90
    review_title: float = 0.72
    accept_author: float = 0.34      # one shared surname out of three is enough
    year_slack: int = 2

    def judge(self, reference: Reference, candidate: Candidate) -> tuple[str, float, float]:
        """Return (outcome, title score, author score).

        Outcomes: ``accept``, ``review`` (send to the model), ``reject``.
        """
        score = candidate.score(reference)
        authors = author_overlap(reference.authors, candidate.authors)
        if candidate.exact_id:
            return "accept", 1.0, authors
        if score < self.review_title:
            return "reject", score, authors
        if reference.year and candidate.year and abs(reference.year - candidate.year) > self.year_slack:
            # Same title, wrong year: usually a preprint/proceedings pair, so
            # this is a question for the adjudicator rather than a rejection.
            return "review", score, authors
        if score >= self.accept_title:
            if authors >= 0.0 and authors < self.accept_author:
                return "review", score, authors
            return "accept", score, authors
        return "review", score, authors
