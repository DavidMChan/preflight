"""Comparing a citation against the records that confirm it.

Finding a work is only half of checking a reference. A citation can name a real
paper and still print a DOI that belongs to a different one, the wrong pages, a
misspelled author, or an arXiv preprint whose peer-reviewed version has been out
for a year. Every comparison here is deterministic: it reads fields off records
that key sources returned, and reports only what those records contradict.

A work often has several records — an arXiv preprint, the proceedings version,
the DOI registration — that legitimately differ from one another. So a citation
field is checked against the records of the version it cites, and an author's
name is accepted if any record of the work spells it that way.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .core import Candidate, Discrepancy, Kind, Reference, parse_page_range, strip_spacing_accents

# ---------------------------------------------------------------------------
# Author names
# ---------------------------------------------------------------------------

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_ETAL = re.compile(r"(?i)^\s*(?:et\.?\s*al\.?|others|and others|\.\.\.|…)\s*$")
_CORPORATE = re.compile(
    r"(?i)\b(?:team|group|consortium|collaboration|committee|project|organi[sz]ation|"
    r"foundation|institute|university|laborator(?:y|ies)|labs?|inc|ltd|llc|corp(?:oration)?|"
    r"research|deepmind|openai|anthropic|google|meta|microsoft|nvidia|mistral|xai)\b"
)


@dataclass(frozen=True, slots=True)
class Name:
    """A person's name, reduced to what can be compared across sources."""

    raw: str
    given: tuple[str, ...]
    surname: str


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", strip_spacing_accents(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.replace("ł", "l").replace("Ł", "L").replace("ø", "o").replace("Ø", "O")


def _tokens(text: str) -> list[str]:
    """Lowercased name parts. Capitalised initials run together ("ZY") are split."""
    out: list[str] = []
    for token in re.split(r"[^A-Za-z0-9]+", _fold(text)):
        if not token:
            continue
        if token.isupper() and 1 < len(token) <= 3:
            out.extend(token.lower())            # "ZY" is two initials, not a name
        else:
            out.append(token.lower())
    return out


def split_name(name: str) -> Name | None:
    """Parse "Jane Q. Doe", "Doe, Jane Q." or DBLP's "Jane Doe 0001"."""
    text = re.sub(r"\s+\d{4}$", "", (name or "").strip())
    if not text or _ETAL.match(text):
        return None
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) >= 3 and parts[1].lower().strip(".") in _SUFFIXES:
            last, first = parts[0], parts[2]
        else:
            last, first = parts[0], " ".join(parts[1:])
        surname_tokens, given = _tokens(last), _tokens(first)
        given = [g for g in given if g not in _SUFFIXES]
    else:
        tokens = _tokens(text)
        while tokens and tokens[-1] in _SUFFIXES:
            tokens.pop()
        surname_tokens, given = tokens[-1:], tokens[:-1]
    if not surname_tokens:
        return None
    return Name(raw=" ".join(name.split()), given=tuple(given), surname=surname_tokens[-1])


#: Short forms a record and a citation can legitimately disagree on.
_NICKNAMES = {
    frozenset(pair) for pair in (
        ("tom", "thomas"), ("bill", "william"), ("will", "william"), ("bob", "robert"),
        ("rob", "robert"), ("jim", "james"), ("mike", "michael"), ("dan", "daniel"),
        ("dave", "david"), ("chris", "christopher"), ("alex", "alexander"), ("alex", "alexandra"),
        ("sam", "samuel"), ("sam", "samantha"), ("ben", "benjamin"), ("joe", "joseph"),
        ("matt", "matthew"), ("nick", "nicholas"), ("tony", "anthony"), ("andy", "andrew"),
        ("jon", "jonathan"), ("kate", "katherine"), ("katie", "katherine"), ("liz", "elizabeth"),
        ("beth", "elizabeth"), ("jen", "jennifer"), ("jenny", "jennifer"), ("steve", "stephen"),
        ("steve", "steven"), ("greg", "gregory"), ("jeff", "jeffrey"), ("pete", "peter"),
        ("rick", "richard"), ("dick", "richard"), ("ed", "edward"), ("ted", "edward"),
        ("ken", "kenneth"), ("ron", "ronald"), ("don", "donald"), ("tim", "timothy"),
        ("fred", "frederick"), ("vicky", "victoria"), ("sue", "susan"), ("nate", "nathan"),
        ("nate", "nathaniel"), ("zach", "zachary"), ("josh", "joshua"), ("abby", "abigail"),
    )
}


def given_compatible(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Whether two given names can belong to the same person.

    Initials match anything with the same first letter, a dropped middle name is
    fine, "Yu-Min" equals "Yumin", and "Tom" is "Thomas". Two different full first
    names are not.
    """
    if not a or not b:
        return True
    joined_a, joined_b = "".join(a), "".join(b)
    if joined_a == joined_b or a[0] == b[0] or frozenset((a[0], b[0])) in _NICKNAMES:
        return True
    if len(a[0]) == 1 or len(b[0]) == 1:
        return a[0][0] == b[0][0]
    return joined_a.startswith(joined_b) or joined_b.startswith(joined_a)


def _is_person(name: str) -> bool:
    return len(name.split()) > 1 and not _CORPORATE.search(name)


def _swapped(cited: Name, record: Name) -> bool:
    """Family-name-first order: "Wang Zekun" for "Zekun Wang"."""
    return bool(cited.given and record.given
                and cited.surname == record.given[0] and record.surname == cited.given[0])


def _align(cited: list[Name], record: list[Name], variants: dict[str, list[tuple[str, ...]]],
           truncated: bool) -> list[str]:
    """Problems with a cited author list, measured against one record of the work."""
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    unmatched: list[int] = []
    for i, person in enumerate(cited):
        same = [j for j, r in enumerate(record) if j not in used and r.surname == person.surname]
        if not same:
            same = [j for j, r in enumerate(record) if j not in used and _swapped(person, r)]
        if not same:
            unmatched.append(i)
            continue
        fitting = [j for j in same if given_compatible(person.given, record[j].given)] or same
        j = min(fitting, key=lambda k: abs(k - i))
        used.add(j)
        pairs.append((i, j))

    problems: list[str] = []
    for i in unmatched:
        hint = record[i].raw if i < len(record) and i not in used else None
        problems.append(f"{cited[i].raw} is not an author" + (f" (the record has {hint})" if hint else ""))

    for i, j in pairs:
        person, match = cited[i], record[j]
        if _swapped(person, match) or given_compatible(person.given, match.given):
            continue
        if any(given_compatible(person.given, g) for g in variants.get(person.surname, [])):
            continue                            # another record of the work spells it this way
        problems.append(f"{person.raw} should be {match.raw}")

    order = [j for _, j in sorted(pairs)]
    inversions = [(order[k - 1], order[k]) for k in range(1, len(order)) if order[k] < order[k - 1]]
    for earlier, later in inversions[:2]:
        problems.append(f"the record lists {record[later].raw} before {record[earlier].raw}")

    if not truncated and not unmatched and len(cited) < len(record):
        problems.append(f"the citation lists {len(cited)} of the record's {len(record)} authors "
                        "without 'et al.'")
    return problems


def compare_authors(cited: list[str], records: list[list[str]], truncated: bool) -> list[str]:
    """Problems with the cited authors, against the record they fit best."""
    people = [n for n in (split_name(a) for a in cited if _is_person(a)) if n is not None]
    versions = [[n for n in (split_name(a) for a in record) if n is not None] for record in records]
    versions = [v for v in versions if v]
    if not people or not versions:
        return []
    variants: dict[str, list[tuple[str, ...]]] = {}
    for version in versions:
        for person in version:
            variants.setdefault(person.surname, []).append(person.given)
    best: list[str] | None = None
    for version in versions:
        problems = _align(people, version, variants, truncated)
        if best is None or len(problems) < len(best):
            best = problems
    return best or []


# ---------------------------------------------------------------------------
# Venues
# ---------------------------------------------------------------------------

_VENUES: tuple[tuple[str, str], ...] = (
    ("findings", r"\bfindings\b"),
    ("workshop", r"\bworkshops?\b|\bsrw\b|\(student\)|student research"),
    ("emnlp", r"\bemnlp\b|empirical methods in natural language processing"),
    ("naacl", r"\bnaacl\b|north american chapter"),
    ("eacl", r"\beacl\b|european chapter of the association for computational linguistics"),
    ("aacl", r"\baacl\b|asia[- ]pacific chapter"),
    ("tacl", r"\btacl\b|transactions of the association for computational linguistics"),
    ("coling", r"\bcoling\b|international conference on computational linguistics"),
    ("lrec", r"\blrec\b|language resources and evaluation"),
    ("acl", r"\bacl\b|annual meeting of the association for computational linguistics"),
    ("neurips", r"\bneurips\b|\bnips\b|neural information processing systems"),
    ("icml", r"\bicml\b|international conference on machine learning"),
    ("iclr", r"\biclr\b|international conference on learning representations"),
    ("cvpr", r"\bcvpr\b|computer vision and pattern recognition"),
    ("iccv", r"\biccv\b|international conference on computer vision"),
    ("eccv", r"\beccv\b|european conference on computer vision"),
    ("aaai", r"\baaai\b"),
    ("ijcai", r"\bijcai\b|international joint conference on artificial intelligence"),
    ("colm", r"\bcolm\b|conference on language modeling"),
    ("aistats", r"\baistats\b|artificial intelligence and statistics"),
    ("uai", r"\buai\b|uncertainty in artificial intelligence"),
    ("kdd", r"\bkdd\b|knowledge discovery and data mining"),
    ("sigir", r"\bsigir\b"),
    ("chi", r"\bchi\b|human factors in computing systems"),
    ("icra", r"\bicra\b|international conference on robotics and automation"),
    ("iros", r"\biros\b|intelligent robots and systems"),
    ("corl", r"\bcorl\b|conference on robot learning"),
    ("interspeech", r"\binterspeech\b"),
    ("icassp", r"\bicassp\b|acoustics,? speech,? and signal processing"),
    ("jmlr", r"\bjmlr\b|journal of machine learning research"),
    ("tmlr", r"\btmlr\b|transactions on machine learning research"),
)
_VENUE_RES = tuple((tag, re.compile(pattern, re.IGNORECASE)) for tag, pattern in _VENUES)
_MARKERS = {"findings", "workshop"}


def venue_tags(text: str | None) -> set[str]:
    """Recognised venues and tracks in a venue string: ``{"findings", "emnlp"}``."""
    return {tag for tag, pattern in _VENUE_RES if pattern.search(text or "")}


def _primary(tags: set[str]) -> set[str]:
    return tags - _MARKERS


def venues_compatible(cited: set[str], record: set[str]) -> bool:
    a, b = _primary(cited), _primary(record)
    return not a or not b or bool(a & b)


_PUBLISHED_HINT = re.compile(
    r"(?i)\bin proceedings\b|\bproceedings of\b|\bjournal\b|\btransactions\b|\bconference\b|"
    r"\bworkshop\b|\bsymposium\b|\bpp\.|\bpages\b|\bvol\.|\bvolume\b|\bpress\b|\bpublishers?\b"
)
_PREPRINT_CITED = re.compile(r"(?i)\barxiv\b|\bcorr\b|\bpreprint\b|biorxiv|medrxiv|\bssrn\b")
_NOT_ARTICLES = {Kind.BOOK, Kind.THESIS, Kind.STANDARD, Kind.SOFTWARE, Kind.MODEL}


def cites_preprint(reference: Reference) -> bool:
    """Whether the citation points at a preprint, or at no venue at all."""
    if reference.venue:
        return bool(_PREPRINT_CITED.search(reference.venue))
    if reference.arxiv_id or _PREPRINT_CITED.search(reference.raw):
        return True
    return not _PUBLISHED_HINT.search(reference.raw)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _pages_agree(cited: str, record: str) -> bool | None:
    """True, False, or None when the two cannot be compared."""
    a, b = parse_page_range(cited), parse_page_range(record)
    if a is None or b is None:
        return None
    if b[1] is None:                           # an article number, or a single page
        return True if a[0] == b[0] else None
    if a[0] != b[0]:
        return False
    return not (a[1] and a[1] != b[1])


def _best_published(records: list[Candidate]) -> Candidate:
    return max(records, key=lambda r: (bool(r.venue), bool(r.doi), bool(r.pages), r.year or 0))


def compare(reference: Reference, records: list[Candidate]) -> list[Discrepancy]:
    """Everything the confirming records contradict in the citation."""
    if not records:
        return []
    out: list[Discrepancy] = []
    preprint_cited = cites_preprint(reference) and reference.kind not in _NOT_ARTICLES
    published = [r for r in records if not r.is_preprint]
    preprints = [r for r in records if r.is_preprint]

    if preprint_cited:
        if published:
            best = _best_published(published)
            what = "an arXiv preprint" if (reference.arxiv_id or _PREPRINT_CITED.search(
                f"{reference.venue or ''} {reference.raw}")) else "a work with no venue"
            out.append(Discrepancy(
                "version", f"cited as {what}, but it was published: {best.describe()}",
                source=best.source))
        else:
            noted = next((r for r in preprints if r.journal_ref or r.published_doi), None)
            if noted is not None:
                where = noted.journal_ref or f"doi:{noted.published_doi}"
                out.append(Discrepancy(
                    "version", f"cited as a preprint, but arXiv records a published version: {where}",
                    source=noted.source))
        same = preprints
    else:
        same = published

    cited_tags = venue_tags(reference.venue)
    matching_venue = [r for r in same if venues_compatible(cited_tags, venue_tags(r.venue))]

    if not preprint_cited and reference.venue:
        if published:
            recognised = [r for r in published if _primary(venue_tags(r.venue))]
            if _primary(cited_tags) and recognised and not any(
                venues_compatible(cited_tags, venue_tags(r.venue)) for r in recognised
            ):
                out.append(Discrepancy(
                    "venue", f"cited in {reference.venue}; the record has {recognised[0].venue}",
                    source=recognised[0].source))
            else:
                for marker in sorted(_MARKERS):
                    pool = [venue_tags(r.venue) for r in matching_venue if _primary(venue_tags(r.venue))]
                    if pool and all((marker in tags) != (marker in cited_tags) for tags in pool):
                        record = next(r for r in matching_venue if _primary(venue_tags(r.venue)))
                        out.append(Discrepancy(
                            "venue", f"cited in {reference.venue}; the record has {record.venue}",
                            source=record.source))
                        break
        elif _primary(cited_tags):
            out.append(Discrepancy(
                "venue", f"the cited venue ({reference.venue}) could not be confirmed; only a "
                "preprint of this work was found"))

    pool = matching_venue or same
    if reference.pages and not preprint_cited:
        answers = [(r, _pages_agree(reference.pages, r.pages)) for r in pool if r.pages]
        answers = [(r, ok) for r, ok in answers if ok is not None]
        if answers and not any(ok for _, ok in answers):
            record = answers[0][0]
            out.append(Discrepancy("pages", f"pages {reference.pages}; the record has {record.pages}",
                                   source=record.source))

    if reference.year and pool:
        years = set().union(*(r.all_years for r in pool))
        if years:
            ok = reference.year >= min(years) if preprint_cited else reference.year in years
            if not ok:
                shown = "/".join(str(y) for y in sorted(years))
                out.append(Discrepancy("year", f"year {reference.year}; the record has {shown}",
                                       source=pool[0].source))

    problems = compare_authors(reference.authors, [r.authors for r in records if r.authors],
                               truncated=reference.truncated_authors)
    if problems:
        out.append(Discrepancy("authors", "; ".join(problems)))
    return out
