"""Cross-checking in-text citations against the bibliography, in both directions.

Purely textual and purely deterministic: no model call, just counting and
matching. Two failure modes matter to authors and are cheap to catch from the
PDF alone:

  * a citation in the prose that resolves to no bibliography entry at all --
    usually the ghost of an entry someone deleted while trimming the list, and
    a real hazard if a reviewer follows it;
  * a bibliography entry that is never cited anywhere in the document -- almost
    always a leftover from editing, not misconduct, so this is worded gently.

Neither is a documented desk-rejection condition, so both report WARNING,
never ERROR -- a venue profile can promote either via its ``severity:`` map.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import analysis
from ..context import CheckContext
from ..models import Evidence, Finding
from ..refcheck.parse import bibliography_lines, segment
from ..registry import register

MODULE = "core.citations"

# ---------------------------------------------------------------------------
# Author-year citation grammar
# ---------------------------------------------------------------------------

#: A capitalized name token: letters (incl. accented), hyphens and apostrophes,
#: so "Müller", "Garcia-Molina" and "O'Neill" all match as one token.
_NAME = r"[A-ZÀ-ÖØ-Þ][\w''\-À-ÖØ-öø-ÿ]*"
#: "(see also ...)", "(e.g., ...)" -- prefixes that introduce a citation without
#: being part of it. Consumed but not part of the extracted name.
_PREFIX = r"(?:(?:[Ss]ee\s+also|[Ss]ee|[Ee]\.g\.|[Ii]\.e\.|[Cc]f\.|[Aa]lso)\s*,?\s*)?"
#: One or more 4-digit years, comma-separated, each with an optional a/b/c
#: disambiguation suffix: "2024", "2024a", "2025, 2026". A later item may also
#: be a bare suffix letter ("2023a,b"), shorthand for "2023a" and "2023b".
_YEARS = r"(?P<years>(?:19|20)\d{2}[a-z]?(?:\s*,\s*(?:(?:19|20)\d{2}[a-z]?|[a-z]))*)"
_AUTHORS = (
    rf"(?P<name>{_NAME})(?:\s+(?:and|&)\s+{_NAME})?(?:\s+et\s*al\.?)?"
)

#: One citation unit inside a parenthetical, split on ";" first:
#: "Smith and Jones, 2021", "Vaswani et al., 2017", "OpenAI, 2024".
_PAREN_UNIT = re.compile(rf"^\s*{_PREFIX}{_AUTHORS}\s*,?\s+{_YEARS}\s*$")
#: Narrative form: "Vaswani et al. (2017)", "Khattab and Zaharia (2020)".
#: natbib's plainnat brackets the year instead -- "Vickers et al. [2012, 2018]"
#: -- which is still author-year, not a numeric citation. The delimiters are
#: matched as a pair of character classes rather than as an alternation, so a
#: mismatched "Smith (2020]" also matches; that costs nothing worth the noise.
_NARRATIVE = re.compile(rf"{_AUTHORS}\s*[(\[]\s*{_YEARS}\s*[)\]]")
#: Candidate parenthetical spans to split and test against `_PAREN_UNIT`.
#: Square brackets are included because natbib's plainnat wraps the whole
#: citation in them -- "[Hendrycks et al., 2021b,a]" -- and a bracket holding a
#: bare number still fails `_PAREN_UNIT`, which wants a name.
_PAREN_SPAN = re.compile(r"[(\[]([^()\[\]]{3,300})[)\]]")

#: Numeric styles: "[12]", "[3, 4]", "[5-7]".
_NUMERIC_CITE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
#: Most numbers a real bracketed citation group carries. Longer bracketed runs
#: of numbers are data -- a label vector, a shape, a JSON array in an example.
_MAX_NUMERIC_GROUP = 8
#: A bracketed number this large is a year, not an entry number: no reference
#: list runs to 1900 entries, but "Vickers et al. [2012, 2018]" is everywhere.
_YEAR_LIKE = 1900

#: The publication year of a bibliography entry. The lookarounds keep it from
#: reading the year out of a number that merely contains one: "arXiv:1904.09223"
#: is a 2019 paper, and the page range in "31(9):2019-2029, 2024." holds two
#: year-shaped numbers before the real year. Taking either leaves the entry
#: matching no citation at all.
_ENTRY_YEAR = re.compile(r"(?<![\d.:\-–—])(?:19|20)\d{2}[a-z]?\b(?![.\-–—]\d)")
_ENTRY_LEADING_INDEX = re.compile(r"^\s*\[?(\d+)\]?[.)]?\s")


@dataclass(slots=True, frozen=True)
class _Citation:
    surname: str
    year: str  # includes the a/b/c suffix if present
    start: int  # character offset into the text this was extracted from
    context: str


def _split_years(years_blob: str) -> list[str]:
    """Expand a comma-joined year list, including the bare-suffix shorthand.

    "2023a,b" means "2023a" and "2023b"; the second token borrows the digits
    from whichever full year token most recently appeared.
    """
    out: list[str] = []
    last_digits: str | None = None
    for token in (t.strip() for t in years_blob.split(",")):
        if not token:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}[a-z]?", token):
            last_digits = token[:4]
            out.append(token)
        elif re.fullmatch(r"[a-z]", token) and last_digits:
            out.append(last_digits + token)
        else:
            out.append(token)
    return out


def _first_surname(name_blob: str) -> str:
    """First author's surname from a captured author group.

    "Smith and Jones" -> "Smith"; "Vaswani" (before an "et al." that regex
    already stripped) -> "Vaswani"; a bare corporate name is its own surname.
    """
    return name_blob.split()[0]


def _snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return " ".join(text[lo:hi].split())


def extract_author_year_citations(text: str) -> list[_Citation]:
    """Every author-year citation found in ``text``, one per (author, year) pair.

    A multi-year citation like "(Wang et al., 2025, 2026)" expands to two
    citations sharing one author; a semicolon list expands to one per item.
    """
    out: list[_Citation] = []
    seen_spans: set[tuple[int, int]] = set()

    for pmatch in _PAREN_SPAN.finditer(text):
        inner = pmatch.group(1)
        base = pmatch.start(1)
        offset = 0
        for chunk in inner.split(";"):
            unit = _PAREN_UNIT.match(chunk)
            if unit:
                surname = _first_surname(unit.group("name"))
                for year in _split_years(unit.group("years")):
                    out.append(_Citation(surname, year, base + offset,
                                         _snippet(text, pmatch.start(), pmatch.end())))
            offset += len(chunk) + 1
        seen_spans.add((pmatch.start(), pmatch.end()))

    for nmatch in _NARRATIVE.finditer(text):
        # Skip narrative matches that are really just the inside of a
        # parenthetical span already counted above.
        if any(s <= nmatch.start() and nmatch.end() <= e for s, e in seen_spans):
            continue
        surname = _first_surname(nmatch.group("name"))
        for year in _split_years(nmatch.group("years")):
            out.append(_Citation(surname, year, nmatch.start(),
                                 _snippet(text, nmatch.start(), nmatch.end())))
    return out


def detect_numeric_style(text: str, entry_count: int) -> bool:
    """Whether the paper cites by bracketed number rather than author-year."""
    author_year_hits = sum(
        1 for pmatch in _PAREN_SPAN.finditer(text) for chunk in pmatch.group(1).split(";")
        if _PAREN_UNIT.match(chunk)
    ) + len(_NARRATIVE.findall(text))
    numeric_hits = sum(len(n) for _, n in _numeric_citations(text, entry_count))
    if author_year_hits == 0 and numeric_hits >= max(3, entry_count // 10):
        return True
    return numeric_hits > author_year_hits * 3


def _expand_numeric(blob: str) -> list[int]:
    out: list[int] = []
    for part in blob.split(","):
        part = part.strip()
        if "-" in part:
            lo, _, hi = part.partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                out.extend(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            out.append(int(part))
    return out


def _is_citation_group(numbers: list[int], ceiling: int | None) -> bool:
    """Whether a bracketed run of numbers can be a citation rather than data.

    Papers print plenty of bracketed numbers that are not citations at all:
    label vectors, tensor shapes, JSON arrays quoted from a prompt. Counting
    those as citations is not a harmless overcount -- enough of them flip
    `detect_numeric_style` for an author-year paper, and the whole citation
    pair of checks then reads the bibliography by position and reports nearly
    every entry as uncited. Three properties separate the two cheaply:

      * no zero -- reference lists start at [1];
      * no repeats -- "[0, 0, 2]" is data, "[2, 2]" is nobody's citation;
      * not too long -- see `_MAX_NUMERIC_GROUP`;
      * nothing year-sized -- see `_YEAR_LIKE`.

    ``ceiling``, when given, additionally requires every number to name an
    entry that exists. Callers asking "is this paper numeric-style at all?"
    pass it; the resolution check does not, because a citation pointing past
    the end of the list is exactly the failure it exists to report.
    """
    if not numbers or len(numbers) > _MAX_NUMERIC_GROUP:
        return False
    if any(n < 1 or n >= _YEAR_LIKE for n in numbers):
        return False
    if len(set(numbers)) != len(numbers):
        return False
    if ceiling is not None and any(n > ceiling for n in numbers):
        return False
    return True


def _numeric_citations(text: str, ceiling: int | None) -> list[tuple[re.Match[str], list[int]]]:
    """Bracketed spans in ``text`` that survive `_is_citation_group`."""
    out = []
    for match in _NUMERIC_CITE.finditer(text):
        numbers = _expand_numeric(match.group(1))
        if _is_citation_group(numbers, ceiling):
            out.append((match, numbers))
    return out


# ---------------------------------------------------------------------------
# Bibliography indexing
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Entry:
    index: int  # 1-based position in the bibliography as printed
    raw: str
    surname: str
    year: str | None  # first year token found, with suffix if present


def _index_entries(entries: list[str]) -> list[_Entry]:
    out = []
    for i, raw in enumerate(entries, start=1):
        year_match = _ENTRY_YEAR.search(raw)
        idx_match = _ENTRY_LEADING_INDEX.match(raw)
        printed_index = int(idx_match.group(1)) if idx_match else i
        prefix = raw[: year_match.start()] if year_match else raw
        first_author = re.split(r"\s+and\s+|,", prefix, maxsplit=1)[0].strip()
        # A lone corporate author ("DeepSeek-AI. DeepSeek-V3 technical report.
        # CoRR, ...") has no comma before the title, so the split above runs
        # on into it. The author block ends at the first full stop that is not
        # an initial's.
        first_author = re.split(r"(?<!\b[A-Z])\.\s+", first_author, maxsplit=1)[0].strip()
        words = [w.strip(".,") for w in first_author.split() if w.strip(".,")]
        surname = words[-1] if words else (raw.split()[0] if raw.split() else "")
        out.append(_Entry(index=printed_index, raw=raw, surname=surname,
                          year=year_match.group(0) if year_match else None))
    return out


def _dehyphenate(text: str) -> str:
    return re.sub(r"[-''‑]", "", text)


#: A capitalized name at the end of a longer word: the shape a lost space
#: leaves behind. The run-in text is as often an acronym ("MTCMBKong") as
#: ordinary prose ("TCMBenchYue"), so what precedes the capital is not
#: constrained here -- `_unglue` uses its length instead.
_GLUED_SURNAME = re.compile(r"(?<=\w)([A-Z][a-z''\-À-ÖØ-öø-ÿ]+)$")


def _unglue(surname: str) -> str | None:
    """"TCMBenchYue" -> "Yue"; a name with nothing run into it -> None.

    PDF text extraction drops the space in front of a citation often enough
    that it is worth undoing: "the TCMBench Yue et al. [2024] benchmark" comes
    back as one word, and the surname the grammar then captures matches no
    bibliography entry. The signal is a capital letter mid-word, which
    ordinary prose does not produce.

    What precedes the capital has to be either long enough to be a word of its
    own or an acronym -- all caps, digits and hyphens, as in "MTCMBKong" or
    the "7BChen" left by a model name. "McDonald", "MacLeod" and "DeSantis"
    carry the same shape but fail both tests, so they stay whole.
    """
    match = _GLUED_SURNAME.search(surname)
    if not match:
        return None
    prefix = surname[: match.start(1)]
    if len(prefix) >= 4 or re.fullmatch(r"[A-Z0-9\-]+", prefix):
        return match.group(1)
    return None


def _name_pattern(surname: str) -> str:
    """A regex source matching ``surname`` in body text, glued or not.

    The plain form is case-insensitive; the glued form is not, because the
    capital is the whole reason to believe a word boundary was lost there.
    Case-folding it would let a short name match inside an unrelated word --
    "Li" in "Mali", say.
    """
    name = re.escape(surname)
    return rf"(?:(?i:\b{name}\b)|(?<=[A-Za-z0-9]){name}\b)"


def _name_matches(surname: str, text: str) -> bool:
    """Whether ``surname`` appears in ``text``, tolerant of one PDF artifact.

    A hyphenated surname that happens to break across a line ("Chevalier-\\n
    Boisvert") gets its hyphen eaten by the shared line-rejoin logic, so the
    exact word can go missing on one side of the comparison even though the
    citation is completely legitimate. Falling back to a dehyphenated
    substring match catches that without loosening matching for ordinary
    short surnames, which the length floor guards against.
    """
    if re.search(rf"\b{re.escape(surname)}\b", text, re.IGNORECASE):
        return True
    bare = _dehyphenate(surname)
    if len(bare) >= 6 and ("-" in surname or "-" in text or "‑" in text):
        return bare.lower() in _dehyphenate(text).lower()
    return False


def _resolves(citation: _Citation, entries: list[str]) -> bool:
    """Lenient on purpose: any entry naming the surname in the right year counts.

    A stricter "must be the first author" match would occasionally reject a
    citation whose author-list parsing we got slightly wrong, which is a far
    worse failure here than the reverse -- see the module docstring.
    """
    year_re = re.compile(rf"\b{re.escape(citation.year)}\b")
    names = [citation.surname, *filter(None, [_unglue(citation.surname)])]
    return any(any(_name_matches(n, entry) for n in names) and year_re.search(entry)
               for entry in entries)


def _resolves_numeric(index: int, entries: list[_Entry]) -> bool:
    return any(e.index == index for e in entries)


def _surnames_equal(a: str, b: str) -> bool:
    if a.lower() == b.lower():
        return True
    if (_unglue(a) or a).lower() == (_unglue(b) or b).lower():
        return True
    bare_a, bare_b = _dehyphenate(a).lower(), _dehyphenate(b).lower()
    return len(bare_a) >= 6 and bare_a == bare_b


def _cited_by_index(entry: _Entry, citations: list[_Citation]) -> bool:
    """Match against citations already parsed by the shared grammar.

    Catches forms the window search below cannot, chiefly the "2023a,b"
    shorthand: the literal substring "2023b" never appears in that text, only
    after `_split_years` has expanded it.
    """
    return any(entry.year == c.year and _surnames_equal(entry.surname, c.surname) for c in citations)


def _cited_anywhere(entry: _Entry, text: str) -> bool:
    """Whether an entry's (surname, year) pair turns up together anywhere.

    A window rather than an exact grammar match: the reverse direction only
    needs to know the reference was used *somewhere* -- in a caption, a table,
    an appendix section -- not to re-derive citation syntax a second time.
    """
    if not entry.year or not entry.surname:
        return False
    name_re = _name_pattern(entry.surname)
    year_re = re.escape(entry.year)
    pattern = re.compile(
        rf"{name_re}[\s\S]{{0,80}}\b{year_re}\b|\b{year_re}\b[\s\S]{{0,80}}{name_re}"
    )
    if pattern.search(text):
        return True
    bare = _dehyphenate(entry.surname)
    if len(bare) >= 6 and ("-" in entry.surname or "-" in text or "‑" in text):
        loose = re.compile(
            rf"{re.escape(bare)}[\s\S]{{0,80}}\b{year_re}\b|\b{year_re}\b[\s\S]{{0,80}}{re.escape(bare)}",
            re.IGNORECASE,
        )
        return bool(loose.search(_dehyphenate(text)))
    return False


def _document_minus_bibliography(ctx: CheckContext) -> str:
    """Everything except the reference list itself: body, appendix, captions.

    A reference's own bibliography line always contains its author and year,
    so leaving that span in would make every entry trivially "cited" by
    itself. This mirrors the boundary `bibliography_lines` finds and inverts
    it, rather than re-deriving it differently.
    """
    aliases = [str(s) for s in (ctx.conf("structure.references_aliases",
                                          ["References", "Bibliography"]) or [])]
    start = ctx.doc.find_heading(aliases)
    if start is None:
        return analysis.full_text(ctx)
    end = next((h for h in ctx.doc.headings if h.order > start.order), None)

    ignore = re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))
    lines: list[str] = []
    in_biblio = False
    for line in ctx.doc.reading_order:
        if ignore.match(line.text.strip()):
            continue
        if not in_biblio and line.page == start.page and abs(line.bbox[1] - start.bbox[1]) < 1.0:
            in_biblio = True
            continue
        if in_biblio:
            if end is not None and line.page == end.page and abs(line.bbox[1] - end.bbox[1]) < 1.0:
                in_biblio = False
                lines.append(line.text)
            continue
        lines.append(line.text)
    return analysis.clean_text(ctx, "\n".join(lines))


def _find_page(ctx: CheckContext, surname: str, year: str) -> int | None:
    """Best-effort page lookup for a citation, by scanning body lines."""
    name_re = re.compile(_name_pattern(surname))
    year_re = re.compile(rf"\b{re.escape(year)}\b")
    fallback: int | None = None
    for line in ctx.doc.reading_order:
        text = line.text
        if name_re.search(text):
            if year_re.search(text):
                return line.page
            if fallback is None:
                fallback = line.page
    return fallback


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@register("citation_resolution", "Citations resolve to the bibliography",
          module=MODULE, category="references", order=64)
def check_citation_resolution(ctx: CheckContext) -> Finding:
    """Every in-text citation should match some entry in the reference list."""
    entries = segment(ctx, bibliography_lines(ctx))
    if not entries:
        return ctx.skip("citation_resolution", "Citations resolve to the bibliography",
                        "No bibliography was found to check citations against.", category="references")

    body = analysis.body_text(ctx)
    numeric = detect_numeric_style(body, len(entries))
    max_report = int(ctx.conf("citations.max_reported", 15))

    if numeric:
        indexed = _index_entries(entries)
        seen: dict[int, tuple[int, str]] = {}
        for m, numbers in _numeric_citations(body, None):
            for n in numbers:
                seen.setdefault(n, (m.start(), _snippet(body, m.start(), m.end())))
        unresolved = [(n, ctx_str) for n, (pos, ctx_str) in sorted(seen.items())
                     if not _resolves_numeric(n, indexed)]
        if not unresolved:
            return ctx.ok("citation_resolution", "Citations resolve to the bibliography",
                          f"All {len(seen)} distinct numeric citation(s) resolve to one of the "
                          f"{len(entries)} bibliography entries.", category="references")
        evidence = [Evidence(detail=f"citation [{n}] has no bibliography entry at that position",
                             quote=snippet) for n, snippet in unresolved[:max_report]]
        return ctx.warn(
            "citation_resolution", "Citations resolve to the bibliography",
            f"{len(unresolved)} of {len(seen)} distinct numeric citation(s) do not resolve to any "
            f"of the {len(entries)} bibliography entries.",
            category="references", evidence=evidence,
            remedy="Check that no entry was renumbered or deleted after the in-text citations "
            "were written, and that the citation list was not truncated.",
            confidence="medium — numeric bibliography numbering is inferred from list position",
        )

    citations = extract_author_year_citations(body)
    if not citations:
        return ctx.skip("citation_resolution", "Citations resolve to the bibliography",
                        "No author-year in-text citations were detected in the body text.",
                        category="references")

    unique: dict[tuple[str, str], _Citation] = {}
    for c in citations:
        unique.setdefault((c.surname.lower(), c.year), c)

    unresolved_cites = [c for c in unique.values() if not _resolves(c, entries)]
    if not unresolved_cites:
        return ctx.ok(
            "citation_resolution", "Citations resolve to the bibliography",
            f"All {len(unique)} distinct in-text citation(s) resolve to one of the "
            f"{len(entries)} bibliography entries.",
            category="references",
        )

    unresolved_cites.sort(key=lambda c: c.start)
    evidence = [
        Evidence(page=_find_page(ctx, c.surname, c.year),
                 detail=f'citation "{c.surname} {c.year}" has no matching bibliography entry',
                 quote=c.context)
        for c in unresolved_cites[:max_report]
    ]
    return ctx.warn(
        "citation_resolution", "Citations resolve to the bibliography",
        f"{len(unresolved_cites)} of {len(unique)} distinct in-text citation(s) do not resolve to "
        f"any of the {len(entries)} bibliography entries: "
        f"{', '.join(f'{c.surname} {c.year}' for c in unresolved_cites[:8])}"
        f"{'...' if len(unresolved_cites) > 8 else ''}.",
        category="references",
        evidence=evidence,
        remedy="Check whether the cited entry was deleted or renamed while editing the "
        "bibliography, or whether the year/suffix in the citation is a typo.",
        confidence="medium — matched on (surname, year) against entry text; a wrong surname "
        "parse could hide a real citation, but this reports precise matches only",
    )


@register("uncited_references", "Uncited bibliography entries",
          module=MODULE, category="references", order=65)
def check_uncited_references(ctx: CheckContext) -> Finding:
    """Bibliography entries that are never cited anywhere in the document."""
    raw_entries = segment(ctx, bibliography_lines(ctx))
    if not raw_entries:
        return ctx.skip("uncited_references", "Uncited bibliography entries",
                        "No bibliography was found to check.", category="references")

    entries = _index_entries(raw_entries)
    search_text = _document_minus_bibliography(ctx)
    max_report = int(ctx.conf("citations.max_reported", 15))

    if detect_numeric_style(search_text, len(entries)):
        cited_indices = {n for _, numbers in _numeric_citations(search_text, len(entries))
                         for n in numbers}
        uncited = [e for e in entries if e.index not in cited_indices]
    else:
        citations = extract_author_year_citations(search_text)
        uncited = [
            e for e in entries
            if not _cited_by_index(e, citations) and not _cited_anywhere(e, search_text)
        ]
    if not uncited:
        return ctx.ok(
            "uncited_references", "Uncited bibliography entries",
            f"All {len(entries)} bibliography entries are cited somewhere in the document.",
            category="references",
        )

    evidence = [
        Evidence(detail=f"entry {e.index} ({e.surname}"
                        f"{', ' + e.year if e.year else ''}) never appears in the document text",
                 quote=e.raw[:160])
        for e in uncited[:max_report]
    ]
    return ctx.warn(
        "uncited_references", "Uncited bibliography entries",
        f"{len(uncited)} of {len(entries)} bibliography entries are never cited in the body, "
        "appendix, or captions. This is usually a leftover from editing rather than misconduct, "
        "but a bloated or stale reference list can draw reviewer attention.",
        category="references",
        evidence=evidence,
        remedy="Remove entries that are no longer cited, or add the citation if the omission "
        "was accidental.",
        confidence="medium — an entry cited only under a very different name form (e.g. an "
        "alias or translated title) could be missed",
    )
