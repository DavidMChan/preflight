"""Turning a bibliography into structured references.

Two steps. Segmentation is done from the PDF's own layout — bibliographies have
a hanging indent, so a line starting at the column edge begins a new entry — and
that is far more reliable than regexes on text that also contains the anonymous
template's line numbers.

Parsing each entry is then handed to a model in concurrent batches. Citation
styles vary enough that a regex parser either misses fields or invents them, and
a wrong title is worse than no title: it turns into a false accusation later.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ..context import CheckContext
from ..document import Line
from ..llm.client import AsyncLLMClient, LLMError
from .core import Reference, compact, find_pages, normalise_title, normalise_url, parse_page_range

PARSE_SYSTEM = (
    "You parse bibliography entries into structured records. You are precise and literal: you "
    "copy what the entry says and never guess, correct, complete or invent a field. If a field "
    "is not present in the entry, return null for it. The entries come from PDF text extraction, "
    "so expect broken spacing and hyphenation; recover the intended text where it is obvious, but "
    "never fabricate."
)

_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "the entry number given in the prompt"},
                    "title": {"type": ["string", "null"]},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "year": {"type": ["integer", "null"]},
                    "venue": {"type": ["string", "null"]},
                    "doi": {"type": ["string", "null"]},
                    "arxiv_id": {"type": ["string", "null"]},
                    "url": {"type": ["string", "null"]},
                    "pages": {"type": ["string", "null"],
                              "description": "the page range as printed, e.g. 28-40"},
                    "kind": {
                        "type": "string",
                        "enum": ["paper", "preprint", "book", "thesis", "software",
                                 "model", "blog", "standard", "web", "unknown"],
                    },
                },
                "required": ["index", "title", "authors", "year", "venue", "doi",
                             "arxiv_id", "url", "pages", "kind"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["references"],
    "additionalProperties": False,
}

_KIND_GUIDE = (
    "For `kind`, classify what is being cited: `paper` for anything published in a venue or "
    "journal; `preprint` for arXiv and similar; `book`; `thesis`; `software` for a repository or "
    "package; `model` for a model or dataset card (Hugging Face, Kaggle); `blog` for vendor "
    "announcements, blog posts and news articles; `standard` for RFCs, ISO documents, legislation "
    "and technical reports; `web` for any other web page; `unknown` if you cannot tell."
)


def line_number_pattern(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


def bibliography_lines(ctx: CheckContext) -> list[Line]:
    """Lines between the bibliography heading and the next section heading.

    The end bound is the *first heading of any kind* after the references, not a
    heading literally called "Appendix". Plenty of papers letter their appendix
    sections ("A Main Experiments", "C Sample trajectory"), and looking only for
    the word "Appendix" then runs the bibliography to the end of the document —
    sweeping in every table and prompt listing as though each were a citation.
    """
    start = ctx.doc.find_heading(
        [str(s) for s in (ctx.conf("structure.references_aliases", ["References", "Bibliography"]) or [])]
    )
    if start is None:
        return []
    end = next((h for h in ctx.doc.headings if h.order > start.order), None)

    ignore = line_number_pattern(ctx)
    running = running_lines(ctx)
    out: list[Line] = []
    started = False
    for line in ctx.doc.reading_order:
        if ignore.match(line.text.strip()):
            continue          # the anonymous template numbers every line
        if _running_key(line) in running and _in_margin_band(ctx, line):
            continue          # a running header or footer, not part of any entry
        if not started:
            started = line.page == start.page and abs(line.bbox[1] - start.bbox[1]) < 1.0
            continue
        if end is not None and line.page == end.page and abs(line.bbox[1] - end.bbox[1]) < 1.0:
            break
        out.append(line)
    return out


def _running_key(line: Line) -> str:
    return re.sub(r"\d+", "#", " ".join(line.text.lower().split()))


def _in_margin_band(ctx: CheckContext, line: Line, band: float = 0.12) -> bool:
    page = ctx.doc.pages[line.page - 1] if 0 < line.page <= len(ctx.doc.pages) else None
    if page is None or not page.height:
        return False
    return line.bbox[3] < page.height * band or line.bbox[1] > page.height * (1 - band)


def running_lines(ctx: CheckContext, pages: int = 3) -> set[str]:
    """Text repeated near the top or bottom of several pages: headers and footers.

    A review copy's footer ("Confidential reviewer copy. This manuscript is
    submitted to ...") otherwise lands inside whichever entry it interrupts, and
    its conference name then reads as that entry's venue.
    """
    seen: dict[str, set[int]] = {}
    for line in ctx.doc.reading_order:
        if len(line.text.strip()) >= 12 and _in_margin_band(ctx, line):
            seen.setdefault(_running_key(line), set()).add(line.page)
    return {key for key, where in seen.items() if len(where) >= pages}


#: Characters that cannot end a URL, so a line ending in one was certainly cut
#: mid-URL. "." and ":" are deliberately absent: they end citation sentences at
#: least as often as they appear mid-path.
_URL_CONNECTORS = "/-_=&?%~#"


_DOI_TAIL = re.compile(r"(?i)^(?:doi:)?10\.\d{4,9}/")


def _continues_url(tail: str, nxt: str) -> bool:
    """Whether a line ended part-way through a URL or DOI, and the next line resumes it.

    Long URLs and DOIs get broken across lines and columns; joining the halves
    with a space produces an identifier that points nowhere, and treating a
    hyphen at the break as hyphenation deletes a character that belongs to it.
    But a line can also break exactly where a URL ends, and gluing the
    following sentence onto it is just as wrong — so the next line has to look
    like a continuation, not new prose or a second link.
    """
    low = tail.lower()
    is_url = "://" in low or low.startswith(("www.", "//")) or low.endswith(("http:", "https:"))
    if not (is_url or _DOI_TAIL.match(low)):
        return False
    if nxt.lower().startswith(("http", "www.", "url", "doi")):
        return False                      # a new link, not the rest of this one
    if tail.endswith(tuple(_URL_CONNECTORS)):
        return True                       # cannot end a URL: definitely mid-path
    head = nxt[:1]
    return bool(head) and (head.islower() or head.isdigit() or head in _URL_CONNECTORS)


def _join(lines: list[Line]) -> str:
    text = ""
    for line in lines:
        piece = " ".join(line.text.split())
        if not piece:
            continue
        if not text:
            text = piece
        elif _continues_url(text.rsplit(" ", 1)[-1], piece):
            text += piece                        # the line broke inside a URL or DOI
        elif text.endswith("-"):
            text = text[:-1] + piece            # end-of-line hyphenation
        else:
            text = f"{text} {piece}"
    return text.strip()


_NUMERIC_LABEL = re.compile(r"^\s*\[\d{1,3}\]")


def _rows(lines: list[Line], lefts: list[float]) -> list[Line]:
    """Put the pieces of each printed row back in left-to-right order.

    A row set in two fonts arrives as separate lines, and an italic span's
    baseline can sit a fraction of a point above its roman neighbours, so
    sorting by height puts "Intention, Plans, and Practical Reason." ahead of
    the "[62] Michael E. Bratman." it follows — and onto the previous entry.
    """
    def column(line: Line) -> int:
        return max((i for i, left in enumerate(lefts) if line.bbox[0] >= left - 3.0), default=0)

    out: list[Line] = []
    row: list[Line] = []
    for line in lines:
        if row:
            anchor = row[0]
            height = min(anchor.bbox[3] - anchor.bbox[1], line.bbox[3] - line.bbox[1]) or 1.0
            centre = (line.bbox[1] + line.bbox[3]) / 2
            anchor_centre = (anchor.bbox[1] + anchor.bbox[3]) / 2
            if (line.page == anchor.page and column(line) == column(anchor)
                    and abs(centre - anchor_centre) < 0.4 * height):
                row.append(line)
                continue
            out.extend(sorted(row, key=lambda item: item.bbox[0]))
        row = [line]
    out.extend(sorted(row, key=lambda item: item.bbox[0]))
    return out


def segment(ctx: CheckContext, lines: list[Line]) -> list[str]:
    """Group lines into entries using the bibliography's hanging indent."""
    if not lines:
        return []
    tolerance = float(ctx.conf("refcheck.entry_indent_tolerance_pt", 2.5))
    lefts = [band[0] for band in ctx.doc.column_bands] or [min(line.bbox[0] for line in lines)]

    entries: list[list[Line]] = []
    current: list[Line] = []
    for line in _rows(lines, sorted(lefts)):
        offset = min(abs(line.bbox[0] - left) for left in lefts)
        # IEEE right-aligns the numeric label, so "[1]" through "[9]" begin one
        # digit's width inside the column edge that "[10]" sits on. A line that
        # opens with a bracketed number close to the edge starts an entry too.
        starts_entry = offset <= tolerance or (
            offset <= 4 * tolerance and _NUMERIC_LABEL.match(line.text) is not None
        )
        if starts_entry and current:
            entries.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(current)

    minimum = int(ctx.conf("refcheck.min_entry_chars", 25))
    return [
        text for text in (_join(entry) for entry in entries)
        if len(text) >= minimum and looks_like_entry(text)
    ]


# The trailing letter is ACL's disambiguation suffix: "2024a", "2024b".
_YEAR = re.compile(r"\b(?:19|20)\d{2}[a-z]?\b")
_LOCATOR = re.compile(r"(?i)https?://|www\.|\bdoi\b|\barxiv\b|\bisbn\b|\bin proceedings\b|\bpages?\b")


def looks_like_entry(text: str) -> bool:
    """Whether a segment plausibly is a citation rather than stray page content.

    Even with the right region, a bibliography's last column can trail into a
    figure or a table. Two signals separate them: a citation carries a year or a
    locator, and it is mostly words where a table row is mostly numbers.

    The bar is deliberately low on length. "xAI. 2025. Grok-4.3. model-card.pdf."
    is a real citation, and dropping it silently would be a worse failure than
    letting through the occasional stray line.
    """
    words = text.split()
    if len(words) < 3:
        return False
    alphabetic = sum(1 for w in words if any(c.isalpha() for c in w))
    if alphabetic / len(words) < 0.5:
        return False        # mostly numbers: a table row, not a reference
    return bool(_YEAR.search(text) or _LOCATOR.search(text))


def quick_parse(raw: str, index: int) -> Reference:
    """A structural first pass, so identifiers survive even if the model call fails."""
    reference = Reference(raw=raw, index=index)
    reference.scavenge()
    return reference


async def parse_entries(
    ctx: CheckContext,
    entries: list[str],
    client: AsyncLLMClient,
    batch_size: int = 20,
) -> list[Reference]:
    """Parse every entry, in concurrent batches. Falls back to the quick parse."""
    references = [quick_parse(raw, i) for i, raw in enumerate(entries)]
    if not references or not client.available:
        for reference in references:
            reference.classify_kind()
        return references

    batches = [references[i : i + batch_size] for i in range(0, len(references), batch_size)]
    results = await asyncio.gather(*(_parse_batch(batch, client) for batch in batches),
                                   return_exceptions=True)
    for batch, outcome in zip(batches, results, strict=False):
        if isinstance(outcome, BaseException) or not isinstance(outcome, dict):
            continue
        for reference in batch:
            _apply(reference, outcome.get(reference.index))

    for reference in references:
        reference.scavenge()
        reference.classify_kind()
    return references


async def _parse_batch(batch: list[Reference], client: AsyncLLMClient) -> dict[int, dict[str, Any]]:
    # The entry id must not look like a citation label: an IEEE entry already
    # starts with "[81]", and "[80] [81] ..." invites the model to answer with
    # the wrong number, which files one entry's fields under its neighbour.
    listing = "\n\n".join(f"ENTRY {r.index}:\n{r.raw}" for r in batch)
    prompt = (
        "Parse each bibliography entry below into a structured record. Return one record per "
        "entry. `index` is the number after the word ENTRY, not any bracketed label the citation "
        "itself begins with.\n\n"
        f"{_KIND_GUIDE}\n\n"
        "The `title` is the title of the work being cited, not the venue or the container "
        "publication. Return authors as they are written, in order, including a trailing "
        "'et al.' if the entry has one. `pages` is the page range as printed. Never invent a "
        "DOI, an arXiv id, pages or a URL that is not in the entry.\n\n"
        f"--- ENTRIES ---\n{listing}\n--- END ---"
    )
    try:
        data = await client.structured(PARSE_SYSTEM, prompt, schema=_ENTRY_SCHEMA, name="references")
    except LLMError:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for item in (data.get("references") or []):
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            out[item["index"]] = item
    return out


_RANGE = re.compile(r"[A-Za-z]?\d+\s*(?:-{1,3}|–|—)\s*[A-Za-z]?\d+")


def _printed_pages(raw: str, pages: str) -> bool:
    """Whether a parsed page range is really printed as pages in the entry.

    "Sociological Methods and Research, 54, 2025" has a volume and no pages; a
    lone number is only a page when the entry labels it as one.
    """
    wanted = parse_page_range(pages)
    if wanted is None:
        return False
    labelled = find_pages(raw)
    if labelled and parse_page_range(labelled) == wanted:
        return True
    printed = {parse_page_range(m.group(0)) for m in _RANGE.finditer(raw)}
    return wanted[1] is not None and wanted in printed


def _belongs(reference: Reference, parsed: dict[str, Any]) -> bool:
    """Whether a parsed record came from this entry's text, not a neighbour's."""
    title = (parsed.get("title") or "").strip()
    if not title:
        return True
    words = [w for w in normalise_title(title).split() if len(w) > 2]
    if not words:
        return True
    raw = compact(reference.raw)
    return compact(title) in raw or sum(w in raw for w in words) / len(words) >= 0.8


def _apply(reference: Reference, parsed: dict[str, Any] | None) -> None:
    if not parsed or not _belongs(reference, parsed):
        return
    from .core import Kind

    title = (parsed.get("title") or "").strip()
    if title:
        reference.title = title
    authors = [str(a).strip() for a in (parsed.get("authors") or []) if str(a).strip()]
    if authors:
        reference.authors = authors
    if isinstance(parsed.get("year"), int):
        reference.year = parsed["year"]
    for field in ("venue", "doi", "arxiv_id"):
        value = parsed.get(field)
        if isinstance(value, str) and value.strip():
            setattr(reference, field, value.strip())
    pages = parsed.get("pages")
    if isinstance(pages, str) and _printed_pages(reference.raw, pages):
        reference.pages = pages.strip()
    url = parsed.get("url")
    if isinstance(url, str) and url.strip():
        reference.url = normalise_url(url) or reference.url
    kind = parsed.get("kind")
    if isinstance(kind, str):
        try:
            reference.kind = Kind(kind)
        except ValueError:
            pass
