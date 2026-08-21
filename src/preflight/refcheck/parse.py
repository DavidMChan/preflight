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
from .core import Reference, normalise_url

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
                    "kind": {
                        "type": "string",
                        "enum": ["paper", "preprint", "book", "thesis", "software",
                                 "model", "blog", "standard", "web", "unknown"],
                    },
                },
                "required": ["index", "title", "authors", "year", "venue", "doi",
                             "arxiv_id", "url", "kind"],
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
    out: list[Line] = []
    started = False
    for line in ctx.doc.reading_order:
        if ignore.match(line.text.strip()):
            continue          # the anonymous template numbers every line
        if not started:
            started = line.page == start.page and abs(line.bbox[1] - start.bbox[1]) < 1.0
            continue
        if end is not None and line.page == end.page and abs(line.bbox[1] - end.bbox[1]) < 1.0:
            break
        out.append(line)
    return out


#: Characters that cannot end a URL, so a line ending in one was certainly cut
#: mid-URL. "." and ":" are deliberately absent: they end citation sentences at
#: least as often as they appear mid-path.
_URL_CONNECTORS = "/-_=&?%~#"


def _continues_url(tail: str, nxt: str) -> bool:
    """Whether a line ended part-way through a URL, and the next line resumes it.

    Long URLs get broken across lines and columns; joining the halves with a
    space produces a URL that points nowhere. But a line can also break exactly
    where a URL ends, and gluing the following sentence onto it is just as
    wrong — so the next line has to look like a continuation, not new prose.
    """
    low = tail.lower()
    if not ("://" in low or low.startswith(("www.", "//")) or low.endswith(("http:", "https:"))):
        return False
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
        elif text.endswith("-"):
            text = text[:-1] + piece            # end-of-line hyphenation
        elif _continues_url(text.rsplit(" ", 1)[-1], piece):
            text += piece                        # the line broke inside a URL
        else:
            text = f"{text} {piece}"
    return text.strip()


def segment(ctx: CheckContext, lines: list[Line]) -> list[str]:
    """Group lines into entries using the bibliography's hanging indent."""
    if not lines:
        return []
    tolerance = float(ctx.conf("refcheck.entry_indent_tolerance_pt", 2.5))
    lefts = [band[0] for band in ctx.doc.column_bands] or [min(line.bbox[0] for line in lines)]

    entries: list[list[Line]] = []
    current: list[Line] = []
    for line in lines:
        if any(abs(line.bbox[0] - left) <= tolerance for left in lefts) and current:
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
    listing = "\n\n".join(f"[{r.index}] {r.raw}" for r in batch)
    prompt = (
        "Parse each bibliography entry below into a structured record. Return one record per "
        "entry, keeping the entry number in `index`.\n\n"
        f"{_KIND_GUIDE}\n\n"
        "The `title` is the title of the work being cited, not the venue or the container "
        "publication. Return authors as they are written. Never invent a DOI, an arXiv id or a "
        "URL that is not in the entry.\n\n"
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


def _apply(reference: Reference, parsed: dict[str, Any] | None) -> None:
    if not parsed:
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
    url = parsed.get("url")
    if isinstance(url, str) and url.strip():
        reference.url = normalise_url(url) or reference.url
    kind = parsed.get("kind")
    if isinstance(kind, str):
        try:
            reference.kind = Kind(kind)
        except ValueError:
            pass
