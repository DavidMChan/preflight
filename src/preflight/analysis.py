"""Structural views over a parsed paper.

Checks that reason about prose, captions or equations need the same handful of
derived views, and the LLM checks need one *byte-identical* excerpt so the
provider can serve a long shared prefix from cache. Both live here so there is
exactly one implementation of each.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .context import CheckContext
from .document import Heading

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

_ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "cf.",
    "et al.",
    "vs.",
    "etc.",
    "Fig.",
    "Eq.",
    "Sec.",
    "Tab.",
    "Ref.",
    "Refs.",
    "Alg.",
    "App.",
    "approx.",
    "resp.",
    "Dr.",
    "Prof.",
    "Mr.",
    "Ms.",
}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\\$])")
_CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table|Algorithm|Listing)\s+([A-Z]?\.?\d+(?:\.\d+)?)\s*[:.]\s*(.*)$")
_MATH_HINT = re.compile(r"[=+\-−×·∑∏∫√≈≤≥≠∈∇∂]|\\[a-zA-Z]+|\b[a-zA-Z]_\{?\d")


def _line_number_re(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


def clean_text(ctx: CheckContext, text: str) -> str:
    """Drop margin line numbers and rejoin words hyphenated across line breaks."""
    pattern = _line_number_re(ctx)
    lines = [line for line in text.splitlines() if not pattern.match(line)]
    joined = "\n".join(lines)
    joined = re.sub(r"(\w)-\n(\w)", r"\1\2", joined)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def full_text(ctx: CheckContext) -> str:
    """The whole paper, cleaned. Cached for the run."""
    cached = ctx.shared.get("analysis_full_text")
    if isinstance(cached, str):
        return cached
    text = clean_text(ctx, ctx.doc.text)
    ctx.shared["analysis_full_text"] = text
    return text


def _first_unlimited_heading(ctx: CheckContext) -> Heading | None:
    """Where the page-limited part of the paper stops."""
    keys = {
        "limitations": "structure.limitations_aliases",
        "references": "structure.references_aliases",
        "acknowledgments": "structure.acknowledgments_aliases",
        "ethics": "structure.ethics_aliases",
        "appendix": "structure.appendix_aliases",
    }
    found = []
    for kind, conf_key in keys.items():
        aliases = [str(a) for a in (ctx.conf(conf_key, [kind.capitalize()]) or [])]
        heading = ctx.doc.find_heading(aliases)
        if heading is not None:
            found.append(heading)
    return min(found, key=lambda h: h.order) if found else None


def body_text(ctx: CheckContext) -> str:
    """Main content only: everything before the references and appendices.

    Prose diagnostics must not be run over the bibliography, where "sentences"
    are citation strings and every third token is a proper noun.
    """
    cached = ctx.shared.get("analysis_body_text")
    if isinstance(cached, str):
        return cached
    stop = _first_unlimited_heading(ctx)
    if stop is None:
        text = full_text(ctx)
    else:
        lines = []
        for line in ctx.doc.reading_order:
            if line.page == stop.page and abs(line.bbox[1] - stop.bbox[1]) < 0.6:
                break
            lines.append(line.text)
        text = clean_text(ctx, "\n".join(lines))
    ctx.shared["analysis_body_text"] = text
    return text


def title_text(ctx: CheckContext) -> str:
    """The paper's title: the largest type on page 1, joined across its lines.

    The first line on the page is usually a running head ("Under review as a
    conference paper at ICLR 2027") or a margin line number, so position alone
    picks the wrong text. The title is the largest type on the page, and it
    may wrap, so consecutive lines at that size are joined.
    """
    cached = ctx.shared.get("analysis_title")
    if isinstance(cached, str):
        return cached
    ignore = _line_number_re(ctx)
    lines = [
        ln for ln in ctx.doc.reading_order
        if ln.page == 1 and ln.text.strip() and not ignore.match(ln.text.strip())
        and any(c.isalpha() for c in ln.text)
    ]
    title = ""
    if lines:
        largest = max(ln.heading_size for ln in lines)
        picked: list[str] = []
        for ln in sorted(lines, key=lambda ln: (ln.bbox[1], ln.bbox[0])):
            if abs(ln.heading_size - largest) < 0.5:
                picked.append(" ".join(ln.text.split()))
            elif picked:
                break
        title = " ".join(picked)
    ctx.shared["analysis_title"] = title
    return title


def section_text(ctx: CheckContext, aliases: list[str]) -> str:
    """Text of the first section matching ``aliases``, up to the next heading."""
    heading = ctx.doc.find_heading(aliases)
    if heading is None:
        return ""
    following = next((h for h in ctx.doc.headings if h.order > heading.order), None)
    return clean_text(ctx, ctx.doc.text_between(heading, following))


def abstract_text(ctx: CheckContext) -> str:
    return section_text(ctx, ["Abstract"])


def conclusion_text(ctx: CheckContext) -> str:
    for alias in ("Conclusion", "Conclusions", "Conclusion and Future Work", "Discussion"):
        text = section_text(ctx, [alias])
        if text:
            return text
    return ""


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Sentence:
    """One sentence of body prose, with what the rhythm checks need."""

    text: str
    index: int
    page: int | None = None

    @property
    def words(self) -> list[str]:
        return self.text.split()

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def opener(self) -> str:
        """First word, lowercased and stripped of punctuation."""
        words = self.words
        return re.sub(r"[^a-z]", "", words[0].lower()) if words else ""

    @property
    def has_internal_break(self) -> bool:
        """Whether the sentence offers a natural place to split."""
        return bool(re.search(r"[:;—]|\s--\s", self.text))


def sentences(ctx: CheckContext, text: str | None = None) -> list[Sentence]:
    """Split body prose into sentences.

    Deliberately conservative: display equations are treated as boundaries,
    common abbreviations do not end sentences, and citation-only fragments are
    dropped. Cached per run when reading the default body text.
    """
    use_cache = text is None
    if use_cache:
        cached = ctx.shared.get("analysis_sentences")
        if isinstance(cached, list):
            return cached
        text = body_text(ctx)

    out: list[Sentence] = []
    for block in re.split(r"\n\s*\n", text or ""):
        flat = " ".join(block.split())
        if not flat:
            continue
        pieces = _SENTENCE_END.split(flat)
        merged: list[str] = []
        for piece in pieces:
            if merged and any(merged[-1].endswith(a) for a in _ABBREVIATIONS):
                merged[-1] = f"{merged[-1]} {piece}"
            else:
                merged.append(piece)
        for piece in merged:
            piece = piece.strip()
            # Needs some real prose: at least a few alphabetic words.
            if len(piece) < 25 or sum(w.isalpha() for w in piece.split()) < 4:
                continue
            out.append(Sentence(text=piece, index=len(out)))

    if use_cache:
        ctx.shared["analysis_sentences"] = out
    return out


# ---------------------------------------------------------------------------
# Captions, tables, equations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Caption:
    """A figure/table caption as printed."""

    kind: str  # "figure" | "table" | "algorithm" | "listing"
    label: str  # "3", "B.2"
    text: str
    page: int
    word_count: int = 0
    numbers: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.kind.capitalize()} {self.label}"


def captions(ctx: CheckContext) -> list[Caption]:
    """Every figure/table/algorithm caption, in reading order."""
    cached = ctx.shared.get("analysis_captions")
    if isinstance(cached, list):
        return cached

    out: list[Caption] = []
    lines = ctx.doc.reading_order
    for i, line in enumerate(lines):
        flat = " ".join(line.text.split())
        match = _CAPTION_RE.match(flat)
        if not match:
            continue
        kind = match.group(1).lower().rstrip(".").replace("fig", "figure")
        body = match.group(3)
        # Captions wrap; absorb following lines until the next caption or a gap.
        number_re = _line_number_re(ctx)
        for follower in lines[i + 1 : i + 12]:
            nxt = " ".join(follower.text.split())
            if not nxt or _CAPTION_RE.match(nxt) or follower.page != line.page:
                break
            if number_re.match(nxt):
                continue  # a margin line number interleaved with the caption
            if abs(follower.bbox[1] - line.bbox[1]) > 60:
                break
            body += " " + nxt
        body = body.strip()
        out.append(
            Caption(
                kind=kind if kind in {"figure", "table", "algorithm", "listing"} else "figure",
                label=match.group(2),
                text=body,
                page=line.page,
                word_count=len(body.split()),
                numbers=re.findall(r"\d+(?:\.\d+)?", body),
            )
        )
    ctx.shared["analysis_captions"] = out
    return out


def numeric_tokens(text: str) -> list[str]:
    """Decimal numbers, for precision-consistency checks."""
    return re.findall(r"(?<![\w.])\d+\.\d+(?![\w.])", text)


def looks_like_math(text: str) -> bool:
    return bool(_MATH_HINT.search(text))


def equation_lines(ctx: CheckContext) -> list[tuple[int, str]]:
    """(page, text) for lines that look like display equations.

    A crude but useful filter: short, centred-ish lines carrying maths symbols.
    Intended to give an auditor something to point at, not to parse the maths.
    """
    cached = ctx.shared.get("analysis_equations")
    if isinstance(cached, list):
        return cached

    bands = ctx.doc.column_bands
    out: list[tuple[int, str]] = []
    for line in ctx.doc.reading_order:
        flat = " ".join(line.text.split())
        if not flat or len(flat) > 220 or not looks_like_math(flat):
            continue
        if sum(c.isalpha() for c in flat) > len(flat) * 0.75:
            continue  # mostly prose
        indented = bands and all(abs(line.bbox[0] - b0) > 8.0 for b0, _ in bands)
        if indented or re.match(r"^\(?\d+\)?\s*$", flat) is None and len(flat) < 160:
            out.append((line.page, flat))
    ctx.shared["analysis_equations"] = out
    return out


# ---------------------------------------------------------------------------
# The shared LLM prefix
# ---------------------------------------------------------------------------

_CONTEXT_HEADER = (
    "Below is the full text of an anonymous paper submission, extracted from its PDF. "
    "Extraction artifacts (hyphenation at line breaks, table text flattened into lines, "
    "figure captions inline) are expected and are not defects in the paper.\n\n"
    "=== BEGIN PAPER TEXT ===\n"
)
_CONTEXT_FOOTER = "\n=== END PAPER TEXT ===\n"


def paper_context(ctx: CheckContext) -> str:
    """The paper excerpt every model-backed check sends, byte for byte.

    Built once and reused verbatim. Dozens of independent calls then share one
    long prompt prefix, which the provider can serve from cache; the part that
    varies per check is appended afterwards and is short.
    """
    cached = ctx.shared.get("llm_paper_context")
    if isinstance(cached, str):
        return cached

    limit = int(ctx.conf("llm.max_context_chars", 1_000_000))
    body = full_text(ctx)
    if len(body) > limit:
        # Keep the front and the back: the front carries the claims and setup,
        # the back carries limitations, ethics, and the appendix detail that the
        # audits most often need. Record it so the run can say so out loud --
        # a model verdict on a paper it only half read must never look complete.
        head = int(limit * 0.6)
        omitted = len(body) - limit
        body = (
            body[:head]
            + f"\n\n[... {omitted} characters from the middle of the paper omitted for length ...]\n\n"
            + body[-(limit - head) :]
        )
        ctx.shared["llm_context_truncated"] = {
            "total_chars": len(full_text(ctx)),
            "kept_chars": limit,
            "omitted_chars": omitted,
        }

    context = _CONTEXT_HEADER + body + _CONTEXT_FOOTER
    ctx.shared["llm_paper_context"] = context
    return context
