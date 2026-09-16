"""Deterministic abbreviation hygiene: defined-before-use, and one-abbreviation-
one-meaning.

Both checks are counting exercises over the paper's own text, not a style
opinion, so they stay WARNING regardless -- an undefined acronym does not get
a paper desk-rejected. The defined-before-use check in particular is a false
positive minefield: NLP papers are dense with standard acronyms (LLM, BERT,
GPU...), model/dataset/benchmark names that look like abbreviations but are
proper nouns nobody is obliged to expand (BERT, MIND, GLUE), and metrics that
are conventionally never spelled out (F1, BLEU). The exemption list below is
the actual content of this module; a long finding list on a competent paper
means the exemptions are wrong, not that the paper is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..analysis import Sentence, captions, sentences
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.abbreviations"

# A candidate abbreviation: 2-6 characters, letters + optional digits, entirely
# upper-case, with an optional hyphenated digit suffix ("GPT-4", "T5-11B").
# Because every character but the hyphen must be upper-case, this already
# excludes ordinary capitalised words -- "The", "Results", a sentence-initial
# "We" -- without any extra bookkeeping.
_CANDIDATE_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]{1,3})?\b")

# The abbreviation half of "... (KGB)". The expansion is recovered separately,
# by walking backward word-by-word from here -- a single greedy regex for
# "words immediately before the paren" over-matches onto whatever capitalised
# text happens to open the sentence.
_ABBR_PAREN_RE = re.compile(r"\(\s*([A-Z][A-Z0-9]{1,5}(?:-[A-Z0-9]{1,3})?)\s*\)")
_WORD_RE = re.compile(r"[A-Za-z][a-zA-Z]*")

_DEFAULT_ALLOWLIST = [
    "NLP", "AI", "GPU", "CPU", "API", "URL", "PDF", "LLM", "LLMS", "SOTA", "MLP", "RNN", "CNN",
    "BERT", "GPT", "RL", "ML", "NLU", "NLG", "IR", "QA", "NLI", "OCR", "UI", "UX", "IT", "US", "UK",
    "EU", "ID", "TODO", "vs",
]

_DEFAULT_METRICS = [
    "F1", "BLEU", "ROUGE", "METEOR", "AUC", "ROC", "EM", "MAP", "MRR", "NDCG", "PPL", "WER", "CER",
    "MSE", "MAE", "RMSE", "GB", "MB", "KB", "TB", "MS", "GPU-HOURS", "FLOPS", "ACC", "STD",
]

# Words nearby that mark a token as a proper-noun name (dataset, benchmark,
# model, task) rather than a term the authors are expected to spell out.
_NAME_CONTEXT_RE = re.compile(
    r"\b(dataset|corpus|benchmark|model|task|challenge|leaderboard|framework|toolkit|library|"
    r"architecture)s?\b",
    re.IGNORECASE,
)


def _config_set(ctx: CheckContext, key: str, default: list[str]) -> set[str]:
    return {str(x).upper() for x in (ctx.conf(key, default) or default)}


def _initials_match(expansion: str, abbr: str) -> bool:
    """Loose check that ``abbr`` is plausibly an acronym of ``expansion``.

    Guards the definition regex against grabbing an unrelated parenthetical
    that just happens to sit next to a capitalised phrase.
    """
    words = [w for w in re.split(r"[\s-]+", expansion.strip()) if w]
    abbr_letters = re.sub(r"[^A-Z]", "", abbr.upper())
    if not words or not abbr_letters:
        return False
    initials = [w[0].upper() for w in words]
    if initials[0] != abbr_letters[0]:
        return False
    if len(initials) < len(abbr_letters) - 1:
        return False
    hits = sum(1 for a, b in zip(abbr_letters, initials, strict=False) if a == b)
    return hits >= max(1, len(abbr_letters) - 1)


def _base_letters(abbr: str) -> str:
    """The abbreviation without a trailing '-<digits>' variant suffix."""
    return abbr.split("-")[0]


def _normalize_expansion(expansion: str) -> str:
    """Case/plural/hyphenation-insensitive key for comparing two definitions."""
    norm = expansion.lower().strip()
    norm = re.sub(r"[\s-]+", " ", norm)
    norm = re.sub(r"s\b", "", norm)  # crude pluralisation strip, word by word
    return " ".join(w.rstrip("s") if len(w) > 3 else w for w in norm.split())


def _looks_like_quote_or_prompt(text: str) -> bool:
    """A quoted example or an inline prompt is not the authors' own defining prose."""
    if text.count('"') >= 2 or text.count("“") >= 1:
        return True
    return bool(re.search(r"[{}]|Prompt\s*:|Input\s*:|Output\s*:", text))


def _looks_like_table_row(text: str) -> bool:
    """A results table flattened into 'sentence'-shaped text by extraction.

    ``sentences()`` filters on word count, not layout, so a dense row of
    numbers ("0.99 0.96 ... Rewriter-Only 43.4 (+26.3)") can still pass its
    prose heuristic. Digit density and result-table glyphs are the tell.
    """
    if any(sym in text for sym in "✓✗±→"):
        return True
    digits = sum(c.isdigit() for c in text)
    return len(text) > 0 and digits / len(text) > 0.12


_CITATION_AFTER_RE = re.compile(r"^\s*\(\s*[A-Za-z][^()]{0,50}\d{4}[a-z]?\s*\)")

# Symbols that only occur in extracted mathematics. A capitalised token sitting
# next to one of these ("the KN k, Ckn) ∈ Rd") is a product of variables that
# extraction ran together, not an acronym. Ordinary hyphens are deliberately
# absent: prose is full of them.
_MATH_CONTEXT_RE = re.compile(r"[=≤≥∈∉∑∏∫√≈≠∇∂×−·]")
_MATH_WINDOW = 30
_LOWERCASE_WORD_RE = re.compile(r"\b[a-z]{2,}\b")


@dataclass(slots=True)
class _Def:
    abbr: str
    expansion: str
    sentence_index: int
    span: tuple[int, int]  # character offsets of the abbr token, within the sentence


def _sentence_definitions(text: str) -> list[_Def]:
    """Find every "Expansion (ABBR)" in ``text`` by walking back from each "(ABBR)"."""
    out = []
    for m in _ABBR_PAREN_RE.finditer(text):
        abbr = m.group(1)
        letters = re.sub(r"[^A-Z]", "", abbr.upper())
        if not letters:
            continue
        words = _WORD_RE.findall(text[: m.start()])
        if not words:
            continue
        # Try the word count that should match the abbreviation's letter count
        # first (one word per letter), then +/-1 for a dropped stopword or an
        # extra adjective the acronym skips.
        for n in (len(letters), len(letters) - 1, len(letters) + 1):
            if n <= 0 or n > len(words) or n > 7:
                continue
            expansion = " ".join(words[-n:])
            if _initials_match(expansion, abbr):
                out.append(_Def(abbr, expansion, -1, (m.start(1), m.end(1))))
                break
    return out


def _in_math(text: str, start: int, end: int) -> bool:
    return bool(_MATH_CONTEXT_RE.search(text[max(0, start - _MATH_WINDOW) : end + _MATH_WINDOW]))


def _shouted(text: str, start: int, end: int) -> bool:
    """True if the token is one word of a run of capitals.

    Small-caps headings ("EVALUATION OF REPRESENTATION STEERING") and prompt
    text set in capitals ("LENGTH BALANCED ACROSS LISTS") reach the extractor
    as ordinary upper-case words, and every short one of them looks like an
    acronym. Acronyms are not written in runs; capitalised prose is.
    """
    before = re.search(r"\b[A-Z]{2,}\s+$", text[:start])
    after = re.match(r"\s+[A-Z]{2,}\b", text[end:])
    return bool(before or after)


def _is_in_caption(sentence_text: str, caption_texts: list[str]) -> bool:
    flat = " ".join(sentence_text.split())
    return any(flat in cap or (len(flat) > 20 and cap in flat) for cap in caption_texts)


def _find_page(ctx: CheckContext, snippet: str) -> int | None:
    norm = " ".join(snippet.split())[:70]
    if not norm:
        return None
    for page in ctx.doc.pages:
        if norm in " ".join(page.text.split()):
            return page.number
    return None


@register("abbreviation_defined", "Abbreviations defined before use", module=MODULE,
          category="prose", order=60)
def check_abbreviation_defined(ctx: CheckContext) -> Finding:
    """Flag an acronym used with no expansion anywhere nearby its first use."""
    allow = _config_set(ctx, "abbreviations.allowlist", _DEFAULT_ALLOWLIST)
    metrics = _config_set(ctx, "abbreviations.metric_allowlist", _DEFAULT_METRICS)
    min_uses = int(ctx.conf("abbreviations.min_uses_to_flag", 2))
    cap = int(ctx.conf("abbreviations.max_reported", 5))
    exempt = allow | metrics

    sents = sentences(ctx)
    if not sents:
        return ctx.skip("abbreviation_defined", "Abbreviations defined before use",
                        "No extractable body prose to check.", category="prose")

    caption_texts = [" ".join(c.text.split()) for c in captions(ctx)]
    # A token whose lower-case form is used as a word elsewhere in the paper
    # ("NOT", "OF", "LENGTH") is a word set in capitals, not an abbreviation.
    ordinary_words = set(_LOWERCASE_WORD_RE.findall(" ".join(s.text for s in sents)))

    defined_at: dict[str, int] = {}
    first_use_at: dict[str, tuple[int, Sentence]] = {}
    counts: dict[str, int] = {}
    # A token seen even once right after a citation, or right next to "dataset" /
    # "benchmark" / "model" etc., is a proper noun for the rest of the paper --
    # authors do not re-cite or re-gloss a name at every later mention.
    established_name: set[str] = set()

    for sent in sents:
        text = sent.text
        if (_is_in_caption(text, caption_texts) or _looks_like_quote_or_prompt(text)
                or _looks_like_table_row(text)):
            continue

        defs = _sentence_definitions(text)
        consumed = [d.span for d in defs]
        for d in defs:
            defined_at.setdefault(d.abbr, sent.index)

        for m in _CANDIDATE_RE.finditer(text):
            token = m.group(0)
            if any(c[0] <= m.start() < c[1] for c in consumed):
                continue  # this occurrence *is* the definition, not a bare use
            base = _base_letters(token)
            if base in exempt or token in exempt:
                continue
            if token.lower() in ordinary_words or _shouted(text, m.start(), m.end()):
                continue  # a word in capitals, not an acronym
            if _in_math(text, m.start(), m.end()):
                continue  # variables that extraction ran together
            tail = text[m.end() : m.end() + 1]
            if tail == "-" and text[m.end() + 1 : m.end() + 2].isalpha():
                continue  # a fragment of a longer hyphenated proper noun ("EB-NeRD")
            if _CITATION_AFTER_RE.match(text[m.end():]):
                established_name.add(token)  # "MIND (Wu et al., 2020)" -- a cited resource
                continue
            window = text[max(0, m.start() - 25) : m.end() + 40]
            if _NAME_CONTEXT_RE.search(window):
                established_name.add(token)  # "the MIND dataset" / "MIND benchmark"
                continue
            counts[token] = counts.get(token, 0) + 1
            first_use_at.setdefault(token, (sent.index, sent))

    first_use_at = {t: v for t, v in first_use_at.items() if t not in established_name}

    offenders: list[Evidence] = []
    for token, (use_idx, sent) in first_use_at.items():
        if counts.get(token, 0) < min_uses:
            continue  # a single stray mention is not worth an author's time
        def_idx = defined_at.get(token)
        if def_idx is not None and def_idx <= use_idx:
            continue  # defined at or before its first bare use
        page = _find_page(ctx, sent.text[:70])
        offenders.append(
            Evidence(
                page=page,
                detail=f"'{token}' used {counts[token]}x in the body; no expansion found before "
                "its first use",
                quote=sent.text[:180],
            )
        )

    if not offenders:
        return ctx.ok(
            "abbreviation_defined", "Abbreviations defined before use",
            "Every recurring non-standard abbreviation found in the body text is either "
            "spelled out at first use or on the venue's standard-terms allowlist.",
            category="prose",
        )

    offenders.sort(key=lambda e: e.page or 0)
    return ctx.warn(
        "abbreviation_defined", "Abbreviations defined before use",
        f"{len(offenders)} abbreviation(s) recur in the body text without an expansion found "
        "before their first use. Standard NLP/ML terms, metrics, and proper nouns (model, "
        "dataset, and benchmark names) are exempted; this is only the residue.",
        category="prose",
        evidence=offenders[:cap],
        remedy="Spell out each abbreviation once, at its first use: \"large language model (LLM)\".",
        confidence="medium — a definition just outside the containing sentence, or a name this "
        "allowlist does not yet know, can still slip through as a false positive",
    )


@register("abbreviation_consistency", "Abbreviation consistency", module=MODULE,
          category="prose", order=61)
def check_abbreviation_consistency(ctx: CheckContext) -> Finding:
    """One abbreviation should map to exactly one expansion, defined exactly once."""
    sents = sentences(ctx)
    if not sents:
        return ctx.skip("abbreviation_consistency", "Abbreviation consistency",
                        "No extractable body prose to check.", category="prose")

    caption_texts = [" ".join(c.text.split()) for c in captions(ctx)]

    # abbr -> list of (normalized_expansion, original_expansion, sentence)
    by_abbr: dict[str, list[tuple[str, str, Sentence]]] = {}
    for sent in sents:
        if _is_in_caption(sent.text, caption_texts) or _looks_like_table_row(sent.text):
            continue
        for d in _sentence_definitions(sent.text):
            by_abbr.setdefault(d.abbr, []).append((_normalize_expansion(d.expansion), d.expansion, sent))

    conflicts: list[Evidence] = []
    duplicates: list[Evidence] = []
    for abbr, occurrences in by_abbr.items():
        norms = {n for n, _, _ in occurrences}
        if len(norms) > 1:
            forms = ", ".join(f'"{orig}"' for _, orig, _ in occurrences[:4])
            page = _find_page(ctx, occurrences[0][2].text[:70])
            conflicts.append(Evidence(
                page=page,
                detail=f"'{abbr}' is expanded {len(norms)} different ways: {forms}",
            ))
        elif len(occurrences) > 1:
            page = _find_page(ctx, occurrences[0][2].text[:70])
            duplicates.append(Evidence(
                page=page,
                detail=f"'{abbr}' ({occurrences[0][1]}) is spelled out {len(occurrences)} times",
            ))

    if not conflicts and not duplicates:
        return ctx.ok(
            "abbreviation_consistency", "Abbreviation consistency",
            "Every abbreviation defined in the body text is defined exactly once, with one "
            "consistent expansion.",
            category="prose",
        )

    parts = []
    if conflicts:
        parts.append(f"{len(conflicts)} abbreviation(s) are expanded inconsistently")
    if duplicates:
        parts.append(f"{len(duplicates)} are defined more than once with the same expansion")
    return ctx.warn(
        "abbreviation_consistency", "Abbreviation consistency",
        ", ".join(parts).capitalize() + ". Redundant or drifting definitions are usually a sign "
        "of text merged from different drafts.",
        category="prose",
        evidence=(conflicts + duplicates)[:8],
        remedy="Define each abbreviation once, at its first use, and reuse that exact expansion "
        "everywhere else (or just the abbreviation).",
        confidence="medium — a repeated definition after a major section break (e.g. restated for "
        "the appendix) is sometimes intentional",
    )
