"""Advisory surface checks: abstract sanity, title/body agreement, number consistency.

Everything here is deterministic (counting, regex, set membership) and every finding is a
WARNING -- these are pointers for a human to look at, not provable defects. A check that
fires on a competent paper is worse than no check, so each one is written to require real,
repeated absence of support before it speaks.
"""

from __future__ import annotations

import re
import string

from ..analysis import abstract_text, body_text, conclusion_text, title_text
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.surface"

# ---------------------------------------------------------------------------
# abstract_present
# ---------------------------------------------------------------------------

_DEFAULT_MIN_WORDS = 80
_DEFAULT_MAX_WORDS = 300


@register("abstract_present", "Abstract", module=MODULE, category="surface", order=40)
def check_abstract_present(ctx: CheckContext) -> Finding:
    """Is there an abstract, and is its length within the venue's usual range."""
    text = abstract_text(ctx).strip()
    if not text:
        return ctx.skip(
            "abstract_present",
            "Abstract",
            "No 'Abstract' heading was found, so length cannot be checked.",
            category="surface",
        )

    words = len(text.split())
    min_words = int(ctx.conf("surface.abstract_min_words", _DEFAULT_MIN_WORDS))
    max_words = int(ctx.conf("surface.abstract_max_words", _DEFAULT_MAX_WORDS))
    evidence = [Evidence(detail="abstract word count", measured=float(words),
                         expected=f"{min_words}-{max_words} words (a profile setting, not a universal rule)")]

    if words < min_words:
        return ctx.warn(
            "abstract_present",
            "Abstract",
            f"The abstract is {words} words, under the {min_words}-word floor this profile is "
            "configured with. That bound is a house style, not a rule every venue shares -- "
            "confirm the target venue does not expect more before padding it.",
            category="surface",
            evidence=evidence,
        )
    if words > max_words:
        return ctx.warn(
            "abstract_present",
            "Abstract",
            f"The abstract is {words} words, over the {max_words}-word ceiling this profile is "
            "configured with. Some venues enforce an abstract word limit at submission; check "
            "the call for papers before trimming.",
            category="surface",
            evidence=evidence,
        )
    return ctx.ok(
        "abstract_present",
        "Abstract",
        f"Abstract found, {words} words, within the configured {min_words}-{max_words} range.",
        category="surface",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# title_support
# ---------------------------------------------------------------------------

_DEFAULT_STOPWORDS = [
    "a", "an", "the", "of", "for", "and", "or", "nor", "but", "with", "without", "via",
    "using", "use", "to", "in", "on", "at", "by", "from", "as", "is", "are", "be", "into",
    "toward", "towards", "about", "over", "under", "than", "that", "this", "these", "those",
    "how", "what", "when", "where", "why", "which", "who", "whom", "not", "do", "does",
    "your", "our", "its", "it", "new", "study", "studies", "paper", "papers", "case", "part",
]

_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z]*|[a-z]+|\d+")


def _subterms(token: str) -> list[str]:
    """Split a hyphenated or camelCase title token into its component words."""
    token = token.strip(string.punctuation)
    if not token:
        return []
    out: list[str] = []
    for part in re.split(r"[-/]", token):
        if not part:
            continue
        camel = _CAMEL_RE.findall(part)
        out.extend(camel if len(camel) > 1 else [part])
    return [p for p in out if p]


def _stem(term: str) -> str:
    """Crude singular/plural fold: 'agents' and 'agent' should count as the same term."""
    low = term.lower()
    if len(low) > 3 and low.endswith("s") and not low.endswith("ss"):
        return low[:-1]
    return low


def _count_in_body(body_lower: str, term: str) -> int:
    stem = re.escape(_stem(term))
    pattern = re.compile(rf"\b{stem}e?s?\b")
    return len(pattern.findall(body_lower))


@register("title_support", "Title support", module=MODULE, category="surface", order=41)
def check_title_support(ctx: CheckContext) -> Finding:
    """Every load-bearing word of the title should be traceable somewhere in the body."""
    stopwords = {str(w).lower() for w in ctx.conf("surface.title_stopwords", _DEFAULT_STOPWORDS)}
    min_len = int(ctx.conf("surface.title_min_term_length", 3))
    min_occurrences = int(ctx.conf("surface.title_min_occurrences", 2))

    title_line = title_text(ctx)
    if not title_line or len(title_line.split()) < 2:
        return ctx.skip(
            "title_support",
            "Title support",
            "Could not confidently locate a title line on page 1.",
            category="surface",
        )

    # Titles of the form "Rhetorical hook: Formal description" are common; the hook is
    # decorative and its wording (often a pun or a colloquialism) is not expected to
    # recur in formal prose. Only the part after the last colon carries the paper's
    # load-bearing vocabulary, so that is what gets checked.
    scoped_title = title_line.rsplit(":", 1)[-1].strip() if ":" in title_line else title_line

    body = body_text(ctx)
    body_lower = body.lower()

    unsupported: list[Evidence] = []
    checked = 0
    for raw in scoped_title.split():
        cleaned = raw.strip(string.punctuation)
        if not cleaned or cleaned.lower() in stopwords or len(cleaned) < min_len:
            continue
        checked += 1
        subterms = _subterms(cleaned) or [cleaned]
        # A hyphenated/camelCase title term is supported if the whole token is used
        # again, or if every one of its parts shows up somewhere in the body -- either
        # is enough evidence the concept is not a title-only flourish.
        whole_count = _count_in_body(body_lower, cleaned.replace("-", "").replace("/", ""))
        part_counts = [_count_in_body(body_lower, p) for p in subterms if len(p) >= min_len]
        # The body excludes the title block, so the title's own use is added back.
        best = (max([whole_count, *part_counts]) if part_counts else whole_count) + 1
        if best < min_occurrences:
            unsupported.append(
                Evidence(
                    detail=f"title term '{cleaned}'",
                    quote=scoped_title,
                    measured=float(best),
                    expected=f">= {min_occurrences} occurrences in the body (title use counts as one)",
                )
            )

    if checked == 0:
        return ctx.skip(
            "title_support",
            "Title support",
            "The title contained no terms long enough to check after removing stopwords.",
            category="surface",
        )

    if unsupported:
        terms = ", ".join(f"'{e.detail.split(chr(39))[1]}'" for e in unsupported[:6])
        return ctx.warn(
            "title_support",
            "Title support",
            f"{len(unsupported)} title term(s) barely appear in the body beyond the title itself: "
            f"{terms}. This can be a real gap, or just a coined name, an acronym expanded once, or a "
            "word phrased differently in prose -- worth a skim, not necessarily a fix.",
            category="surface",
            evidence=unsupported[:8],
        )
    return ctx.ok(
        "title_support",
        "Title support",
        f"All {checked} load-bearing title term(s) recur in the body.",
        category="surface",
    )


# ---------------------------------------------------------------------------
# abstract_conclusion_numbers
# ---------------------------------------------------------------------------

# Percentages and decimals only: plain small integers ("3 datasets", "two models"),
# years, section/table/citation numbers are noise, not claims, and are deliberately
# left unmatched by this pattern.
_CLAIM_NUMBER_RE = re.compile(r"(?<![\w./])(\d+\.\d+%?|\d+%)(?![\w%])")


def _is_version_like(text: str, match: re.Match[str], lookback_chars: int) -> bool:
    """Rule out model/version numbers such as 'GPT-4.5' or 'Haiku 4.5'.

    A decimal directly glued to a hyphen ('GPT-4.5') or immediately preceded by a
    capitalized word ('Haiku 4.5', 'Gemini 3.0 Flash') names a model, not a result.
    """
    start = match.start()
    before = text[:start]
    if before.endswith("-"):
        return True
    window = before.rstrip()[-lookback_chars:] if before.rstrip() else ""
    prev_word = window.split()[-1:] if window else []
    if prev_word:
        word = prev_word[0].strip(string.punctuation)
        if word and word[0].isupper() and word.lower() not in {"section", "table", "figure"}:
            return True
    return False


def _interesting_numbers(text: str, lookback_chars: int) -> set[str]:
    out: set[str] = set()
    for m in _CLAIM_NUMBER_RE.finditer(text):
        if _is_version_like(text, m, lookback_chars):
            continue
        out.add(m.group(1))
    return out


@register(
    "abstract_conclusion_numbers",
    "Abstract and conclusion agreement",
    module=MODULE,
    category="surface",
    order=42,
)
def check_abstract_conclusion_numbers(ctx: CheckContext) -> Finding:
    """Numeric claims (percentages, decimal scores) should line up across the paper."""
    lookback_chars = int(ctx.conf("surface.number_model_name_lookback_chars", 24))

    abstract = abstract_text(ctx)
    conclusion = conclusion_text(ctx)
    if not abstract.strip():
        return ctx.skip(
            "abstract_conclusion_numbers",
            "Abstract and conclusion agreement",
            "No abstract found, so there is nothing to cross-check.",
            category="surface",
        )

    abstract_nums = _interesting_numbers(abstract, lookback_chars)
    if not abstract_nums:
        return ctx.ok(
            "abstract_conclusion_numbers",
            "Abstract and conclusion agreement",
            "The abstract makes no percentage or decimal-score claims to cross-check.",
            category="surface",
        )

    conclusion_nums = _interesting_numbers(conclusion, lookback_chars) if conclusion.strip() else set()
    body_lower = body_text(ctx).lower()

    mismatches: list[Evidence] = []
    if conclusion.strip():
        only_abstract = sorted(abstract_nums - conclusion_nums)
        only_conclusion = sorted(conclusion_nums - abstract_nums)
        for n in only_abstract:
            mismatches.append(
                Evidence(
                    detail=f"'{n}' appears in the abstract but not the conclusion",
                    expected="verify these line up, or that the conclusion isn't just phrasing it differently",
                )
            )
        for n in only_conclusion:
            mismatches.append(
                Evidence(
                    detail=f"'{n}' appears in the conclusion but not the abstract",
                    expected="verify these line up, or that the abstract isn't just phrasing it differently",
                )
            )

    orphans: list[Evidence] = []
    for n in sorted(abstract_nums):
        count = len(re.findall(re.escape(n), body_lower, flags=re.IGNORECASE))
        if count <= 1:  # the abstract's own occurrence is the only one
            orphans.append(
                Evidence(
                    detail=f"'{n}' is claimed in the abstract but does not recur anywhere else in the body",
                    quote=abstract.strip()[:160],
                )
            )

    evidence = mismatches[:6] + orphans[:6]
    if evidence:
        return ctx.warn(
            "abstract_conclusion_numbers",
            "Abstract and conclusion agreement",
            f"{len(evidence)} numeric claim(s) in the abstract don't obviously line up elsewhere in the "
            "paper. This is a pointer, not proof -- numbers get restated with different precision or "
            "wording all the time. Worth a quick skim, not an automatic edit.",
            category="surface",
            evidence=evidence,
        )
    return ctx.ok(
        "abstract_conclusion_numbers",
        "Abstract and conclusion agreement",
        f"{len(abstract_nums)} numeric claim(s) in the abstract are echoed elsewhere in the paper.",
        category="surface",
    )
