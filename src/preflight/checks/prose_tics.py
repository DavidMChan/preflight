"""Deterministic prose-style checks: connective overuse, vague demonstratives,
and hollow wrap-up sentences.

These are all counting exercises, not model judgements -- the skill this comes
from is explicit that the fix for a fuzzy stylistic complaint is to count the
thing instead of eyeballing it. All three are advisory (WARNING at worst) and
none of them may claim a competent paper is wrong to write the way it does:
a Related Work section is legitimately connective-dense, a demonstrative
followed by a noun is fine English, and a phrase quoted from an example or a
figure caption is not the authors' own prose.
"""

from __future__ import annotations

import re

from ..analysis import section_text, sentences
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.prose"

_QUOTE_RE = re.compile(r'["“”‘’]')

_DEFAULT_CONNECTIVES = [
    "furthermore",
    "moreover",
    "notably",
    "consequently",
    "additionally",
    "importantly",
    "therefore",
    "thus",
    "hence",
    "it is worth noting",
    "it should be noted",
]

_DEFAULT_DEMONSTRATIVE_VERBS = [
    "shows",
    "show",
    "suggests",
    "suggest",
    "indicates",
    "indicate",
    "implies",
    "imply",
    "means",
    "demonstrates",
    "demonstrate",
    "highlights",
    "highlight",
    "allows",
    "allow",
    "enables",
    "enable",
    "results",
    "result",
    "leads",
    "lead",
    "confirms",
    "confirm",
    "reveals",
    "reveal",
    "proves",
    "prove",
]

_DEFAULT_HOLLOW_WRAPUPS = [
    "together, these observations suggest",
    "together, these results suggest",
    "together, these results demonstrate",
    "these results highlight",
    "these findings highlight",
    "taken together, these findings",
    "taken together, these results",
    "this underscores",
    "these findings underscore",
    "this highlights the importance of",
    "these results suggest that",
    "these findings suggest that",
]

_DEFAULT_RELATED_WORK_ALIASES = [
    "Related Work",
    "Related Works",
    "Background and Related Work",
    "Related Work and Background",
]


def _is_quoted(text: str) -> bool:
    """A sentence carrying quotation marks is probably reproducing someone
    else's words (a prompt, an example output, a quoted claim) rather than
    stating the authors' own prose -- do not judge it as their writing."""
    return bool(_QUOTE_RE.search(text))


def _related_work_sentence_texts(ctx: CheckContext) -> set[str]:
    """Flattened sentence-shaped text drawn from the Related Work section.

    Used only to exempt that section from the connective-density check: a
    literature survey is expected to lean on "furthermore" and "moreover" to
    thread together other people's findings, which is not the same tic as an
    author padding their own argument.
    """
    default_aliases = _DEFAULT_RELATED_WORK_ALIASES
    aliases = [str(a) for a in (ctx.conf("structure.related_work_aliases", default_aliases) or [])]
    text = section_text(ctx, aliases)
    if not text:
        return set()
    return {" ".join(s.text.split()) for s in sentences(ctx, text)}


# ---------------------------------------------------------------------------
# 1. Connective overuse
# ---------------------------------------------------------------------------


@register("connective_tics", "Connective overuse", module=MODULE, category="prose", order=60)
def check_connective_tics(ctx: CheckContext) -> Finding:
    """Count throat-clearing connectives per 1000 words of body prose."""
    terms = [str(t) for t in (ctx.conf("prose.connective_terms", _DEFAULT_CONNECTIVES) or [])]
    rate_limit = float(ctx.conf("prose.connective_max_rate_per_1000_words", 3.0))
    min_count = int(ctx.conf("prose.connective_min_count", 3))

    related_work = _related_work_sentence_texts(ctx)
    counted = [
        s
        for s in sentences(ctx)
        if not _is_quoted(s.text) and " ".join(s.text.split()) not in related_work
    ]

    words_counted = sum(s.word_count for s in counted) or 1
    per_term: dict[str, list[str]] = {t: [] for t in terms}
    patterns = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in terms}

    for s in counted:
        for term, pat in patterns.items():
            if pat.search(s.text):
                per_term[term].append(s.text)

    offenders: list[Evidence] = []
    for term in terms:
        hits = per_term[term]
        count = len(hits)
        if count == 0:
            continue
        rate = count / words_counted * 1000
        if count >= min_count and rate > rate_limit:
            target = max(1, int(rate_limit * words_counted / 1000))
            offenders.append(
                Evidence(
                    detail=f'"{term}" used {count} times ({rate:.1f} / 1000 words, '
                    f"target under {rate_limit:.1f} / 1000, roughly {target} uses)",
                    quote=hits[0],
                )
            )

    if not offenders:
        return ctx.ok(
            "connective_tics",
            "Connective overuse",
            f"No connective from the watch list exceeds {rate_limit:.1f} uses per 1000 words "
            f"over {words_counted} words of body prose.",
            category="prose",
        )

    names = ", ".join(f'"{e.detail.split(chr(34))[1]}"' for e in offenders[:6])
    return ctx.warn(
        "connective_tics",
        "Connective overuse",
        f"{len(offenders)} connective(s) run above the watch-list rate over {words_counted} words: "
        f"{names}. The goal is to cut these DOWN toward the target count noted for each, not to "
        "remove every instance -- a paper with zero connectives reads worse, not better.",
        category="prose",
        evidence=offenders[:10],
        remedy="Read each flagged sentence aloud without the connective; keep it only where the "
        "logical link is not already obvious from the sentence order.",
        confidence="high — literal counts over body prose, excluding quoted text and Related Work",
    )


# ---------------------------------------------------------------------------
# 2. Vague demonstratives
# ---------------------------------------------------------------------------


@register("vague_demonstratives", "Vague demonstratives", module=MODULE, category="prose", order=61)
def check_vague_demonstratives(ctx: CheckContext) -> Finding:
    """Flag 'This <verb>' / 'These <verb>' with no named subject in between."""
    verbs = [str(v) for v in (ctx.conf("prose.demonstrative_verbs", _DEFAULT_DEMONSTRATIVE_VERBS) or [])]
    cap = int(ctx.conf("prose.demonstrative_max_report", 8))
    verb_alt = "|".join(re.escape(v) for v in verbs)
    pattern = re.compile(rf"\b(This|These)\s+({verb_alt})\b", re.IGNORECASE)

    hits: list[Evidence] = []
    total = 0
    for s in sentences(ctx):
        if _is_quoted(s.text):
            continue
        match = pattern.search(s.text)
        if not match:
            continue
        total += 1
        if len(hits) < cap:
            quote = s.text if len(s.text) <= 160 else s.text[:157] + "..."
            hits.append(
                Evidence(
                    page=s.page,
                    detail=f'"{match.group(1)} {match.group(2)}" has no named subject',
                    quote=quote,
                )
            )

    if not hits:
        return ctx.ok(
            "vague_demonstratives",
            "Vague demonstratives",
            "No sentence opens with a bare 'This/These <verb>' that leaves the subject to "
            "be reconstructed by the reader.",
            category="prose",
        )

    more = f" ({total - cap} more not shown)" if total > cap else ""
    return ctx.warn(
        "vague_demonstratives",
        "Vague demonstratives",
        f"{total} sentence(s) use a bare demonstrative as the subject of a verb, forcing the "
        f"reader to reconstruct what 'this' or 'these' refers to{more}.",
        category="prose",
        evidence=hits,
        remedy='Name the subject: "This shows..." -> "This ablation shows...", '
        '"These suggest..." -> "These results suggest...".',
        confidence="high — literal pattern match; a demonstrative followed by a noun is not flagged",
    )


# ---------------------------------------------------------------------------
# 3. Hollow wrap-up sentences
# ---------------------------------------------------------------------------


@register("hollow_wrapups", "Hollow wrap-up sentences", module=MODULE, category="prose", order=62)
def check_hollow_wrapups(ctx: CheckContext) -> Finding:
    """Flag sentences matching stock closing templates that add no information."""
    phrases = [str(p) for p in (ctx.conf("prose.hollow_wrapup_phrases", _DEFAULT_HOLLOW_WRAPUPS) or [])]
    cap = int(ctx.conf("prose.hollow_wrapup_max_report", 6))

    hits: list[Evidence] = []
    total = 0
    for s in sentences(ctx):
        if _is_quoted(s.text):
            continue
        lowered = s.text.lower()
        matched = next((p for p in phrases if p.lower() in lowered), None)
        if not matched:
            continue
        total += 1
        if len(hits) < cap:
            quote = s.text if len(s.text) <= 160 else s.text[:157] + "..."
            hits.append(Evidence(page=s.page, detail=f'matches wrap-up template "{matched}"', quote=quote))

    if not hits:
        return ctx.ok(
            "hollow_wrapups",
            "Hollow wrap-up sentences",
            "No sentence matches a stock hollow-summary template.",
            category="prose",
        )

    more = f" ({total - cap} more not shown)" if total > cap else ""
    return ctx.warn(
        "hollow_wrapups",
        "Hollow wrap-up sentences",
        f"{total} sentence(s) match a hollow closing template that restates having a result "
        f"without adding one{more}.",
        category="prose",
        evidence=hits,
        remedy="Replace each with one concrete fact (a number, a comparison, a named "
        "consequence) or cut the sentence -- the paragraph usually reads fine without it.",
        confidence="medium — substring match against a template list; wording that departs "
        "even slightly from the templates will not be caught",
    )
