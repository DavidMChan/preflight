"""Statistics checks: p-value reporting and percentage arithmetic.

These are deterministic string/arithmetic checks, not desk-rejection rules, so
every finding here is a WARNING regardless of how confident the match is. A
venue that wants one of these treated harder can promote it via its profile's
``severity:`` map.
"""

from __future__ import annotations

import re

from ..analysis import full_text
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.statistics"

# ---------------------------------------------------------------------------
# p_value_zero
# ---------------------------------------------------------------------------

# A p-value is a probability and is never exactly zero. Only match the literal
# "reported as zero" forms: p = 0, p = .000, p = 0.000, p = 0.0, P=0, p < 0,
# p < .000. Deliberately anchored so that "p = 0.001", "p = 0.04", or a longer
# mantissa like "p = 0.0003" never matches -- those are legitimate values that
# merely round toward, but not to, zero.
_ZERO_PATTERN = re.compile(
    r"""
    \b[pP]\s*(=|<|<=|\\leq|\\le)\s*
    (
        0(?:\.0+)?      # 0, 0.0, 0.00, 0.000, ...
        |
        \.0+            # .0, .00, .000, ...
    )
    (?![0-9.])
    """,
    re.VERBOSE,
)

# A sentence-ish window of context around a match, for the quote.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentence_containing(text: str, start: int, end: int) -> str:
    """Return the sentence (or a bounded window) surrounding text[start:end]."""
    lo = text.rfind(".", 0, start)
    lo = lo + 1 if lo != -1 else max(0, start - 120)
    hi = text.find(".", end)
    hi = hi + 1 if hi != -1 else min(len(text), end + 120)
    snippet = text[lo:hi].strip()
    snippet = " ".join(snippet.split())
    if len(snippet) > 220:
        # Fall back to a tight window centered on the match itself.
        window = text[max(0, start - 60) : min(len(text), end + 60)]
        snippet = " ".join(window.split())
    return snippet


@register("p_value_zero", "p-values reported as zero", module=MODULE, category="statistics", order=40)
def check_p_value_zero(ctx: CheckContext) -> Finding:
    """A p-value can never be exactly 0; papers should report a threshold instead."""
    text = full_text(ctx)

    evidence: list[Evidence] = []
    seen: set[str] = set()
    for m in _ZERO_PATTERN.finditer(text):
        quote = _sentence_containing(text, m.start(), m.end())
        if not quote or quote in seen:
            continue
        seen.add(quote)
        evidence.append(Evidence(detail="p-value reported as exactly zero", quote=quote))

    if not evidence:
        return ctx.ok(
            "p_value_zero",
            "p-values reported as zero",
            "No p-values were reported as exactly zero.",
            category="statistics",
        )

    return ctx.warn(
        "p_value_zero",
        "p-values reported as zero",
        f"{len(evidence)} place(s) report a p-value as exactly zero. A p-value is a probability and "
        "is never exactly 0; this is almost always a truncated or misreported threshold.",
        category="statistics",
        evidence=evidence[:8],
        remedy='Report a threshold instead, e.g. "p < .001" (or the smallest precision your test '
        "actually supports), never \"p = 0\" or \"p = .000\".",
        confidence="high — the literal text matches a value that rounds to zero at its own stated precision",
    )


# ---------------------------------------------------------------------------
# p_value_format
# ---------------------------------------------------------------------------

# Captures the operator, an optional leading zero, and the decimal digits, so
# style (leading-zero convention) and precision (decimal-place count) can both
# be read off the match without re-deriving them from the raw string.
_PVAL_PATTERN = re.compile(
    r"\b[pP]\s*(?:=|<|<=|\\leq|\\le)\s*(0?)(\.\d+)\b"
)


@register("p_value_format", "p-value formatting consistency", module=MODULE, category="statistics", order=41)
def check_p_value_format(ctx: CheckContext) -> Finding:
    """p-values should use one leading-zero convention and one decimal precision throughout."""
    text = full_text(ctx)
    min_count = int(ctx.conf("statistics.format_min_occurrences", 4))

    with_zero = 0
    without_zero = 0
    precisions: dict[int, int] = {}
    zero_examples: list[str] = []
    nozero_examples: list[str] = []

    for m in _PVAL_PATTERN.finditer(text):
        leading, decimals = m.group(1), m.group(2)
        places = len(decimals) - 1  # exclude the dot
        precisions[places] = precisions.get(places, 0) + 1
        quote = _sentence_containing(text, m.start(), m.end())
        if leading:
            with_zero += 1
            if len(zero_examples) < 3:
                zero_examples.append(quote)
        else:
            without_zero += 1
            if len(nozero_examples) < 3:
                nozero_examples.append(quote)

    total = with_zero + without_zero
    if total < min_count:
        return ctx.skip(
            "p_value_format",
            "p-value formatting consistency",
            f"Only {total} p-value(s) found in the text; too few to judge a consistent convention.",
            category="statistics",
        )

    evidence: list[Evidence] = []
    style_mixed = with_zero > 0 and without_zero > 0
    if style_mixed:
        evidence.append(
            Evidence(
                detail=f'"p = 0.xx" style used {with_zero} time(s); "p = .xx" style used {without_zero} time(s)',
            )
        )
        for q in zero_examples[:2]:
            evidence.append(Evidence(detail='example of "0.xx" style', quote=q))
        for q in nozero_examples[:2]:
            evidence.append(Evidence(detail='example of ".xx" style', quote=q))

    precision_mixed = len(precisions) > 1
    if precision_mixed:
        counts_str = ", ".join(f"{places} decimal place(s): {n}x" for places, n in sorted(precisions.items()))
        evidence.append(Evidence(detail=f"decimal-place counts observed — {counts_str}"))

    if not style_mixed and not precision_mixed:
        return ctx.ok(
            "p_value_format",
            "p-value formatting consistency",
            f"All {total} p-value(s) use a single, consistent convention.",
            category="statistics",
        )

    parts = []
    if style_mixed:
        parts.append('mixes the "p = 0.xx" and "p = .xx" leading-zero conventions')
    if precision_mixed:
        parts.append("mixes different numbers of decimal places")
    detail = " and ".join(parts)

    return ctx.warn(
        "p_value_format",
        "p-value formatting consistency",
        f"The paper {detail} across {total} p-value(s). Both conventions are correct on their own; "
        "only the mixing is untidy.",
        category="statistics",
        evidence=evidence[:8],
        remedy="Pick one leading-zero convention and one decimal precision for p-values and apply it "
        "throughout the paper.",
        confidence="medium — based on a regex scan of the extracted text, not the typeset source",
    )


# ---------------------------------------------------------------------------
# percentage_arithmetic
# ---------------------------------------------------------------------------

# "N of M (X%)" / "N of M, X%" / "N/M (X%)" / "N out of M, X%" — count and
# total sitting immediately next to each other with a percentage in the same
# breath. Deliberately narrow: does not match "increased by X%" (no count
# pair), ranges, or sentences where other numbers intervene between N and M.
_PCT_PATTERN = re.compile(
    r"""
    \b(?P<n>\d{1,7})
    \s*
    (?:/|(?:out\s+of)|of)
    \s*
    (?P<m>\d{1,7})
    \b
    (?:[^.\n]{0,12}?)          # short connective, e.g. " cases (" or ", or "
    \(?\s*(?P<pct>\d{1,3}(?:\.\d+)?)\s*\%\)?
    """,
    re.VERBOSE | re.IGNORECASE,
)


@register(
    "percentage_arithmetic", "Percentages match their counts", module=MODULE, category="statistics", order=42
)
def check_percentage_arithmetic(ctx: CheckContext) -> Finding:
    """Recompute N/M for every "N of M (X%)"-shaped statement and compare to the stated X%."""
    text = full_text(ctx)
    tolerance = float(ctx.conf("statistics.percentage_tolerance", 0.55))

    mismatches: list[Evidence] = []
    checked = 0
    seen: set[str] = set()

    for m in _PCT_PATTERN.finditer(text):
        n, mm, pct = int(m.group("n")), int(m.group("m")), float(m.group("pct"))
        if mm == 0 or n > mm:
            continue  # not a count-of-total relationship
        quote = _sentence_containing(text, m.start(), m.end())
        if quote in seen:
            continue
        seen.add(quote)
        computed = 100.0 * n / mm
        checked += 1
        if abs(computed - pct) > tolerance:
            mismatches.append(
                Evidence(
                    detail=f"stated {n}/{mm} = {pct:g}%",
                    quote=quote,
                    measured=computed,
                    expected=f"{pct:g}% (tolerance +/- {tolerance:g} pts)",
                )
            )

    if checked == 0:
        return ctx.skip(
            "percentage_arithmetic",
            "Percentages match their counts",
            'No "N of M (X%)"-shaped statements were found to recompute.',
            category="statistics",
        )

    if not mismatches:
        return ctx.ok(
            "percentage_arithmetic",
            "Percentages match their counts",
            f"Checked {checked} count-derived percentage(s); all match the stated arithmetic "
            f"within {tolerance:g} percentage points.",
            category="statistics",
        )

    return ctx.warn(
        "percentage_arithmetic",
        "Percentages match their counts",
        f"{len(mismatches)} of {checked} count-derived percentage(s) do not match the arithmetic "
        f"stated in the same sentence (tolerance +/- {tolerance:g} percentage points).",
        category="statistics",
        evidence=mismatches[:8],
        remedy="Recheck the count, total, and percentage in each flagged sentence; one of the three is "
        "likely a typo or a leftover from an earlier draft.",
        confidence="medium — only fires on an explicit \"N of/out of M (X%)\" pattern in one sentence",
    )
