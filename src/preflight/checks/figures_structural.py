"""Deterministic figure/table checks: caption content, numbering, decimal precision.

All three are counting exercises over ``captions(ctx)`` and the raw page text, not
judgements about whether a figure is any good. They report WARNING only -- a caption
that fails these counts is a prompt for the author or a reviewer to look, never a
provable defect the way a page-size or margin violation is.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..analysis import Caption, captions
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.figures"

_LABEL_RE = re.compile(r"^(?:([A-Za-z]+)\.)?(\d+)$")
_DECIMAL_RE = re.compile(r"(?<![\w.])\d{1,4}\.\d{1,3}(?![\w.])")


# ---------------------------------------------------------------------------
# 1. caption_completeness
# ---------------------------------------------------------------------------


def _default_metric_words() -> list[str]:
    return [
        "accuracy", "f1", "bleu", "rouge", "precision", "recall", "score", "auc",
        "perplexity", "loss", "latency", "rate", "rates", "wr", "em", "exact match",
        "rmse", "mae", "correlation", "throughput", "selection", "win",
    ]


def _default_dataset_words() -> list[str]:
    return [
        "dataset", "corpus", "benchmark", "split", "training set", "test set",
        "dev set", "validation", "subset", "held-out",
    ]


def _default_direction_words() -> list[str]:
    return [
        "higher", "lower", "better", "worse", "outperform", "improves", "improve",
        "increase", "decrease", "larger is better", "smaller is better", "↑", "↓",
    ]


@register("caption_completeness", "Caption self-containedness", module=MODULE,
          category="figures", order=40)
def check_caption_completeness(ctx: CheckContext) -> Finding:
    """Flag captions that are short, or that report numbers with no stated metric,
    dataset, or direction-of-better -- a reader skimming figures alone may not be
    able to follow them without hunting through the prose."""
    min_words = int(ctx.conf("figures.caption_min_words", 8))
    min_numbers = int(ctx.conf("figures.caption_results_min_numbers", 2))
    metric_words = [w.lower() for w in ctx.conf("figures.caption_metric_words", _default_metric_words())]
    dataset_words = [w.lower() for w in ctx.conf("figures.caption_dataset_words", _default_dataset_words())]
    direction_words = [w.lower() for w in ctx.conf("figures.caption_direction_words",
                                                     _default_direction_words())]
    cap = int(ctx.conf("figures.caption_max_reported", 10))

    caps = captions(ctx)
    if not caps:
        return ctx.skip(
            "caption_completeness", "Caption self-containedness",
            "No figure or table captions were extracted from this PDF.",
            category="figures",
        )

    flagged: list[Evidence] = []
    for c in caps:
        lower = c.text.lower()
        terse = c.word_count < min_words
        results_like = len(c.numbers) >= min_numbers
        has_metric = any(w in lower for w in metric_words)
        has_dataset = any(w in lower for w in dataset_words)
        has_direction = any(w in lower for w in direction_words)
        missing_context = results_like and not (has_metric or has_dataset or has_direction)

        if not (terse or missing_context):
            continue

        missing_bits = []
        if terse:
            missing_bits.append(f"only {c.word_count} words (threshold {min_words})")
        if missing_context:
            gaps = [name for name, hit in
                    (("metric", has_metric), ("dataset", has_dataset), ("direction-of-better", has_direction))
                    if not hit]
            missing_bits.append("mentions no " + ", ".join(gaps))
        flagged.append(Evidence(
            page=c.page,
            detail=f"{c.name}: " + "; ".join(missing_bits),
            quote=c.text,
        ))

    if flagged:
        return ctx.warn(
            "caption_completeness",
            "Caption self-containedness",
            f"{len(flagged)} of {len(caps)} caption(s) may be worth a second look for "
            "self-containedness -- a reader skimming figures without the surrounding prose "
            "might struggle with these. This is a prompt to check, not a defect: a short "
            "caption on a schematic or qualitative figure is completely normal.",
            category="figures",
            evidence=flagged[:cap],
            remedy="For any caption reporting numbers, consider stating the metric, the "
            "dataset/split, and which direction is better, so the figure stands alone.",
        )
    return ctx.ok(
        "caption_completeness",
        "Caption self-containedness",
        f"All {len(caps)} caption(s) are reasonably long and, where they report numbers, "
        "name a metric, dataset, or direction-of-better.",
        category="figures",
    )


# ---------------------------------------------------------------------------
# 2. caption_numbering
# ---------------------------------------------------------------------------


def _sequence_key(c: Caption) -> tuple[str, str] | None:
    match = _LABEL_RE.match(c.label.strip())
    if not match:
        return None
    prefix = match.group(1) or ""
    return (c.kind, prefix)


@register("caption_numbering", "Figure and table numbering", module=MODULE,
          category="figures", order=41)
def check_caption_numbering(ctx: CheckContext) -> Finding:
    """Look for gaps and duplicates within each figure/table numbering sequence.

    Appendix labels like "B.2" form their own sequence per prefix, so a gap in the
    main-body figures never gets compared against appendix figures.
    """
    cap = int(ctx.conf("figures.numbering_max_reported", 20))
    caps = captions(ctx)
    if not caps:
        return ctx.skip(
            "caption_numbering", "Figure and table numbering",
            "No figure or table captions were extracted from this PDF.",
            category="figures",
        )

    groups: dict[tuple[str, str], list[Caption]] = defaultdict(list)
    unparsed = 0
    for c in caps:
        key = _sequence_key(c)
        if key is None:
            unparsed += 1
            continue
        groups[key].append(c)

    dup_evidence: list[Evidence] = []
    gap_evidence: list[Evidence] = []
    for (kind, prefix), items in groups.items():
        by_num: dict[int, list[Caption]] = defaultdict(list)
        for c in items:
            by_num[int(_LABEL_RE.match(c.label).group(2))].append(c)

        for _num, group in sorted(by_num.items()):
            if len(group) > 1:
                pages = ", ".join(str(g.page) for g in group)
                label = group[0].name
                dup_evidence.append(Evidence(
                    detail=f"'{label}' appears {len(group)} times, on pages {pages}",
                ))

        nums = sorted(by_num)
        if len(nums) >= 2:
            for missing in range(nums[0], nums[-1] + 1):
                if missing not in by_num:
                    before = max(n for n in nums if n < missing)
                    after = min(n for n in nums if n > missing)
                    seq_name = f"{prefix}." if prefix else ""
                    gap_evidence.append(Evidence(
                        page=by_num[before][-1].page,
                        detail=f"{kind.capitalize()} {seq_name}{missing} is missing between "
                        f"{kind} {seq_name}{before} (page {by_num[before][-1].page}) and "
                        f"{kind} {seq_name}{after} (page {by_num[after][0].page})",
                    ))

    evidence = (dup_evidence + gap_evidence)[:cap]
    if evidence:
        parts = []
        if dup_evidence:
            parts.append(f"{len(dup_evidence)} duplicate label(s)")
        if gap_evidence:
            parts.append(f"{len(gap_evidence)} numbering gap(s)")
        return ctx.warn(
            "caption_numbering",
            "Figure and table numbering",
            f"Found {' and '.join(parts)} across the figure/table sequences. A gap "
            "usually means a caption on that page failed to extract rather than the item "
            "being genuinely absent, so treat this as a prompt to check the PDF, not proof "
            "of a missing figure.",
            category="figures",
            evidence=evidence,
            remedy="Confirm every Figure/Table number is used exactly once per sequence, "
            "including any appendix-lettered sequences.",
        )
    return ctx.ok(
        "caption_numbering",
        "Figure and table numbering",
        f"No gaps or duplicates across {len(groups)} figure/table sequence(s) "
        f"({sum(len(v) for v in groups.values())} caption(s) checked" +
        (f", {unparsed} label(s) not in a standard N or X.N form skipped)" if unparsed else ")"),
        category="figures",
    )


# ---------------------------------------------------------------------------
# 3. decimal_precision
# ---------------------------------------------------------------------------


@register("decimal_precision", "Decimal precision consistency", module=MODULE,
          category="figures", order=42)
def check_decimal_precision(ctx: CheckContext) -> Finding:
    """Flag pages holding a table where decimal numbers mix precisions heavily.

    Deliberately narrow: table structure is not parsed, so this only looks at pages
    that already carry a table caption (a proxy for "this page holds a table"), and
    only fires when a page carries many decimal numbers split across multiple
    precisions in comparable volume -- a single stray number never trips it. It
    cannot tell which column a number belongs to, so it under-reports on purpose
    rather than accusing a table of inconsistency it may not have.
    """
    min_tokens = int(ctx.conf("figures.decimal_precision_min_tokens_per_page", 6))
    min_occurrences = int(ctx.conf("figures.decimal_precision_min_occurrences", 2))
    require_overlap = bool(ctx.conf("figures.decimal_precision_require_range_overlap", True))
    cap = int(ctx.conf("figures.decimal_precision_max_reported_pages", 6))

    table_pages = {c.page for c in captions(ctx) if c.kind == "table"}
    if not table_pages:
        return ctx.skip(
            "decimal_precision", "Decimal precision consistency",
            "No table captions were found, so there is no page to scope this check to.",
            category="figures",
        )

    by_page: dict[int, list[str]] = defaultdict(list)
    for line in ctx.doc.reading_order:
        if line.page not in table_pages:
            continue
        by_page[line.page].extend(_DECIMAL_RE.findall(" ".join(line.text.split())))

    flagged: list[Evidence] = []
    for page in sorted(by_page):
        tokens = by_page[page]
        if len(tokens) < min_tokens:
            continue
        by_precision: dict[int, list[str]] = defaultdict(list)
        for tok in tokens:
            precision = len(tok.split(".")[1])
            by_precision[precision].append(tok)
        significant = {p: v for p, v in by_precision.items() if len(v) >= min_occurrences}
        if len(significant) < 2:
            continue
        if require_overlap:
            ranges = {p: (min(float(x) for x in v), max(float(x) for x in v)) for p, v in significant.items()}
            spans = sorted(ranges.values())
            # Two precisions whose value ranges never overlap are very likely two
            # different metrics/columns with naturally different precision, not the
            # same column reported inconsistently -- skip those.
            if not any(a[1] >= b[0] for a, b in zip(spans, spans[1:], strict=False)):
                continue
        examples = ", ".join(
            f"{p} decimal place(s): {', '.join(v[:3])}" for p, v in sorted(significant.items())
        )
        flagged.append(Evidence(
            page=page,
            detail=f"table page mixes decimal precisions -- {examples}",
        ))

    if flagged:
        return ctx.warn(
            "decimal_precision",
            "Decimal precision consistency",
            f"{len(flagged)} table page(s) mix numbers with different decimal precision "
            "in comparable volume. This may reflect genuinely different quantities (a "
            "percentage next to a p-value, say) rather than an inconsistent table -- worth "
            "a quick look, not a confirmed defect.",
            category="figures",
            evidence=flagged[:cap],
            remedy="Within a single table column, report all values to the same number of "
            "decimal places.",
        )
    return ctx.ok(
        "decimal_precision",
        "Decimal precision consistency",
        f"Checked {len(table_pages)} table page(s); no page mixes decimal precisions in "
        "volume large enough to flag.",
        category="figures",
    )
