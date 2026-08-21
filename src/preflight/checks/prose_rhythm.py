"""Prose-rhythm checks: sentence-length distribution and repeated openers.

Both checks are deterministic counts over ``sentences(ctx)`` -- no model calls,
no judgement calls about style. "Don't eyeball it -- count it." They are
advisory (WARNING only): a long sentence or a run of "We ..." sentences is
never a desk-rejection condition, and a competent paper can legitimately have
a handful of long sentences or a short run of similar openers. The bar for
firing is a genuine long tail (36+ word sentences, especially 46+) and a
genuine *run* of 3 or more consecutive sentences sharing an opener -- not the
mere presence of long sentences or common openers.
"""

from __future__ import annotations

from ..analysis import Sentence, sentences
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.prose"


def _is_table_leakage(ctx: CheckContext, s: Sentence) -> bool:
    """Drop sentence-splitter artifacts from tables/results blocks, not real prose.

    ``sentences()`` splits on punctuation, and a wide results table with no
    periods between cells (metric names, percentages, deltas) can read as one
    huge "sentence". That is a PDF-extraction artifact, not a long sentence a
    reviewer has to parse, and flagging it would be exactly the false positive
    this check must not produce. Numeric-heavy sentences are the tell: real
    prose sentences in this paper run under ~15% digit-bearing words even when
    they cite metrics; leaked table rows run 25%+.
    """
    words = s.words
    if not words:
        return False
    digit_threshold = float(ctx.conf("prose.table_leakage_digit_ratio", 0.2))
    symbol_threshold = float(ctx.conf("prose.table_leakage_symbol_ratio", 0.1))
    unique_threshold = float(ctx.conf("prose.table_leakage_unique_ratio", 0.5))
    digit_words = sum(1 for w in words if any(c.isdigit() for c in w))
    # Checkbox/dash table cells (✗ ✓ –) carry no digits but are just as clearly a
    # leaked row, e.g. an ablation table's "trained  ✗ ✗  Rewriter-Only  – ✓ ✗ ...".
    symbol_words = sum(1 for w in words if not any(c.isalnum() for c in w))
    # A figure with repeated boxes ("The Country" x3) or a repeated headline mock-up
    # extracts as one long, heavily-repetitive "sentence" with no digits or symbols
    # at all; real prose this long never collapses onto so few distinct words.
    unique_ratio = len({w.lower() for w in words}) / len(words)
    return (
        (digit_words / len(words)) >= digit_threshold
        or (symbol_words / len(words)) >= symbol_threshold
        or unique_ratio < unique_threshold
    )


@register("sentence_length", "Sentence length", module=MODULE, category="prose", order=40)
def check_sentence_length(ctx: CheckContext) -> Finding:
    """Histogram body-prose sentences by length; flag the long tail, not the mean."""
    hard_max = int(ctx.conf("prose.sentence_hard_max_words", 46))
    soft_max = int(ctx.conf("prose.sentence_soft_max_words", 36))
    cap = int(ctx.conf("prose.max_reported_sentences", 8))

    sents = [s for s in sentences(ctx) if not _is_table_leakage(ctx, s)]
    if not sents:
        return ctx.skip("sentence_length", "Sentence length", "No body prose sentences found.",
                         category="prose")

    hard: list[Sentence] = [s for s in sents if s.word_count >= hard_max]
    soft: list[Sentence] = [s for s in sents if soft_max <= s.word_count < hard_max]
    longest = max(s.word_count for s in sents)

    if not hard:
        band_note = f", {len(soft)} in the {soft_max}-{hard_max - 1} word band" if soft else ""
        return ctx.ok(
            "sentence_length",
            "Sentence length",
            f"No sentence reaches the {hard_max}-word hard limit (longest is {longest} words"
            f"{band_note}, out of {len(sents)} body sentences).",
            category="prose",
        )

    def _evidence(s: Sentence, note: str) -> Evidence:
        return Evidence(
            page=s.page,
            detail=note,
            quote=s.text,
            measured=float(s.word_count),
            expected=f"< {hard_max} words",
        )

    worst = sorted(hard, key=lambda s: -s.word_count)
    no_break = sum(1 for s in soft if not s.has_internal_break)
    evidence: list[Evidence] = [_evidence(s, "sentence at or above the hard limit") for s in worst[:cap]]

    message = (
        f"{len(hard)} sentence(s) reach {hard_max}+ words (longest is {longest} words) out of "
        f"{len(sents)} body sentences; {len(soft)} more fall in the {soft_max}-{hard_max - 1} word band"
        f" ({no_break} of those with no natural split point). Sentences this long are hard for a "
        "reviewer to hold in one pass and should be broken up."
    )
    return ctx.warn(
        "sentence_length",
        "Sentence length",
        message,
        category="prose",
        evidence=evidence,
        remedy=f"Break sentences at {hard_max}+ words into two. For the "
        f"{soft_max}-{hard_max - 1} word band, prefer splitting at an existing ':' ';' or "
        "em dash where one is already present.",
        confidence="high — measured word counts over body prose sentences",
    )


@register("repeated_openers", "Repeated sentence openings", module=MODULE, category="prose", order=41)
def check_repeated_openers(ctx: CheckContext) -> Finding:
    """Flag runs of 3+ consecutive body sentences that share the same opening word."""
    run_len = int(ctx.conf("prose.repeated_opener_run", 3))
    cap = int(ctx.conf("prose.max_reported_runs", 6))
    ignore = {w.lower() for w in ctx.conf("prose.repeated_opener_ignore", [])}

    sents = [s for s in sentences(ctx) if not _is_table_leakage(ctx, s)]
    if not sents:
        return ctx.skip("repeated_openers", "Repeated sentence openings", "No body prose sentences found.",
                         category="prose")

    runs: list[list[Sentence]] = []
    current: list[Sentence] = []
    for s in sents:
        opener = s.opener
        if current and opener and opener == current[-1].opener:
            current.append(s)
        else:
            if len(current) >= run_len:
                runs.append(current)
            current = [s] if opener else []
    if len(current) >= run_len:
        runs.append(current)

    # Runs of an ignored opener are common academic scaffolding ("We", "This") and
    # only worth flagging if the run is unusually long; short runs of them are dropped.
    kept = [r for r in runs if r[0].opener not in ignore or len(r) > run_len + 1]

    if not kept:
        return ctx.ok(
            "repeated_openers",
            "Repeated sentence openings",
            f"No run of {run_len}+ consecutive sentences shares the same opening word "
            f"({len(sents)} body sentences checked).",
            category="prose",
        )

    kept.sort(key=len, reverse=True)
    evidence = [
        Evidence(
            page=r[0].page,
            detail=f"{len(r)} consecutive sentences open with \"{r[0].opener.capitalize()}\"",
            quote=r[0].text,
        )
        for r in kept[:cap]
    ]
    message = (
        f"{len(kept)} run(s) of {run_len}+ consecutive sentences share the same opening word "
        f"(longest run: {len(kept[0])} sentences starting with \"{kept[0][0].opener}\"). "
        "This is a rhythm tell for reviewers, not a content problem, but it reads as a draft "
        "that was not revised."
    )
    return ctx.warn(
        "repeated_openers",
        "Repeated sentence openings",
        message,
        category="prose",
        evidence=evidence,
        remedy="Vary the opening of at least one sentence in each run -- lead with the object, "
        "a transition, or a subordinate clause instead of repeating the subject.",
        confidence="high — exact match on the first word of consecutive body sentences",
    )
