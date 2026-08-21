"""ARR Responsible NLP Research checklist checks.

One check per checklist item, all generated from the YAML question specs in
``preflight/conferences/rnlp/``. See :mod:`preflight.rnlp` for why the questions
are data and how the prompts are shaped for caching.

Every item is a warning at worst. The CFP is explicit that answering "no" with a
justification is acceptable and is not grounds for rejection; what *is*
desk-rejectable is systematically failing to provide either the relevant
sections or a justification, which the summary check reports.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..context import CheckContext
from ..llm.client import LLMError
from ..models import Evidence, Finding, Severity
from ..registry import register
from ..rnlp import (
    ANSWER_SCHEMA,
    SCOPE_ALWAYS,
    SCOPE_SCHEMA,
    SYSTEM_PROMPT,
    Question,
    load_questions,
    question_prompt,
    scope_prompt,
)
from .llm_checks import _client

MODULE = "arr.responsible_nlp"

_STATUS_SEVERITY = {
    "yes": Severity.PASS,
    "not_applicable": Severity.PASS,
    "partial": Severity.WARNING,
    "no": Severity.WARNING,
}


async def _scope(ctx: CheckContext) -> dict[str, Any]:
    """Decide once which checklist sections apply to this paper.

    Every gated question needs this answer and they all start at the same
    moment, so the first caller creates the task and the rest await it. Without
    the single-flight guard we would pay for the same call thirteen times.
    """
    task = ctx.shared.get("rnlp_scope_task")
    if task is None:
        async def fetch() -> dict[str, Any]:
            try:
                return await _client(ctx).json(SYSTEM_PROMPT, scope_prompt(ctx), schema_hint=SCOPE_SCHEMA)
            except LLMError as exc:
                return {"_error": str(exc)}

        task = asyncio.ensure_future(fetch())
        ctx.shared["rnlp_scope_task"] = task
    data = await task
    ctx.shared["rnlp_scope"] = data
    return data


async def _applies(ctx: CheckContext, question: Question) -> tuple[bool, str]:
    if question.applies_when == SCOPE_ALWAYS:
        return True, ""
    scope = await _scope(ctx)
    if "_error" in scope:
        return True, ""  # cannot gate; run the item rather than silently skipping it
    entry = scope.get(question.applies_when)
    if not isinstance(entry, dict):
        return True, ""
    return bool(entry.get("answer", True)), str(entry.get("why", ""))


def _record(ctx: CheckContext, question: Question, status: str) -> None:
    ctx.shared.setdefault("rnlp_results", {})[question.code] = status


async def _run_question(ctx: CheckContext, question: Question) -> Finding | None:
    applicable, why = await _applies(ctx, question)
    if not applicable:
        _record(ctx, question, "not_applicable")
        return ctx.ok(
            question.check_id, question.display,
            f"Not applicable to this paper. {why}".strip(),
            category="responsible-nlp",
            confidence=f"scope decided by {ctx.settings.llm_model}",
        )

    try:
        data = await _client(ctx).json(
            SYSTEM_PROMPT, question_prompt(ctx, question), schema_hint=ANSWER_SCHEMA
        )
    except LLMError as exc:
        return ctx.skip(question.check_id, question.display, str(exc), category="responsible-nlp")

    status = str(data.get("status", "")).lower().replace(" ", "_")
    severity = _STATUS_SEVERITY.get(status, Severity.WARNING)
    _record(ctx, question, status)

    where = data.get("where")
    explanation = str(data.get("explanation", "")).strip()
    confidence = str(data.get("confidence", "low"))
    quotes = [str(q) for q in (data.get("quotes") or [])][:3]
    missing = [str(m) for m in (data.get("missing") or [])][:5]

    evidence: list[Evidence] = []
    if where:
        evidence.append(Evidence(detail=f"discussed in: {where}"))
    evidence += [Evidence(detail="quoted from the paper", quote=q) for q in quotes]
    evidence += [Evidence(detail=f"not found: {m}") for m in missing]

    if severity is Severity.PASS:
        message = (
            f"Not applicable. {explanation}" if status == "not_applicable"
            else f"The paper contains this. {explanation}"
        )
        return ctx.ok(question.check_id, question.display, message.strip(),
                      category="responsible-nlp", evidence=evidence,
                      confidence=f"{confidence} — judged by {ctx.settings.llm_model}")

    lead = "Only partly covered." if status == "partial" else "Not found in the paper."
    return ctx.warn(
        question.check_id, question.display,
        f"{lead} {explanation} Answering 'no' with a justification on the submission form is "
        "acceptable and is not grounds for rejection — but a reviewer will look for this.",
        category="responsible-nlp", evidence=evidence,
        confidence=f"{confidence} — judged by {ctx.settings.llm_model}",
        remedy=f"Either add the information and cite the section in checklist item {question.code}, "
        "or justify its absence in the checklist response.",
    )


def _make_check(question: Question):
    async def check(ctx: CheckContext) -> Finding | None:
        return await _run_question(ctx, question)

    check.__name__ = f"check_{question.check_id}"
    check.__doc__ = f"{question.code}: {question.question}"
    return check


# Register one check per checklist item, in checklist order.
_QUESTIONS = load_questions()
for _index, _question in enumerate(_QUESTIONS):
    register(
        _question.check_id,
        _question.display,
        module=MODULE,
        category="responsible-nlp",
        requires=("enable_llm",),
        order=200 + _index,
        description=_question.question,
    )(_make_check(_question))


@register("rnlp_summary", "Responsible NLP checklist coverage", module=MODULE,
          category="responsible-nlp", requires=("enable_llm",), order=299, aggregate=True)
def check_summary(ctx: CheckContext) -> Finding | None:
    """Report overall coverage, and flag systematic failure to address the checklist."""
    results: dict[str, str] = ctx.shared.get("rnlp_results") or {}
    if not results:
        return None

    applicable = {code: status for code, status in results.items() if status != "not_applicable"}
    if not applicable:
        return ctx.ok("rnlp_summary", "Responsible NLP checklist coverage",
                      "No checklist items applied to this paper.", category="responsible-nlp")

    covered = sum(1 for s in applicable.values() if s == "yes")
    partial = sum(1 for s in applicable.values() if s == "partial")
    missing = sorted(code for code, s in applicable.items() if s == "no")
    total = len(applicable)
    ratio = (covered + 0.5 * partial) / total

    evidence = [
        Evidence(detail=f"{covered} of {total} applicable items are fully covered, {partial} partly"),
        Evidence(detail=f"{len(results) - total} item(s) judged not applicable"),
    ]
    if missing:
        evidence.append(Evidence(detail="not found in the paper: " + ", ".join(missing)))

    threshold = float(ctx.conf("responsible_nlp.systematic_failure_ratio", 0.5))
    if ratio < threshold:
        return ctx.error(
            "rnlp_summary", "Responsible NLP checklist coverage",
            f"Only {covered} of {total} applicable checklist items are clearly addressed in the paper "
            f"({partial} partly). Submissions that systematically fail to provide either the relevant "
            "sections or a justification are desk-rejected, so either add the material or make sure "
            "every 'no' carries a justification on the submission form.",
            category="responsible-nlp", evidence=evidence,
            confidence=f"aggregate of per-item judgements by {ctx.settings.llm_model}",
            remedy="Work through the items flagged above before submitting.",
        )
    if missing or partial:
        return ctx.warn(
            "rnlp_summary", "Responsible NLP checklist coverage",
            f"{covered} of {total} applicable checklist items are clearly addressed, {partial} partly, "
            f"{len(missing)} not found. Each gap needs either the information in the paper or a "
            "justification on the submission form.",
            category="responsible-nlp", evidence=evidence,
            confidence=f"aggregate of per-item judgements by {ctx.settings.llm_model}",
        )
    return ctx.ok(
        "rnlp_summary", "Responsible NLP checklist coverage",
        f"All {total} applicable checklist items are addressed in the paper.",
        category="responsible-nlp", evidence=evidence,
        confidence=f"aggregate of per-item judgements by {ctx.settings.llm_model}",
    )
