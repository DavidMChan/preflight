"""Registers one check per model-backed audit spec.

See :mod:`preflight.audit` for the spec format and why the audits are data.
"""

from __future__ import annotations

from ..audit import AUDIT_SCHEMA, AUDIT_SYSTEM, Audit, audit_prompt, load_audits, normalise_quote
from ..context import CheckContext
from ..llm.client import LLMError
from ..models import Evidence, Finding
from ..registry import register
from .llm_checks import _client

_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


async def _run_audit(ctx: CheckContext, audit: Audit) -> Finding | None:
    try:
        data = await _client(ctx).json(AUDIT_SYSTEM, audit_prompt(ctx, audit), schema_hint=AUDIT_SCHEMA)
    except LLMError as exc:
        return ctx.skip(audit.check_id, audit.title, str(exc), category=audit.category)

    summary = str(data.get("summary", "")).strip()
    verdict = data.get("verdict")
    confidence = str(data.get("confidence", "low"))
    issues = [i for i in (data.get("issues") or []) if isinstance(i, dict)]
    issues.sort(key=lambda i: _SEVERITY_RANK.get(str(i.get("severity", "low")).lower(), 3))

    note = f"{ctx.settings.llm_model} — advisory, not a formatting rule"
    headline = f"{verdict} — {summary}" if verdict else summary

    if not issues:
        return ctx.ok(audit.check_id, audit.title,
                      headline or "No issues found on this dimension.",
                      category=audit.category, confidence=f"{confidence} — {note}")

    evidence: list[Evidence] = []
    for issue in issues[: audit.max_issues]:
        severity = str(issue.get("severity", "low")).lower()
        where = str(issue.get("where", "")).strip()
        detail = f"[{severity}] {str(issue.get('title', '')).strip()}"
        if where:
            detail += f" — {where}"
        why = str(issue.get("why", "")).strip()
        if why:
            detail += f": {why}"
        quote = issue.get("quote")
        evidence.append(Evidence(detail=detail, quote=normalise_quote(str(quote)) if quote else None))
        fix = str(issue.get("fix", "")).strip()
        if fix:
            evidence.append(Evidence(detail=f"    fix: {fix}"))

    high = sum(1 for i in issues if str(i.get("severity", "")).lower() == "high")
    counted = f"{len(issues)} issue{'s' if len(issues) != 1 else ''}"
    if high:
        counted += f" ({high} rated high)"
    return ctx.warn(
        audit.check_id, audit.title,
        f"{counted}. {headline}".strip(),
        category=audit.category,
        evidence=evidence,
        confidence=f"{confidence} — {note}",
        remedy="Advisory: these are reviewer-perception issues, not venue rules. Weigh them "
        "against your own judgement of the paper.",
    )


def _make_check(audit: Audit):
    async def check(ctx: CheckContext) -> Finding | None:
        return await _run_audit(ctx, audit)

    check.__name__ = f"check_{audit.check_id}"
    check.__doc__ = audit.description or audit.title
    return check


for _audit in load_audits():
    register(
        _audit.check_id,
        _audit.title,
        module=_audit.module,
        category=_audit.category,
        requires=("enable_llm",),
        order=300 + _audit.order,
        description=_audit.description or _audit.title,
    )(_make_check(_audit))
