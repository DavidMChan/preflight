"""Semantic checks that need a language model.

These run only with ``--llm``. Each one is scoped to a single requirement over a
single excerpt, and every finding says which model produced it, so a reader can
tell deterministic measurement apart from model judgement.
"""

from __future__ import annotations

from typing import Any

from ..analysis import paper_context
from ..context import CheckContext
from ..llm.client import AsyncLLMClient, LLMError
from ..llm.prompts import (
    ANONYMITY_PROMPT,
    ANONYMITY_SCHEMA,
    INJECTION_PROMPT,
    INJECTION_SCHEMA,
    LIMITATIONS_PROMPT,
    LIMITATIONS_SCHEMA,
    REVIEWER_SYSTEM,
    SCORE_PROMPT,
    SCORE_SCHEMA,
)
from ..models import Evidence, Finding, Severity
from ..registry import register
# The same matching the deterministic check uses, so the two agree on what is
# injection-shaped and only the adjudication differs.
from .hidden import _compiled_patterns, _concealed_text

MODULE = "llm.semantic"

_MAX_EXCERPT_CHARS = 24_000


def _client(ctx: CheckContext) -> AsyncLLMClient:
    """The run's shared async client. The runner installs it; this is the fallback."""
    cached = ctx.shared.get("llm_client")
    if isinstance(cached, AsyncLLMClient):
        return cached
    client = AsyncLLMClient(
        model=ctx.settings.llm_model,
        api_key=ctx.settings.openai_api_key,
        timeout=ctx.settings.llm_timeout,
        max_concurrency=ctx.settings.llm_concurrency,
    )
    ctx.shared["llm_client"] = client
    return client


def _enabled(ctx: CheckContext, name: str) -> bool:
    wanted = ctx.conf("llm.checks", None)
    return True if wanted is None else name in [str(w) for w in wanted]


def _clip(text: str, limit: int = _MAX_EXCERPT_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[... excerpt truncated ...]"


def _model_note(ctx: CheckContext) -> str:
    return f"judged by {ctx.settings.llm_model}"


@register("llm_limitations_scope", "Limitations content (model)", module=MODULE,
          category="semantic", requires=("enable_llm",), order=60)
async def check_limitations_semantics(ctx: CheckContext) -> Finding | None:
    """Ask a model whether the Limitations section smuggles in new contributions."""
    if not _enabled(ctx, "limitations_semantics"):
        return None
    aliases = [str(a) for a in (ctx.conf("structure.limitations_aliases", ["Limitations"]) or [])]
    limitations = ctx.doc.find_heading(aliases)
    if limitations is None:
        return None
    references = ctx.doc.find_heading([str(a) for a in (ctx.conf("structure.references_aliases",
                                                                 ["References"]) or [])])
    body = ctx.doc.text_between(limitations, references)
    if len(body.split()) < 20:
        return None

    try:
        data = await _client(ctx).json(REVIEWER_SYSTEM, LIMITATIONS_PROMPT.format(body=_clip(body)),
                                 schema_hint=LIMITATIONS_SCHEMA)
    except LLMError as exc:
        return ctx.skip("llm_limitations_scope", "Limitations content (model)", str(exc), category="semantic")

    explanation = str(data.get("explanation", "")).strip()
    confidence = str(data.get("confidence", "low"))
    quotes = [str(q) for q in (data.get("quotes") or [])][:4]
    evidence = [Evidence(page=limitations.page, detail="model-selected quote", quote=q) for q in quotes]

    if data.get("introduces_new_content"):
        return ctx.warn(
            "llm_limitations_scope", "Limitations content (model)",
            f"The model judges that the Limitations section introduces new methods, analyses or results, "
            f"which the CFP does not permit. {explanation}",
            category="semantic", evidence=evidence,
            confidence=f"{confidence} — {_model_note(ctx)}",
            remedy="Move any new method, analysis or result into the main paper or the appendix.",
            cfp_key="limitations_scope",
        )
    return ctx.ok("llm_limitations_scope", "Limitations content (model)",
                  f"The model found no new methods, analyses or results in the Limitations section. "
                  f"{explanation}".strip(),
                  category="semantic", confidence=f"{confidence} — {_model_note(ctx)}",
                  cfp_key="limitations_scope")


def _anonymity_candidates(ctx: CheckContext) -> list[str]:
    """Pattern hits worth showing the model, with a little surrounding context."""
    out: list[str] = []
    for page, value in [*ctx.doc.emails, *ctx.doc.orcids]:
        out.append(f"[page {page}] {value}")
    identity = [str(d).lower() for d in (ctx.conf("anonymity.identity_url_domains", []) or [])]
    forbidden = [str(d).lower() for d in (ctx.conf("anonymity.forbidden_url_domains", []) or [])]
    seen: set[str] = set()
    for page, url in [*ctx.doc.hyperlinks, *ctx.doc.textual_urls]:
        low = url.lower()
        if url in seen or not any(d in low for d in identity + forbidden):
            continue
        seen.add(url)
        out.append(f"[page {page}] {url}")
    text = ctx.doc.text
    lowered = text.lower()
    for pattern in (ctx.conf("anonymity.self_reference_patterns", []) or []):
        pos = lowered.find(str(pattern).lower())
        if pos >= 0:
            out.append(f"[self-reference] ...{text[max(0, pos - 80): pos + 140]}...")
    for field in (ctx.conf("anonymity.metadata_fields", []) or []):
        value = (ctx.doc.metadata.get(str(field)) or "").strip()
        if value:
            out.append(f"[pdf metadata /{field}] {value}")
    return out[:40]


@register("llm_anonymity", "Anonymity (model)", module=MODULE,
          category="semantic", requires=("enable_llm",), order=61)
async def check_anonymity_semantics(ctx: CheckContext) -> Finding | None:
    """Have a model adjudicate the pattern-matcher's anonymity candidates."""
    if not _enabled(ctx, "anonymity_semantics"):
        return None

    page1 = ctx.doc.pages[0]
    height = float(ctx.conf("anonymity.title_block_height_pt", 260.0))
    title_block = "\n".join(
        ln.text for ln in page1.lines if ln.bbox[1] <= height and ln.text.strip()
    )
    candidates = _anonymity_candidates(ctx)

    try:
        data = await _client(ctx).json(
            REVIEWER_SYSTEM,
            ANONYMITY_PROMPT.format(
                title_block=_clip(title_block, 4000),
                excerpts="\n".join(candidates) or "(none flagged by pattern matching)",
            ),
            schema_hint=ANONYMITY_SCHEMA,
        )
    except LLMError as exc:
        return ctx.skip("llm_anonymity", "Anonymity (model)", str(exc), category="semantic")

    leaks = [leak for leak in (data.get("leaks") or []) if isinstance(leak, dict)]
    confidence = str(data.get("confidence", "low"))
    explanation = str(data.get("explanation", "")).strip()

    if not leaks:
        return ctx.ok("llm_anonymity", "Anonymity (model)",
                      f"The model found no identity leaks in the title block or the flagged excerpts. "
                      f"{explanation}".strip(),
                      category="semantic", confidence=f"{confidence} — {_model_note(ctx)}",
                      cfp_key="anonymity")

    evidence = [
        Evidence(detail=f"{str(leak.get('severity', 'unknown'))} — {str(leak.get('why', ''))}",
                 quote=str(leak.get("text", "")))
        for leak in leaks[:8]
    ]
    return ctx.warn(
        "llm_anonymity", "Anonymity (model)",
        f"The model flagged {len(leaks)} potential anonymity leak(s). Review each manually — an "
        f"institution named in related work is legitimate. {explanation}".strip(),
        category="semantic", evidence=evidence,
        confidence=f"{confidence} — {_model_note(ctx)}",
        remedy="Remove or anonymize anything that identifies the authors, including supplementary links.",
        cfp_key="anonymity",
    )


def _injection_excerpts(ctx: CheckContext, patterns: list[tuple[str, Any]],
                        limit: int = 30) -> list[str]:
    """Injection-shaped passages for the model to judge, with how they render.

    Matched per page against the page's text rather than span by span: the
    patterns are regexes, and a span is often half a sentence, so a phrase
    that straddles two of them would never match either. (An earlier version
    tested the pattern *source* as a literal substring, which no regex with a
    group in it can ever satisfy -- the model was reliably handed nothing.)
    """
    window = int(ctx.conf("hidden.injection_context_chars", 160))
    out: list[str] = []
    for page in ctx.doc.pages:
        page_text = " ".join(page.text.split())
        concealed = " ".join(_concealed_text(ctx, page).split())
        for source, pattern in patterns:
            for match in pattern.finditer(page_text):
                rendering = ("hidden, microscopic or in the page colour"
                             if pattern.search(concealed) else "visible")
                start = max(0, match.start() - window // 2)
                out.append(f"[page {page.number}, {rendering}, pattern {source!r}] "
                           f"{page_text[start : match.end() + window // 2]}")
                if len(out) >= limit:
                    return out
    return out


@register("llm_injection", "Machine-reader manipulation (model)", module=MODULE,
          category="semantic", requires=("enable_llm",), order=62)
async def check_injection_semantics(ctx: CheckContext) -> Finding | None:
    """Distinguish real prompt-injection attempts from papers that study them."""
    if not _enabled(ctx, "injection_semantics"):
        return None

    patterns = _compiled_patterns(ctx)
    if not patterns:
        return None
    excerpts = _injection_excerpts(ctx, patterns)

    if not excerpts:
        return ctx.ok("llm_injection", "Machine-reader manipulation (model)",
                      "No injection-like text was found for the model to adjudicate.",
                      category="semantic", cfp_key="hidden_text")

    try:
        data = await _client(ctx).json(REVIEWER_SYSTEM,
                                 INJECTION_PROMPT.format(excerpts=_clip("\n".join(excerpts))),
                                 schema_hint=INJECTION_SCHEMA)
    except LLMError as exc:
        return ctx.skip("llm_injection", "Machine-reader manipulation (model)", str(exc), category="semantic")

    explanation = str(data.get("explanation", "")).strip()
    confidence = str(data.get("confidence", "low"))
    evidence = [Evidence(detail="model-selected quote", quote=str(q))
                for q in (data.get("quotes") or [])[:5]]

    if data.get("is_manipulation"):
        return ctx.error(
            "llm_injection", "Machine-reader manipulation (model)",
            f"The model judges this to be an attempt to manipulate an automated reviewer, which may result "
            f"in desk rejection. {explanation}",
            category="semantic", evidence=evidence,
            confidence=f"{confidence} — {_model_note(ctx)}",
            cfp_key="hidden_text",
        )
    return ctx.ok(
        "llm_injection", "Machine-reader manipulation (model)",
        f"{len(excerpts)} injection-like excerpt(s) were reviewed and judged legitimate scholarly content. "
        f"{explanation}".strip(),
        category="semantic", evidence=evidence,
        confidence=f"{confidence} — {_model_note(ctx)}", cfp_key="hidden_text",
    )


@register("llm_scores", "Reviewer-style scores", module=MODULE,
          category="scores", requires=("enable_llm", "enable_scores"), order=70)
async def check_scores(ctx: CheckContext) -> Finding | None:
    """Produce a rehearsal review with per-dimension scores.

    The scores land in ``report.scores`` for the renderer; the finding itself is
    informational and never blocks.
    """
    if not ctx.conf("llm.scoring", True):
        return None

    # The whole paper, byte-identical to what every other model-backed check
    # sends. Scoring a paper from its first two pages produced exactly the hedge
    # you would expect — "cannot be assessed from the excerpt" — and it shares
    # the cached prefix with the other checks rather than paying for a bespoke one.
    try:
        data = await _client(ctx).json(REVIEWER_SYSTEM,
                                       SCORE_PROMPT.format(body=paper_context(ctx)),
                                       schema_hint=SCORE_SCHEMA)
    except LLMError as exc:
        return ctx.skip("llm_scores", "Reviewer-style scores", str(exc), category="scores")

    ctx.shared.setdefault("scores", {}).update(_normalize_scores(data, ctx.settings.llm_model))
    scores = data.get("scores") or {}
    summary = ", ".join(
        f"{name} {value.get('score')}/5"
        for name, value in scores.items()
        if isinstance(value, dict) and value.get("score") is not None
    )
    overall = data.get("overall") or {}
    risks = [str(r) for r in (data.get("desk_reject_risks") or [])]

    evidence = [Evidence(detail=f"{name}: {str(value.get('justification', ''))}")
                for name, value in scores.items() if isinstance(value, dict)]
    if risks:
        evidence.append(Evidence(detail="desk-reject risks the model saw: " + "; ".join(risks[:5])))

    return ctx.finding(
        "llm_scores", "Reviewer-style scores",
        Severity.PASS,
        f"{summary or 'no scores returned'}"
        + (f" | overall {overall.get('score')}/5 — {overall.get('recommendation', '')}" if overall else ""),
        category="scores", evidence=evidence,
        confidence=f"estimate only — {_model_note(ctx)}, read over the whole paper",
    )


def _normalize_scores(data: dict[str, Any], model: str) -> dict[str, Any]:
    scores = {}
    for name, value in (data.get("scores") or {}).items():
        if isinstance(value, dict) and value.get("score") is not None:
            scores[str(name)] = {
                "score": value.get("score"),
                "justification": str(value.get("justification", "")),
            }
    return {
        "model": model,
        "dimensions": scores,
        "overall": data.get("overall") or {},
        "strengths": [str(s) for s in (data.get("strengths") or [])],
        "weaknesses": [str(s) for s in (data.get("weaknesses") or [])],
        "desk_reject_risks": [str(s) for s in (data.get("desk_reject_risks") or [])],
    }


@register("llm_context_coverage", "Model input coverage", module=MODULE,
          category="semantic", requires=("enable_llm",), order=298, aggregate=True)
def check_context_coverage(ctx: CheckContext) -> Finding | None:
    """Say plainly when a model judged only part of the paper.

    Every model-backed finding in this report is only as good as the text the
    model saw. If the paper had to be trimmed to fit, that caveat belongs in the
    report, not in a comment.
    """
    if not ctx.shared.get("llm_paper_context"):
        return None      # no model call was made
    truncated = ctx.shared.get("llm_context_truncated")
    if not isinstance(truncated, dict):
        total = len(str(ctx.shared.get("llm_paper_context", "")))
        return ctx.ok(
            "llm_context_coverage", "Model input coverage",
            f"Every model-backed check read the whole paper ({total:,} characters).",
            category="semantic",
        )

    omitted = int(truncated.get("omitted_chars", 0))
    total = int(truncated.get("total_chars", 0))
    kept = int(truncated.get("kept_chars", 0))
    share = (omitted / total * 100) if total else 0.0
    return ctx.warn(
        "llm_context_coverage", "Model input coverage",
        f"The paper was trimmed to fit the model input: {omitted:,} of {total:,} characters "
        f"({share:.0f}%) from the middle of the document were NOT seen by any model-backed "
        "check. Treat every finding from those checks as covering only the parts that were "
        "read, and treat their silence about the omitted middle as meaningless.",
        category="semantic",
        evidence=[Evidence(detail="kept the first 60% and last 40% of the budget",
                           measured=float(kept), expected=f"{total:,} characters to read it whole")],
        remedy="Raise `llm.max_context_chars` in your profile — the default already accommodates "
        "a very long paper, so hitting it suggests an unusually large appendix.",
        confidence="high — this is measured, not judged",
    )
