"""Semantic checks that need a language model.

These run only with ``--llm``. Each one is scoped to a single requirement over a
single excerpt, and every finding says which model produced it, so a reader can
tell deterministic measurement apart from model judgement.
"""

from __future__ import annotations

import re
from typing import Any

from ..analysis import paper_context
from ..context import CheckContext
from ..llm.client import AsyncLLMClient, LLMError
from ..llm.prompts import (
    ANONYMITY_PROMPT,
    ANONYMITY_SCHEMA,
    CANARY_PROMPT,
    CANARY_SYSTEM,
    INJECTION_PROMPT,
    INJECTION_SCHEMA,
    LIMITATIONS_PROMPT,
    LIMITATIONS_SCHEMA,
    REVIEWER_SYSTEM,
    SCORE_PROMPT,
    SCORE_SCHEMA,
    STATEMENT_COVERAGE_PROMPT,
    STATEMENT_COVERAGE_SCHEMA,
)
from ..models import Evidence, Finding, Severity
from ..registry import register

# The same matching the deterministic check uses, so the two agree on what is
# injection-shaped and only the adjudication differs.
from .hidden import _compiled_patterns, _concealed_text
from .statements import statement_text

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


@register("llm_statement_coverage", "Statement coverage (model)", module=MODULE,
          category="semantic", requires=("enable_llm",), order=60)
async def check_statement_coverage(ctx: CheckContext) -> list[Finding] | None:
    """Ask a model whether each declared statement covers the items it must.

    Whether a statement exists is the deterministic check's job; whether its
    prose settles every listed item is a reading task only a model can do.
    """
    if not _enabled(ctx, "statement_semantics"):
        return None
    out: list[Finding] = []
    for key, spec in (ctx.conf("statements", {}) or {}).items():
        items = [str(i) for i in ((spec or {}).get("must_address") or [])]
        found = statement_text(ctx, str(key))
        if not items or found is None:
            continue    # a missing statement is already an error of its own
        page, body = found
        check_id = f"llm_statement_coverage.{key}"
        title = f"{spec.get('title') or key} coverage (model)"
        prompt = STATEMENT_COVERAGE_PROMPT.format(
            title=spec.get("title") or key, rule=str(spec.get("must_address_rule", "")).strip(),
            items="\n".join(f"- {i}" for i in items), body=_clip(body),
        )
        try:
            data = await _client(ctx).json(REVIEWER_SYSTEM, prompt, schema_hint=STATEMENT_COVERAGE_SCHEMA)
        except LLMError as exc:
            out.append(ctx.skip(check_id, title, str(exc), category="semantic"))
            continue

        verdicts = {str(v.get("item", "")).strip().lower(): v for v in (data.get("items") or [])
                    if isinstance(v, dict)}
        # An item the model skipped counts against the statement, not for it.
        status = {i: str(verdicts.get(i.lower(), {}).get("status", "unaddressed")) for i in items}
        unaddressed = [i for i in items if status[i] == "unaddressed"]
        blanket = [i for i in items if status[i] == "blanket"]
        confidence = f"{data.get('confidence', 'low')} — {_model_note(ctx)}"
        explanation = str(data.get("explanation", "")).strip()
        if not unaddressed and not blanket:
            out.append(ctx.ok(check_id, title,
                              f"The statement names all {len(items)} required items. {explanation}".strip(),
                              category="semantic", confidence=confidence))
            continue
        parts = []
        if unaddressed:
            parts.append(f"{len(unaddressed)} of {len(items)} required items are not addressed at all: "
                         + "; ".join(unaddressed) + ".")
        if blanket:
            parts.append(f"{len(blanket)} are covered only by a catch-all sentence rather than named: "
                         + "; ".join(blanket) + ".")
        quotes = [str(v.get("quote")) for v in verdicts.values() if v.get("quote")][:4]
        out.append(ctx.warn(
            check_id, title, " ".join(parts) + (f" {explanation}" if explanation else ""),
            category="semantic", confidence=confidence,
            evidence=[Evidence(page=page, detail="model-selected quote", quote=q) for q in quotes],
            remedy=str(spec.get("remedy", "")).strip() or None,
        ))
    return out


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


#: Phrases a passage demands are usually quoted inside it; four words or more
#: of a quotation coming back verbatim is the passage writing the review.
_QUOTED_SPAN = re.compile("[\"“]([^\"”]{15,200})[\"”]")
#: Verdict language a one-sentence summary has no reason to reach for.
_VERDICT_WORDS = re.compile(
    r"\b(?:accept|acceptance|strong\s+accept|high\s+score|score\s+of|rating\s+of|"
    r"recommend(?:s|ed|ation)?)\b", re.IGNORECASE)
_CANARY_TOKEN = "PREFLIGHT-OK"


def _canary_residue(excerpts: list[str], reply: str) -> list[str]:
    """What the passage did to the probe, as a list of plain-language signals.

    Deterministic and separate from the probe itself, so the judgement stays
    inspectable: every signal is something a reader can confirm against the
    reply by eye. An empty list is not evidence of safety, only an absence of
    residue -- a passage can fail to move this particular probe and still be
    aimed squarely at a reviewer.
    """
    signals: list[str] = []
    flat = " ".join(reply.split())
    if not flat:
        return signals
    if _CANARY_TOKEN not in flat:
        signals.append("the probe dropped its own control token, so something in the passage "
                       "displaced its instructions")
    demanded = {m.group(1).strip() for excerpt in excerpts for m in _QUOTED_SPAN.finditer(excerpt)}
    echoed = [phrase for phrase in demanded
              if len(phrase.split()) >= 4 and phrase.lower() in flat.lower()]
    if echoed:
        signals.append(f"the probe reproduced {len(echoed)} phrase(s) the passage demanded: "
                       + "; ".join(f'"{p[:60]}"' for p in echoed[:3]))
    verdict = _VERDICT_WORDS.search(flat)
    if verdict:
        signals.append("the probe's summary reached for reviewer verdict language "
                       f"({verdict.group(0)!r})")
    return signals


async def _canary_probe(ctx: CheckContext, excerpts: list[str]) -> tuple[str, list[str]] | None:
    """Run the passage past a model that can do nothing with it, then read the reply.

    The adjudication call asks a model whether the text *is* manipulation, which
    puts the untrusted text and the question in one prompt. This asks something
    the text cannot argue with: given a task it has no part in, does the reply
    come back changed? Returns None when the probe could not run, which is not
    evidence in either direction.
    """
    if not ctx.conf("llm.canary_probe", True):
        return None
    try:
        reply = await _client(ctx).complete(
            CANARY_SYSTEM, CANARY_PROMPT.format(excerpts=_clip("\n".join(excerpts))))
    except LLMError:
        return None
    return reply, _canary_residue(excerpts, reply)


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

    probe = await _canary_probe(ctx, excerpts)
    residue = probe[1] if probe else []

    explanation = str(data.get("explanation", "")).strip()
    confidence = str(data.get("confidence", "low"))
    evidence = [Evidence(detail="model-selected quote", quote=str(q))
                for q in (data.get("quotes") or [])[:5]]

    if residue:
        evidence.append(Evidence(detail="canary probe reply", quote=probe[0][:300]))

    if data.get("is_manipulation") or residue:
        if residue and not data.get("is_manipulation"):
            lead = ("The model read this as legitimate content, but a probe run on the same text came "
                    "back changed, which is the stronger signal: ")
            confidence = "medium"
        else:
            lead = f"The model judges this to be an attempt to manipulate an automated reviewer. {explanation} "
        tail = (" ".join(f"Probe residue: {s}." for s in residue) if residue
                else "The probe run on the same text came back clean.")
        return ctx.error(
            "llm_injection", "Machine-reader manipulation (model)",
            f"{lead}{tail} This may result in desk rejection.",
            category="semantic", evidence=evidence,
            confidence=f"{confidence} — {_model_note(ctx)}",
            cfp_key="hidden_text",
        )
    probe_note = ("" if probe is None
                  else " A probe run on the same text came back with no residue.")
    return ctx.ok(
        "llm_injection", "Machine-reader manipulation (model)",
        f"{len(excerpts)} injection-like excerpt(s) were reviewed and judged legitimate scholarly content. "
        f"{explanation}{probe_note}".strip(),
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
