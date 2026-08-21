"""Model-backed paper audits, as data.

An audit is a reviewer's lens applied to the whole paper: does the evaluation
support the claims, do the figures stand alone, does the prose read like nobody
wrote it. They all have the same shape — read the paper, return issues with
quotes — so they are YAML files under ``preflight/conferences/audits/`` rather
than one Python function each.

Every audit is advisory. None of them can produce an error: they are judgements
about quality and presentation, not measurements of a documented desk-rejection
condition. That distinction is the whole point of this tool.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

from .analysis import captions, equation_lines, paper_context, sentences
from .context import CheckContext

DEFAULT_MODULE = "llm.audit"

#: The one response shape every audit uses.
AUDIT_SCHEMA = """{
  "summary": "one or two sentences stating what you found",
  "verdict": "optional short verdict, or null",
  "confidence": "low"|"medium"|"high",
  "issues": [
    {
      "title": "short label for the issue",
      "severity": "high"|"medium"|"low",
      "where": "section, figure, table or equation it applies to",
      "quote": "short verbatim quote from the paper, or null",
      "why": "why this matters to a reviewer",
      "fix": "the concrete change you would make"
    }
  ]
}"""

AUDIT_SYSTEM = (
    "You are an experienced reviewer for a competitive CS/AI venue, reading a paper on behalf of "
    "its own authors before they submit.\n\n"
    "Rules you must follow:\n"
    "- Judge only from the text provided. Never invent quotations, numbers, section names or "
    "citations. Every quote must appear verbatim in the text.\n"
    "- Report only issues you can point at. An audit that returns no issues is a valid, useful "
    "result; padding the list with generic advice is not.\n"
    "- Be specific. 'The evaluation is weak' is useless; 'Table 3 reports a single run with no "
    "variance, so the 0.4-point gain over the baseline is uninterpretable' is actionable.\n"
    "- PDF extraction artifacts — hyphenation, flattened tables, inline captions, lost maths "
    "formatting — are not defects in the paper. Never report them.\n"
    "- Rank by what would actually change a reviewer's score, not by what is easiest to say."
)


@dataclass(slots=True)
class Audit:
    id: str
    title: str
    prompt: str
    module: str = DEFAULT_MODULE
    category: str = "audit"
    order: int = 0
    context: str = "none"
    description: str = ""
    guidance: str = ""
    max_issues: int = 8
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def check_id(self) -> str:
        return f"audit_{self.id}"


# ---------------------------------------------------------------------------
# Structured context providers
# ---------------------------------------------------------------------------

CONNECTIVES = [
    "furthermore", "moreover", "notably", "consequently", "additionally",
    "importantly", "therefore", "thus", "hence", "overall", "crucially",
    "it is worth noting", "it should be noted", "in particular",
]


def _ctx_none(ctx: CheckContext) -> str:
    return ""


def _ctx_captions(ctx: CheckContext) -> str:
    found = captions(ctx)
    if not found:
        return "\n=== CAPTIONS ===\n(No figure or table captions could be extracted.)\n"
    lines = [f"- [page {c.page}] {c.name} ({c.word_count} words): {c.text}" for c in found]
    return "\n=== CAPTIONS EXTRACTED FROM THE PDF ===\n" + "\n".join(lines) + "\n"


def _ctx_equations(ctx: CheckContext) -> str:
    found = equation_lines(ctx)
    if not found:
        return "\n=== DISPLAY EQUATIONS ===\n(None could be extracted; the paper may have no display maths.)\n"
    lines = [f"- [page {page}] {text}" for page, text in found[:60]]
    return (
        "\n=== LINES THAT LOOK LIKE DISPLAY EQUATIONS ===\n"
        "Extracted heuristically from the PDF, so symbols and sub/superscripts may be mangled. "
        "Use them to locate the maths in the text above; do not treat mangling as an error.\n"
        + "\n".join(lines) + "\n"
    )


def _ctx_long_sentences(ctx: CheckContext) -> str:
    found = [s for s in sentences(ctx) if s.word_count >= 36]
    if not found:
        return "\n=== LONG SENTENCES ===\n(No sentence in the body runs to 36 words or more.)\n"
    lines = [f"- [{s.word_count} words] {s.text}" for s in sorted(found, key=lambda s: -s.word_count)[:30]]
    return "\n=== SENTENCES OF 36+ WORDS (measured) ===\n" + "\n".join(lines) + "\n"


def _ctx_prose_stats(ctx: CheckContext) -> str:
    body = sentences(ctx)
    if not body:
        return "\n=== PROSE STATISTICS ===\n(No body prose could be extracted.)\n"
    lengths = [s.word_count for s in body]
    text = " ".join(s.text.lower() for s in body)
    tics = Counter({c: text.count(c) for c in CONNECTIVES})
    openers = Counter(s.opener for s in body if s.opener)
    runs: list[str] = []
    streak = 1
    for prev, cur in zip(body, body[1:], strict=False):
        if cur.opener and cur.opener == prev.opener:
            streak += 1
        else:
            if streak >= 3:
                runs.append(f"{streak} consecutive sentences opening with '{prev.opener}'")
            streak = 1
    if streak >= 3:
        runs.append(f"{streak} consecutive sentences opening with '{body[-1].opener}'")

    buckets = Counter(min(w // 10 * 10, 50) for w in lengths)
    histogram = ", ".join(f"{k}-{k + 9}w: {buckets[k]}" for k in sorted(buckets))
    return (
        "\n=== MEASURED PROSE STATISTICS (body text only) ===\n"
        f"sentences: {len(body)}; mean length {sum(lengths) / len(lengths):.1f} words; "
        f"longest {max(lengths)} words\n"
        f"length histogram: {histogram}\n"
        f"sentences >= 46 words: {sum(1 for w in lengths if w >= 46)}; "
        f"36-45 words: {sum(1 for w in lengths if 36 <= w < 46)}\n"
        f"connective counts: {', '.join(f'{k}={v}' for k, v in tics.most_common() if v) or 'none'}\n"
        f"most common sentence openers: {', '.join(f'{k}={v}' for k, v in openers.most_common(6))}\n"
        f"repeated-opener runs: {'; '.join(runs) if runs else 'none'}\n"
    )


def _ctx_surface(ctx: CheckContext) -> str:
    from .analysis import abstract_text, conclusion_text

    title = ""
    for line in ctx.doc.reading_order:
        if line.page == 1 and line.text.strip():
            title = " ".join(line.text.split())
            break
    return (
        "\n=== TRIAGE SURFACE ===\n"
        f"Title (first line of page 1): {title}\n\n"
        f"Abstract:\n{abstract_text(ctx) or '(not extracted)'}\n\n"
        f"Conclusion:\n{conclusion_text(ctx) or '(not extracted)'}\n"
        + _ctx_captions(ctx)
    )


CONTEXT_PROVIDERS: dict[str, Callable[[CheckContext], str]] = {
    "none": _ctx_none,
    "captions": _ctx_captions,
    "equations": _ctx_equations,
    "long_sentences": _ctx_long_sentences,
    "prose_stats": _ctx_prose_stats,
    "surface": _ctx_surface,
}


# ---------------------------------------------------------------------------
# Loading and prompt assembly
# ---------------------------------------------------------------------------


def _coerce(data: dict[str, Any], source: str) -> Audit:
    missing = [k for k in ("id", "title", "prompt") if not data.get(k)]
    if missing:
        raise ValueError(f"{source}: audit is missing {', '.join(missing)}")
    context = str(data.get("context", "none"))
    if context not in CONTEXT_PROVIDERS:
        raise ValueError(
            f"{source}: unknown context {context!r}; available: {', '.join(sorted(CONTEXT_PROVIDERS))}"
        )
    return Audit(
        id=str(data["id"]),
        title=str(data["title"]),
        prompt=str(data["prompt"]).strip(),
        module=str(data.get("module", DEFAULT_MODULE)),
        category=str(data.get("category", "audit")),
        order=int(data.get("order", 0)),
        context=context,
        description=str(data.get("description", "")).strip(),
        guidance=str(data.get("guidance", "")).strip(),
        max_issues=int(data.get("max_issues", 8)),
        raw=data,
    )


def load_audits() -> list[Audit]:
    out: list[Audit] = []
    root = resources.files("preflight.conferences").joinpath("audits")
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith((".yaml", ".yml")):
            continue
        data = yaml.safe_load(entry.read_text("utf-8")) or {}
        out.append(_coerce(data, entry.name))
    return sorted(out, key=lambda a: (a.module, a.order, a.id))


def audit_prompt(ctx: CheckContext, audit: Audit) -> str:
    """Shared paper text first, then the audit's own material."""
    parts = [paper_context(ctx)]
    extra = CONTEXT_PROVIDERS[audit.context](ctx)
    if extra:
        parts.append(extra)
    parts.append(f"\n=== AUDIT: {audit.title} ===\n")
    if audit.guidance:
        parts.append(f"What this audit covers:\n{audit.guidance}\n\n")
    parts.append(audit.prompt.strip() + "\n")
    parts.append(
        f"\nReport at most {audit.max_issues} issues, most important first. Return an empty "
        "`issues` list if the paper is sound on this dimension.\n"
    )
    return "".join(parts)


def normalise_quote(text: str, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]
