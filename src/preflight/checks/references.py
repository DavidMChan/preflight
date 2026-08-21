"""Reference verification.

Wraps :mod:`preflight.refcheck`, which verifies a bibliography concurrently and
routes each citation to the verifier that actually fits it. See that package for
why it is built the way it is.

The framing is inherited deliberately from the tool that inspired it: a miss
surfaces **suspicion, not proof**. Indexes are incomplete, and "not found" never
means "fabricated".
"""

from __future__ import annotations

from typing import Any

from ..context import CheckContext
from ..llm.client import AsyncLLMClient
from ..models import Evidence, Finding
from ..refcheck import RefCheckConfig, bibliography_lines, parse_entries, segment, verify
from ..refcheck.core import MatchPolicy, Status
from ..registry import register
from .llm_checks import _client

MODULE = "integrations.refcheck"

_SUSPICION = (
    "This surfaces SUSPICION, NOT PROOF. Bibliographic indexes are incomplete and "
    "'not found' does not mean fabricated."
)

_STATUS_LABEL = {
    Status.NOT_FOUND: "no match anywhere",
    Status.AUTHOR_MISMATCH: "authors or year disagree",
    Status.UNRESOLVED: "cited link does not resolve",
    Status.ERROR: "lookup failed",
}


def _config(ctx: CheckContext) -> RefCheckConfig:
    cache = ctx.conf("refcheck.cache_path", "~/.cache/preflight/refcheck.db")
    settings_cap = getattr(ctx.settings, "hallucinator_max_refs", 0)
    return RefCheckConfig(
        max_references=int(settings_cap or ctx.conf("refcheck.max_references", 0)),
        concurrency=int(ctx.conf("refcheck.concurrency", 16)),
        search_concurrency=int(ctx.conf("refcheck.search_concurrency", 6)),
        timeout=float(ctx.conf("refcheck.timeout_secs", 6.0)),
        total_timeout=float(ctx.conf("refcheck.total_timeout_secs", 120.0)),
        use_llm_parse=bool(ctx.conf("refcheck.llm_parse", True)),
        use_web_search=bool(ctx.conf("refcheck.web_search", True)),
        enabled_sources=tuple(str(s) for s in (ctx.conf("refcheck.sources", []) or [])),
        mailto=ctx.conf("refcheck.mailto", None),
        semantic_scholar_key=ctx.conf("refcheck.semantic_scholar_key", None),
        github_token=ctx.conf("refcheck.github_token", None),
        cache_path=str(cache) if cache else None,
        policy=MatchPolicy(
            accept_title=float(ctx.conf("refcheck.accept_title", 0.90)),
            review_title=float(ctx.conf("refcheck.review_title", 0.72)),
            accept_author=float(ctx.conf("refcheck.accept_author", 0.34)),
            year_slack=int(ctx.conf("refcheck.year_slack", 2)),
        ),
    )


async def _run(ctx: CheckContext) -> dict[str, Any]:
    """Segment, parse and verify once; both checks read the result."""
    cached = ctx.shared.get("refcheck_report")
    if cached is not None:
        return cached

    entries = segment(ctx, bibliography_lines(ctx))
    config = _config(ctx)

    # Without a model the parser falls back to a structural pass, which still
    # recovers DOIs, arXiv ids and URLs — enough for most of the routing.
    client: AsyncLLMClient | None = _client(ctx) if ctx.settings.enable_llm else None
    if client is None or not config.use_llm_parse:
        client = None
    references = await parse_entries(ctx, entries, client or AsyncLLMClient(api_key=None))
    if client is None and ctx.settings.enable_llm:
        client = _client(ctx)      # still allowed for the web-search tier

    status = ctx.shared.get("status")

    def progress(done: int, total: int, label: str) -> None:
        if callable(status):
            try:
                status("reference_verification", f"references {done}/{total}: {label}")
            except Exception:      # a noisy UI must never fail the check
                pass

    report = await verify(references, config, client, progress=progress)
    if callable(status):
        try:
            status("reference_verification", "")
        except Exception:
            pass

    result = {"entries": entries, "references": references, "report": report}
    ctx.shared["refcheck_report"] = result
    return result


@register("reference_parsing", "Bibliography parsing", module=MODULE, category="references",
          requires=("enable_refcheck",), order=88)
async def check_reference_parsing(ctx: CheckContext) -> Finding:
    """Report how much of the bibliography could be read into structured records."""
    data = await _run(ctx)
    entries: list[str] = data["entries"]
    references = data["references"]

    if not entries:
        return ctx.skip("reference_parsing", "Bibliography parsing",
                        "No bibliography was found, so there was nothing to parse.",
                        category="references")

    usable = [r for r in references if r.is_usable]
    from collections import Counter
    kinds = Counter(r.kind.value for r in references)
    report = data["report"]
    evidence = [
        Evidence(detail="entry types: " + ", ".join(f"{k} x{v}" for k, v in kinds.most_common())),
        *[Evidence(detail=f"entry {r.index}", quote=r.label) for r in usable[:3]],
    ]

    duplicates = getattr(report, "duplicates", [])
    if duplicates:
        evidence.append(Evidence(
            detail=f"{len(duplicates)} work(s) are cited more than once; each was verified once"
        ))
        for label, indexes in duplicates[:5]:
            evidence.append(Evidence(detail=f"entries {', '.join(str(i) for i in indexes)}", quote=label))
        return ctx.warn(
            "reference_parsing", "Bibliography parsing",
            f"{len(entries)} bibliography entries parsed, but {len(duplicates)} work(s) appear more "
            "than once. A duplicate entry is usually a preprint and its published version listed "
            "separately, or a citation key used twice.",
            category="references", evidence=evidence,
            remedy="Merge the duplicate entries, or confirm they really are distinct works.",
            confidence="high — matched on DOI, arXiv id, or normalised title",
        )

    if len(usable) < len(entries):
        missed = len(entries) - len(usable)
        return ctx.warn(
            "reference_parsing", "Bibliography parsing",
            f"{missed} of {len(entries)} bibliography entries could not be parsed well enough to "
            "look up, so they were not checked. This is a limitation of reading the PDF, not a "
            "defect in the paper.",
            category="references", evidence=evidence,
            confidence="high — entry segmentation is measured from the bibliography's hanging indent",
        )
    return ctx.ok("reference_parsing", "Bibliography parsing",
                  f"All {len(entries)} bibliography entries parsed into structured records.",
                  category="references", evidence=evidence)


@register("reference_verification", "Reference verification", module=MODULE, category="references",
          requires=("enable_refcheck",), order=89)
async def check_reference_verification(ctx: CheckContext) -> Finding:
    """Verify every reference against databases, its own host, and the live web."""
    data = await _run(ctx)
    report = data["report"]

    if not report.verdicts:
        return ctx.skip("reference_verification", "Reference verification",
                        "No references could be parsed, so none were verified.",
                        category="references")

    verified = report.count(Status.VERIFIED)
    resolved = report.count(Status.RESOLVED)
    unindexed = report.count(Status.UNINDEXED)
    suspicious = report.suspicious

    scope = f"{len(report.verdicts)} reference(s)"
    if report.truncated:
        scope += f" (of {report.entries} found; capped)"
    timing = (
        f"Checked {scope} in {report.elapsed:.0f}s"
        f"{f', {report.cache_hits} from cache' if report.cache_hits else ''}"
        f"{f', {report.searched} needed a web search' if report.searched else ''}."
    )
    breakdown = Evidence(
        detail=f"{verified} verified in a database, {resolved} resolved to a live source, "
               f"{unindexed} not the kind of thing databases index"
    )

    notes: list[Evidence] = []
    if report.throttled and not ctx.conf("refcheck.mailto", None):
        hosts = ", ".join(sorted(report.throttled))
        notes.append(Evidence(
            detail=f"{hosts} rate-limited this run. Set `refcheck.mailto` in your profile to your "
                   "email address: CrossRef and OpenAlex both serve identified clients from a "
                   "faster pool, which makes this check quicker and more accurate."
        ))
    elif report.throttled:
        notes.append(Evidence(detail="rate-limited by " + ", ".join(sorted(report.throttled))
                              + "; the affected lookups fell through to slower tiers"))

    if not suspicious:
        return ctx.ok("reference_verification", "Reference verification",
                      f"Every reference was accounted for. {timing}",
                      category="references", evidence=[breakdown, *notes])

    max_evidence = int(ctx.conf("refcheck.max_evidence", 15))
    evidence = [breakdown, *notes]
    for verdict in suspicious[:max_evidence]:
        label = _STATUS_LABEL.get(verdict.status, verdict.status.value)
        detail = f"[{label}]"
        if verdict.source:
            detail += f" via {verdict.source}"
        if verdict.note:
            detail += f" — {verdict.note}"
        evidence.append(Evidence(detail=detail, quote=verdict.reference.label))
    if len(suspicious) > max_evidence:
        evidence.append(Evidence(detail=f"...and {len(suspicious) - max_evidence} more"))

    as_error = bool(ctx.conf("refcheck.not_found_is_error", False))
    reporter = ctx.error if as_error else ctx.warn
    return reporter(
        "reference_verification", "Reference verification",
        f"{len(suspicious)} of {scope} could not be confirmed. {timing} {_SUSPICION}",
        category="references", evidence=evidence,
        remedy="Check each flagged entry by hand against the publisher's page, the arXiv listing "
        f"or the cited URL before changing anything. {_SUSPICION} Fix genuinely wrong titles, "
        "authors or years; leave correct-but-unindexed entries alone.",
        confidence="low — an index miss is a lead to check, never a verdict"
        if not as_error else "medium — the profile promoted misses to errors",
    )
