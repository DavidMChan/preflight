"""Reference verification.

Wraps :mod:`preflight.refcheck`, which verifies a bibliography concurrently and
routes each citation to the verifier that actually fits it. See that package for
why it is built the way it is.

The framing is inherited deliberately from the tool that inspired it: a miss
surfaces **suspicion, not proof**. Indexes are incomplete, and "not found" never
means "fabricated".
"""

from __future__ import annotations

import os
from pathlib import Path
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
    "Indexes are incomplete, so an unconfirmed reference is not proof of fabrication, but a "
    "reader must be able to trace every citation."
)

_STATUS_LABEL = {
    Status.NOT_FOUND: "no match anywhere",
    Status.UNCONFIRMED: "no key source confirms it",
    Status.WEB_ONLY: "found only by web search",
    Status.AUTHOR_MISMATCH: "authors or year disagree",
    Status.UNRESOLVED: "cited link does not resolve",
    Status.ERROR: "lookup failed",
}

_FIELD_LABEL = {
    "doi": "DOI", "arxiv": "arXiv id", "url": "link", "pages": "pages", "year": "year",
    "venue": "venue", "authors": "authors", "version": "version",
}


def _setting(ctx: CheckContext, key: str, env: str) -> str | None:
    """A profile value, or the environment variable that supplies it."""
    value = ctx.conf(f"refcheck.{key}", None) or os.environ.get(env)
    return str(value) if value else None


def _config(ctx: CheckContext) -> RefCheckConfig:
    cache = ctx.conf("refcheck.cache_path", "~/.cache/preflight/refcheck.db")
    anthology = ctx.conf("refcheck.acl_anthology_path", None)
    if not anthology and cache:
        anthology = str(Path(str(cache)).expanduser().parent / "acl_anthology.db")
    settings_cap = getattr(ctx.settings, "hallucinator_max_refs", 0)
    return RefCheckConfig(
        max_references=int(settings_cap or ctx.conf("refcheck.max_references", 0)),
        concurrency=int(ctx.conf("refcheck.concurrency", 16)),
        search_concurrency=int(ctx.conf("refcheck.search_concurrency", 6)),
        timeout=float(ctx.conf("refcheck.timeout_secs", 6.0)),
        total_timeout=float(ctx.conf("refcheck.total_timeout_secs", 150.0)),
        use_llm_parse=bool(ctx.conf("refcheck.llm_parse", True)),
        use_web_search=bool(ctx.conf("refcheck.web_search", True)),
        use_anthology=bool(ctx.conf("refcheck.acl_anthology", True)),
        use_arxiv=bool(ctx.conf("refcheck.arxiv", True)),
        use_dblp=bool(ctx.conf("refcheck.dblp", True)),
        enabled_sources=tuple(str(s) for s in (ctx.conf("refcheck.sources", []) or [])),
        mailto=_setting(ctx, "mailto", "PREFLIGHT_MAILTO"),
        semantic_scholar_key=_setting(ctx, "semantic_scholar_key", "SEMANTIC_SCHOLAR_API_KEY"),
        github_token=_setting(ctx, "github_token", "GITHUB_TOKEN"),
        openalex_key=_setting(ctx, "openalex_key", "OPENALEX_API_KEY"),
        cache_path=str(cache) if cache else None,
        anthology_path=str(anthology) if anthology else None,
        anthology_max_age_days=float(ctx.conf("refcheck.acl_anthology_max_age_days", 7.0)),
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


def _entry(verdict: Any) -> str:
    """How a reader finds the entry: its printed label and its title."""
    reference = verdict.reference
    return f"{reference.marker} {reference.label}" if reference.marker else reference.label


def _source_notes(ctx: CheckContext, report: Any) -> list[Evidence]:
    """Sources that refused or failed this run, since they narrow what could be confirmed."""
    notes: list[Evidence] = []
    throttled = sorted(set(report.throttled) - set(report.source_errors))
    if throttled and not _setting(ctx, "mailto", "PREFLIGHT_MAILTO"):
        hosts = ", ".join(throttled)
        notes.append(Evidence(
            detail=f"{hosts} rate-limited this run. Set `refcheck.mailto` in your profile (or "
                   "PREFLIGHT_MAILTO) to your email address: CrossRef and OpenAlex both serve "
                   "identified clients from a faster pool."
        ))
    elif throttled:
        notes.append(Evidence(detail="rate-limited by " + ", ".join(throttled)
                              + "; the affected lookups fell through to slower tiers"))
    for host, error in sorted(report.source_errors.items()):
        notes.append(Evidence(detail=f"{host}: {error}"))
    if report.timed_out:
        notes.append(Evidence(detail=f"{report.timed_out} reference(s) were not finished within "
                                     "refcheck.total_timeout_secs"))
    return notes


@register("reference_verification", "Reference verification", module=MODULE, category="references",
          requires=("enable_refcheck",), order=89)
async def check_reference_verification(ctx: CheckContext) -> Finding | list[Finding]:
    """Confirm every reference against a key source: a record whose title and authors match.

    A reference that no key source has, but that a web search located at a page
    that answers, is reported apart from the rest and as a warning: a reader can
    follow it to the work, which is not true of one nobody can find.
    """
    data = await _run(ctx)
    report = data["report"]

    if not report.verdicts:
        return ctx.skip("reference_verification", "Reference verification",
                        "No references could be parsed, so none were verified.",
                        category="references")

    confirmed = report.count(Status.VERIFIED) + report.count(Status.DETAILS_MISMATCH)
    resolved = report.count(Status.RESOLVED)
    unindexed = report.count(Status.UNINDEXED)
    # A failed lookup is not a confirmation either, so it is reported with the rest.
    unconfirmed = [v for v in report.verdicts if v.is_suspicious or v.status is Status.ERROR]
    web_only = [v for v in unconfirmed if v.status is Status.WEB_ONLY]
    unconfirmed = [v for v in unconfirmed if v.status is not Status.WEB_ONLY]

    scope = f"{len(report.verdicts)} reference(s)"
    if report.truncated:
        scope += f" (of {report.entries} found; capped)"
    timing = (
        f"Checked {scope} in {report.elapsed:.0f}s"
        f"{f', {report.cache_hits} from cache' if report.cache_hits else ''}"
        f"{f', {report.searched} needed a web search' if report.searched else ''}."
    )
    breakdown = Evidence(
        detail=f"{confirmed} confirmed by a key source, {resolved} resolved to a live source, "
               f"{unindexed} not the kind of thing databases index"
               + (f", {len(web_only)} found only by web search" if web_only else "")
    )
    notes = _source_notes(ctx, report)
    max_evidence = int(ctx.conf("refcheck.max_evidence", 15))
    extra = [_web_only(ctx, web_only, scope, max_evidence)] if web_only else []

    if not unconfirmed:
        summary = (f"Every reference was found, {len(web_only)} of them only by a web search "
                   "(reported separately)." if web_only else "Every reference was confirmed.")
        return [ctx.ok("reference_verification", "Reference verification", f"{summary} {timing}",
                       category="references", evidence=[breakdown, *notes]), *extra]

    evidence = [breakdown, *notes, *_listed(unconfirmed, max_evidence)]

    as_error = bool(ctx.conf("refcheck.not_found_is_error", True))
    reporter = ctx.error if as_error else ctx.warn
    return [reporter(
        "reference_verification", "Reference verification",
        f"{len(unconfirmed)} of {scope} could not be confirmed by any key source. {timing} "
        f"{_SUSPICION}",
        category="references", evidence=evidence,
        remedy="Check each flagged entry by hand against the publisher's page, the arXiv listing "
        f"or the cited URL before changing anything. {_SUSPICION} Fix genuinely wrong titles, "
        "authors or years; a real work that no index lists needs a citation a reader can follow "
        "to it, such as a DOI, arXiv id or URL.",
        confidence="medium — no key source confirms these, though indexes are incomplete",
    ), *extra]


def _listed(verdicts: list[Any], cap: int) -> list[Evidence]:
    evidence = []
    for verdict in verdicts[:cap]:
        label = _STATUS_LABEL.get(verdict.status, verdict.status.value)
        detail = f"[{label}]"
        if verdict.source:
            detail += f" via {verdict.source}"
        if verdict.note:
            detail += f" — {verdict.note}"
        evidence.append(Evidence(detail=detail, quote=_entry(verdict)))
    if len(verdicts) > cap:
        evidence.append(Evidence(detail=f"...and {len(verdicts) - cap} more"))
    return evidence


def _web_only(ctx: CheckContext, verdicts: list[Any], scope: str, cap: int) -> Finding:
    """References a web search traced to a live page that no key source lists."""
    as_error = bool(ctx.conf("refcheck.web_only_is_error", False))
    reporter = ctx.error if as_error else ctx.warn
    return reporter(
        "reference_verification.web_only", "References found only by web search",
        f"{len(verdicts)} of {scope} were located by a web search at a page that answers, but no "
        "key source confirms them, so their titles, authors and years are unchecked.",
        category="references", evidence=_listed(verdicts, cap),
        remedy="Open each page to confirm it is the work cited, and give the entry a DOI, arXiv id "
        "or URL a reader can follow, since no bibliographic index lists it.",
        confidence="medium — the page exists, but nothing on it was matched against the citation",
    )


_FIELD_ORDER = ("doi", "arxiv", "url", "authors", "pages", "year", "venue", "version")


def _ranked(verdicts: list[Any], fields: set[str]) -> list[tuple[Any, list[Any]]]:
    """Each verdict's discrepancies in ``fields``, most serious entries first."""
    def order(field: str) -> int:
        return _FIELD_ORDER.index(field) if field in _FIELD_ORDER else len(_FIELD_ORDER)

    chosen = [(v, sorted((d for d in v.discrepancies if d.field in fields), key=lambda d: order(d.field)))
              for v in verdicts]
    chosen = [(v, found) for v, found in chosen if found]
    return sorted(chosen, key=lambda pair: order(pair[1][0].field))


def _listing(ctx: CheckContext, entries: list[tuple[Any, list[Any]]]) -> list[Evidence]:
    # Every entry is a concrete correction, so all are listed unless the profile caps it.
    cap = int(ctx.conf("refcheck.max_detail_evidence", 0) or 0) or len(entries)
    evidence = [
        Evidence(detail=" · ".join(f"{_FIELD_LABEL.get(d.field, d.field)}: {d.detail}" for d in found),
                 quote=_entry(verdict))
        for verdict, found in entries[:cap]
    ]
    if len(entries) > cap:
        evidence.append(Evidence(detail=f"...and {len(entries) - cap} more"))
    return evidence


#: A citation that gets its work wrong. A preprint cited in place of its
#: published version is right about the work, so it is reported separately.
_DETAIL_FIELDS = {"doi", "arxiv", "url", "authors", "pages", "year", "venue"}


@register("reference_details", "Reference details", module=MODULE, category="references",
          requires=("enable_refcheck",), order=89)
async def check_reference_details(ctx: CheckContext) -> Finding:
    """Compare each confirmed reference against its record: DOI, pages, year, venue, authors."""
    data = await _run(ctx)
    report = data["report"]

    if not report.verdicts:
        return ctx.skip("reference_details", "Reference details",
                        "No references could be parsed, so none were compared.",
                        category="references")

    entries = _ranked(report.miscited, _DETAIL_FIELDS)
    confirmed = report.count(Status.VERIFIED) + report.count(Status.DETAILS_MISMATCH)
    if not entries:
        return ctx.ok(
            "reference_details", "Reference details",
            f"All {confirmed} confirmed reference(s) match their records: every printed DOI "
            "resolves to the cited work, and the pages, year, venue and authors agree.",
            category="references")

    fields = sorted({_FIELD_LABEL.get(d.field, d.field) for _, found in entries for d in found})
    as_error = bool(ctx.conf("refcheck.details_mismatch_is_error", True))
    reporter = ctx.error if as_error else ctx.warn
    return reporter(
        "reference_details", "Reference details",
        f"{len(entries)} reference(s) cite a real work with details its record contradicts "
        f"({', '.join(fields)}).",
        category="references", evidence=_listing(ctx, entries),
        remedy="Replace each flagged entry with the record's own citation — the BibTeX from the "
        "ACL Anthology, the DOI, DBLP or arXiv — rather than correcting fields by hand.",
        confidence="high — each discrepancy is read off the key source's own record",
    )


@register("reference_versions", "Published versions", module=MODULE, category="references",
          requires=("enable_refcheck",), order=89)
async def check_reference_versions(ctx: CheckContext) -> Finding:
    """Find preprints, and works cited without a venue, that have a published version."""
    data = await _run(ctx)
    report = data["report"]

    if not report.verdicts:
        return ctx.skip("reference_versions", "Published versions",
                        "No references could be parsed, so none were compared.",
                        category="references")

    entries = _ranked(report.miscited, {"version"})
    if not entries:
        return ctx.ok("reference_versions", "Published versions",
                      "No preprint is cited in place of a published version.",
                      category="references")

    as_error = bool(ctx.conf("refcheck.published_version_is_error", False))
    reporter = ctx.error if as_error else ctx.warn
    return reporter(
        "reference_versions", "Published versions",
        f"{len(entries)} reference(s) cite a preprint, or no venue, for a work that has been "
        "published.",
        category="references", evidence=_listing(ctx, entries),
        remedy="Cite the published version: its venue, year, pages and DOI.",
        confidence="high — the published version is listed by a key source under the same title "
        "and authors, or by the preprint's own arXiv record",
    )
