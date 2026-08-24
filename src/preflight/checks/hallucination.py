"""DEPRECATED: the third-party `hallucinator` backend.

Superseded by :mod:`preflight.refcheck`, which is what `--refcheck` runs by
default. This module is kept so you can get a second opinion from the original
tool, and because its framing — a miss is suspicion, not proof — is the right
one and worth preserving.

Why it was replaced rather than tuned:

* It queries databases in a fallthrough, and its rate limits are global rather
  than per-worker, so the references that match nothing — the ones you actually
  want reported — cost roughly 45 seconds each. A 62-entry bibliography does not
  finish inside a sensible time budget.
* It has no notion of what is being cited, so a GitHub repository, a model card
  or a vendor blog post is looked up in bibliographic databases and then
  reported as a missing paper. Those are false accusations, and they were the
  majority of what it flagged on a real submission.

Enable it with `--hallucinator` alongside or instead of the built-in checker.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..context import CheckContext
from ..document import Line
from ..models import Evidence, Finding
from ..registry import register

MODULE = "integrations.hallucinator"

_SUSPICION_NOTE = (
    "This surfaces SUSPICION, NOT PROOF: bibliographic databases are incomplete, and "
    "'not found' does not mean fabricated."
)


def hallucinator_available() -> bool:
    """Whether the optional ``hallucinator`` extra is importable in this environment."""
    try:
        return find_spec("hallucinator") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken namespace package
        return False


@dataclass(slots=True)
class _Extraction:
    """References recovered from the bibliography, plus how we got them."""

    references: list[Any]
    entries_seen: int
    skipped: int
    strategy: str
    note: str = ""


def _install_hint() -> str:
    return "install with: uv sync --extra hallucinator"


def _line_number_re(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("hallucinator.line_number_regex", r"^\s*\d{1,4}\s*$")))


def _bibliography_lines(ctx: CheckContext) -> list[Line]:
    """Lines between the bibliography heading and whatever follows it, in reading order."""
    start = ctx.doc.find_heading([str(s) for s in ctx.conf(
        "hallucinator.reference_headings", ["References", "Bibliography", "Works Cited"])])
    if start is None:
        return []
    end = ctx.doc.find_heading([str(s) for s in ctx.conf(
        "hallucinator.after_reference_headings",
        ["Appendix", "Appendices", "Supplementary Material", "A Appendix"])])
    if end is not None and (end.page, end.order) < (start.page, start.order):
        end = None

    ignore = _line_number_re(ctx)
    out: list[Line] = []
    started = False
    for line in ctx.doc.reading_order:
        if ignore.match(line.text.strip()):
            continue  # anonymous submission templates number every line
        if not started:
            started = line.page == start.page and abs(line.bbox[1] - start.bbox[1]) < 1.0
            continue
        if end is not None and line.page == end.page and abs(line.bbox[1] - end.bbox[1]) < 1.0:
            break
        out.append(line)
    return out


def _join_entry(lines: list[Line]) -> str:
    """Glue an entry's lines back together, undoing end-of-line hyphenation."""
    text = ""
    for line in lines:
        piece = " ".join(line.text.split())
        if not piece:
            continue
        text = text[:-1] + piece if text.endswith("-") else (text + " " + piece).strip()
    return text


def _split_entries(ctx: CheckContext, lines: list[Line]) -> list[str]:
    """Group lines into entries using the hanging indent of a bibliography."""
    tol = float(ctx.conf("hallucinator.entry_indent_tolerance_pt", 2.5))
    lefts = [band[0] for band in ctx.doc.column_bands] or [min((line.bbox[0] for line in lines), default=0.0)]

    entries: list[list[Line]] = []
    current: list[Line] = []
    for line in lines:
        starts_entry = any(abs(line.bbox[0] - left) <= tol for left in lefts)
        if starts_entry and current:
            entries.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(current)
    return [t for t in (_join_entry(e) for e in entries) if t]


def _extract(ctx: CheckContext) -> _Extraction:
    """Parse the bibliography, preferring our own layout-aware segmentation."""
    from hallucinator import PdfExtractor  # noqa: PLC0415 - optional dependency

    cached = ctx.shared.get("hallucinator_extraction")
    if isinstance(cached, _Extraction):
        return cached

    extractor = PdfExtractor()
    min_entries = int(ctx.conf("hallucinator.min_layout_entries", 5))
    texts = _split_entries(ctx, _bibliography_lines(ctx))

    result: _Extraction
    if len(texts) >= min_entries:
        refs = [extractor.parse_reference(t) for t in texts]
        usable = [r for r in refs if r is not None and r.title and not getattr(r, "skip_reason", None)]
        result = _Extraction(usable, len(texts), len(texts) - len(usable), "layout")
    else:
        native = extractor.extract(str(ctx.doc.path))
        refs = list(native.references)
        usable = [r for r in refs if r.title and not getattr(r, "skip_reason", None)]
        result = _Extraction(
            usable, native.skip_stats.total_raw, native.skip_stats.total_raw - len(usable), "hallucinator",
            note="no hanging-indent bibliography was detected; fell back to hallucinator's own extractor",
        )

    ctx.shared["hallucinator_extraction"] = result
    return result


@register(
    "reference_extraction",
    "Reference extraction (hallucinator)",
    module=MODULE,
    category="references",
    requires=("enable_hallucinator",),
    order=90,
)
def check_reference_extraction(ctx: CheckContext) -> Finding:
    """DEPRECATED backend: how much of the bibliography hallucinator could parse."""
    if not hallucinator_available():
        return ctx.skip("reference_extraction", "Reference extraction (hallucinator)",
                        f"The optional 'hallucinator' package is not installed ({_install_hint()}).",
                        category="references", remedy=_install_hint())
    try:
        ext = _extract(ctx)
    except ImportError:
        return ctx.skip("reference_extraction", "Reference extraction (hallucinator)",
                        f"The optional 'hallucinator' package failed to import ({_install_hint()}).",
                        category="references", remedy=_install_hint())

    minimum = int(ctx.conf("hallucinator.min_expected_references", 5))
    evidence = [Evidence(detail=f"segmentation strategy: {ext.strategy}",
                         measured=float(len(ext.references)),
                         expected=f"of {ext.entries_seen} bibliography entries")]
    if ext.note:
        evidence.append(Evidence(detail=ext.note))
    for ref in ext.references[:3]:
        evidence.append(Evidence(detail="parsed reference", quote=ref.title))

    if not ext.references:
        return ctx.warn(
            "reference_extraction", "Reference extraction (hallucinator)",
            "No references could be parsed from the bibliography, so reference verification "
            "cannot run. This is a limitation of the parser, not a finding about the paper.",
            category="references", evidence=evidence,
            remedy="Check that the paper has a 'References' section with selectable text.",
            confidence="low — extraction failure says nothing about the references themselves",
        )
    if len(ext.references) < minimum:
        return ctx.warn(
            "reference_extraction", "Reference extraction (hallucinator)",
            f"Only {len(ext.references)} of {ext.entries_seen} bibliography entries parsed cleanly; "
            "verification will cover a small part of the bibliography.",
            category="references", evidence=evidence,
            confidence="low — partial extraction, treat the verification result as a sample",
        )
    return ctx.ok(
        "reference_extraction", "Reference extraction (hallucinator)",
        f"Parsed {len(ext.references)} references from {ext.entries_seen} bibliography entries "
        f"({ext.skipped} entry/entries were not usable).",
        category="references", evidence=evidence,
    )


def _validator_config(ctx: CheckContext) -> Any:
    """Build a ValidatorConfig from the profile, letting explicit settings win."""
    from hallucinator import ValidatorConfig  # noqa: PLC0415 - optional dependency

    cfg = ValidatorConfig()
    # References that match nothing are the expensive ones: they fall through every
    # database, including arXiv's mandatory three-second gap and Semantic Scholar's
    # 429 backoff. That is latency, not work, so more workers overlap it.
    cfg.num_workers = int(ctx.conf("hallucinator.num_workers", 8))
    retries = ctx.conf("hallucinator.max_rate_limit_retries", None)
    if retries is not None:
        with suppress(AttributeError, TypeError, ValueError):
            cfg.max_rate_limit_retries = int(retries)
    cfg.db_timeout_secs = int(ctx.conf("hallucinator.timeout_secs", 10))
    disabled = [str(db) for db in (ctx.conf("hallucinator.disabled_dbs", []) or [])]
    if disabled:
        cfg.disabled_dbs = disabled
    mailto = ctx.conf("hallucinator.crossref_mailto", None)
    if mailto:
        cfg.crossref_mailto = str(mailto)
    cache = getattr(ctx.settings, "hallucinator_cache", None) or ctx.conf("hallucinator.cache_path", None)
    if cache:
        # The profile writes "~/.cache/...", which is a literal directory name to
        # anything that is not a shell. Expand it and make sure the parent exists.
        path = Path(str(cache)).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cfg.cache_path = str(path)
        except OSError:
            pass    # an unwritable cache location is not worth failing the run over
    return cfg


def _reference_cap(ctx: CheckContext) -> int:
    """0 means 'check everything'; the CLI setting overrides the profile."""
    cap = int(getattr(ctx.settings, "hallucinator_max_refs", 0) or 0)
    return cap or int(ctx.conf("hallucinator.max_references", 0) or 0)


def _progress_hook(ctx: CheckContext, total: int, state: dict[str, Any]) -> Any:
    """Translate hallucinator's events into a sub-status line for the runner.

    Without this the check looks frozen: a single blocking call that can spend
    minutes waiting on rate-limited databases (arXiv enforces a three-second gap
    between requests) with nothing on screen.
    """
    status = ctx.shared.get("status")

    def publish(text: str) -> None:
        state["last"] = text
        if callable(status):
            try:
                status("reference_hallucination", text)
            except Exception:  # a noisy UI must never fail the check
                pass

    publish(f"references 0/{total}")

    def hook(event: Any) -> None:
        try:
            kind = getattr(event, "event_type", "")
            if kind == "result":
                state["completed"] = state.get("completed", 0) + 1
                publish(f"references {state['completed']}/{total}")
            elif kind == "checking":
                title = (getattr(event, "title", "") or "")[:40]
                publish(f"references {state.get('completed', 0)}/{total}: {title}")
            elif kind == "rate_limit_wait":
                wait = float(getattr(event, "wait_ms", 0) or 0) / 1000.0
                publish(f"references {state.get('completed', 0)}/{total}: "
                        f"rate limited by {getattr(event, 'db_name', 'a database')}, waiting {wait:.0f}s")
            elif kind == "retry_pass":
                publish(f"retrying {getattr(event, 'count', 0)} unresolved references")
        except Exception:  # a noisy UI must never fail the check
            pass

    return hook


@register(
    "reference_hallucination",
    "Unverifiable references (hallucinator)",
    module=MODULE,
    category="references",
    requires=("enable_hallucinator",),
    order=91,
    # Blocking network work: the runner hands it to a worker thread so the
    # database lookups overlap with the model calls. Extraction (order 90) has
    # already run and warmed the shared cache by then.
    concurrent=True,
)
def check_reference_hallucination(ctx: CheckContext) -> Finding:
    """DEPRECATED backend: look every reference up in bibliographic databases."""
    if not hallucinator_available():
        return ctx.skip("reference_hallucination", "Unverifiable references (hallucinator)",
                        f"The optional 'hallucinator' package is not installed ({_install_hint()}).",
                        category="references", remedy=_install_hint())
    try:
        from hallucinator import Validator  # noqa: PLC0415 - optional dependency

        ext = _extract(ctx)
        cfg = _validator_config(ctx)
    except ImportError:
        return ctx.skip("reference_hallucination", "Unverifiable references (hallucinator)",
                        f"The optional 'hallucinator' package failed to import ({_install_hint()}).",
                        category="references", remedy=_install_hint())

    if not ext.references:
        return ctx.skip("reference_hallucination", "Unverifiable references (hallucinator)",
                        "No references could be parsed from the bibliography, so nothing was looked up.",
                        category="references")

    cap = _reference_cap(ctx)
    refs = ext.references[:cap] if cap else list(ext.references)

    # A wall-clock ceiling. These lookups hit up to ten rate-limited databases per
    # reference, so a large bibliography can run for a very long time; without a
    # cancel the whole tool appears to hang on one check.
    budget = float(ctx.conf("hallucinator.total_timeout_secs", 300))
    state: dict[str, Any] = {"completed": 0}
    validator = Validator(cfg)
    timer = threading.Timer(budget, validator.cancel) if budget > 0 else None
    started = time.monotonic()
    if timer is not None:
        timer.daemon = True
        timer.start()
    try:
        results = validator.check(refs, progress=_progress_hook(ctx, len(refs), state))
    finally:
        if timer is not None:
            timer.cancel()
        status = ctx.shared.get("status")
        if callable(status):
            try:
                status("reference_hallucination", "")
            except Exception:
                pass

    elapsed = time.monotonic() - started
    timed_out = len(results) < len(refs)
    timeout_note = ""
    if timed_out:
        timeout_note = (
            f" The lookup was stopped after {elapsed:.0f}s having checked {len(results)} of "
            f"{len(refs)} references; raise hallucinator.total_timeout_secs, narrow the run with "
            "--max-refs, or set a cache path so repeat runs are fast."
        )

    suspicious_statuses = {str(s) for s in (ctx.conf(
        "hallucinator.suspicious_statuses", ["not_found", "author_mismatch", "retracted"]) or [])}
    max_evidence = int(ctx.conf("hallucinator.max_evidence", 12))
    as_error = bool(ctx.conf("hallucinator.not_found_is_error", False))

    suspicious = [r for r in results if r.status in suspicious_statuses]
    verified = sum(1 for r in results if r.status == "verified")
    scope = f"the {len(results)} references checked"
    if cap and cap < len(ext.references):
        scope += f" (of {len(ext.references)} parsed; capped at {cap})"

    if not suspicious:
        return ctx.ok(
            "reference_hallucination", "Unverifiable references (hallucinator)",
            f"Every one of {scope} was found in at least one bibliographic database "
            f"({verified} verified).{timeout_note}",
            category="references",
        )

    evidence = [
        Evidence(
            page=None,
            detail=f"{r.status}" + (f" (source: {r.source})" if r.source else " (no database matched)"),
            quote=r.title or r.raw_citation,
        )
        for r in suspicious[:max_evidence]
    ]
    if len(suspicious) > max_evidence:
        evidence.append(Evidence(detail=f"...and {len(suspicious) - max_evidence} more"))

    message = (
        f"{len(suspicious)} of {scope} could not be confirmed in any bibliographic database "
        f"({verified} verified).{timeout_note} {_SUSPICION_NOTE} Workshop papers, theses, standards "
        "and very recent preprints are routinely missing from these databases."
    )
    remedy = (
        "Check each flagged entry by hand against the publisher's page or the arXiv listing before "
        f"changing anything. {_SUSPICION_NOTE} Fix genuinely wrong titles, authors or years; leave "
        "correct-but-unindexed entries alone."
    )
    reporter = ctx.error if as_error else ctx.warn
    return reporter(
        "reference_hallucination", "Unverifiable references (hallucinator)", message,
        category="references", evidence=evidence, remedy=remedy,
        confidence=("medium — profile promoted database misses to errors" if as_error
                    else "low — database coverage is incomplete; every hit needs human adjudication"),
    )
