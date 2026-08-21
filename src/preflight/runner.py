"""Orchestration: open a PDF, run the applicable checks, assemble a report.

Checks run in three phases:

1. **Deterministic.** Geometry, typography and structure. These are pure
   computation over the parsed PDF, take milliseconds, and run in order.
2. **Concurrent.** Everything that waits on a network: the model-backed checks
   (on the async client) and the bibliographic lookups (blocking, so handed to a
   worker thread). These all overlap.
3. **Aggregate.** Checks that summarise what the earlier phases found, so they
   must not start until phase 2 is finished.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from .context import CheckContext, Settings
from .document import Document
from .llm.client import AsyncLLMClient
from .models import Progress, Report, Severity
from .profile import Profile, load_profile
from .registry import Check, load_builtin_checks

ProgressFn = Callable[[Progress], None]
"""Called with a :class:`~preflight.models.Progress` snapshot as the run advances."""


def run_checks(
    pdf_path: str | Path,
    profile: Profile | str = "arr",
    track: str | None = None,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
) -> Report:
    """Run every check that applies to ``profile`` against ``pdf_path``."""
    return asyncio.run(run_checks_async(pdf_path, profile, track, settings, progress))


async def run_checks_async(
    pdf_path: str | Path,
    profile: Profile | str = "arr",
    track: str | None = None,
    settings: Settings | None = None,
    progress: ProgressFn | None = None,
) -> Report:
    resolved = load_profile(profile) if isinstance(profile, str) else profile
    settings = settings or Settings()
    registry = load_builtin_checks()

    doc = await asyncio.to_thread(Document, pdf_path)
    ctx: CheckContext | None = None
    try:
        ctx = CheckContext(doc=doc, profile=resolved, track=resolved.track(track), settings=settings)
        if progress is not None:
            ctx.shared["progress"] = progress
        ctx.shared["llm_client"] = AsyncLLMClient(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout,
            max_concurrency=settings.llm_concurrency,
        )

        checks: list[Check] = registry.for_context(ctx)
        report = Report(
            pdf_path=str(Path(pdf_path).resolve()),
            conference=resolved.key,
            track=ctx.track.name,
        )

        deterministic = [c for c in checks if not c.aggregate and not c.runs_concurrently]
        concurrent = [c for c in checks if not c.aggregate and c.runs_concurrently]
        aggregate = [c for c in checks if c.aggregate]

        tracker = _Tracker(total=len(checks), report_to=progress)
        # Long-running checks publish their own sub-status through this hook.
        ctx.shared["status"] = tracker.detail

        for check in deterministic:
            tracker.start(check)
            report.add(await check.arun(ctx))
            tracker.finish(check)

        if concurrent:
            async def run_one(check: Check) -> list:
                tracker.start(check)
                try:
                    return await check.arun(ctx)
                finally:
                    tracker.finish(check)

            for findings in await asyncio.gather(*(run_one(c) for c in concurrent)):
                report.add(findings)

        for check in aggregate:
            tracker.start(check)
            report.add(await check.arun(ctx))
            tracker.finish(check)

        tracker.done()

        report.scores = dict(ctx.shared.get("scores") or {})
        report.meta = _build_meta(doc, resolved, ctx, checks, settings)
        return report
    finally:
        await _close_client(ctx)
        doc.close()


class _Tracker:
    """Keeps the set of in-flight checks, so progress reports what is pending."""

    def __init__(self, total: int, report_to: ProgressFn | None) -> None:
        self.total = total
        self.report_to = report_to
        self.completed = 0
        self.running: dict[str, str] = {}
        self.details: dict[str, str] = {}

    def start(self, check: Check) -> None:
        self.running[check.id] = check.title
        self._emit()

    def finish(self, check: Check) -> None:
        self.running.pop(check.id, None)
        self.details.pop(check.id, None)
        self.completed += 1
        self._emit()

    def detail(self, check_id: str, text: str) -> None:
        """Sub-status from inside a check, e.g. "references 12/62"."""
        if text:
            self.details[check_id] = text
        else:
            self.details.pop(check_id, None)
        self._emit()

    def done(self) -> None:
        self.running.clear()
        self.details.clear()
        self.completed = self.total
        self._emit()

    def _emit(self) -> None:
        if self.report_to is None:
            return
        snapshot = Progress(
            done=self.completed,
            total=self.total,
            running=tuple(self.running.values()),
            details=tuple(self.details.values()),
        )
        try:
            self.report_to(snapshot)
        except Exception:  # a noisy UI must never fail the run
            pass


async def _close_client(ctx: CheckContext | None) -> None:
    client = ctx.shared.get("llm_client") if ctx is not None else None
    if isinstance(client, AsyncLLMClient):
        await client.aclose()


def _build_meta(
    doc: Document,
    profile: Profile,
    ctx: CheckContext,
    checks: list[Check],
    settings: Settings,
) -> dict[str, object]:
    return {
        "profile_name": profile.name,
        "profile_source": profile.source,
        "profile_lineage": list(profile.lineage),
        "modules": sorted({c.module for c in checks}),
        "checks_run": len(checks),
        "concurrent_checks": sum(1 for c in checks if c.runs_concurrently and not c.aggregate),
        "page_count": doc.page_count,
        "page_size_pt": [round(doc.pages[0].width, 1), round(doc.pages[0].height, 1)],
        "body_font_size_pt": doc.body_font_size,
        "dominant_font": doc.dominant_font[0],
        "column_bands": [[round(a, 1), round(b, 1)] for a, b in doc.column_bands],
        "track_description": ctx.track.description,
        "content_page_limit": ctx.track.content_page_limit,
        "llm_model": settings.llm_model if settings.enable_llm else None,
        "refcheck": settings.enable_refcheck,
        "hallucinator": settings.enable_hallucinator,
        "pdf_metadata": {k: v for k, v in doc.metadata.items() if v},
    }


def exit_code(report: Report) -> int:
    """0 when nothing blocks, 1 on errors."""
    return 1 if report.errors else 0


def strict_exit_code(report: Report) -> int:
    if report.errors:
        return 1
    if report.of(Severity.WARNING):
        return 2
    return 0
