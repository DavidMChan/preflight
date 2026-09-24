"""Runner phasing, concurrency, and progress reporting."""

from __future__ import annotations

import asyncio
from pathlib import Path

from preflight.context import Settings
from preflight.models import Progress
from preflight.registry import load_builtin_checks
from preflight.runner import run_checks, run_checks_async


def test_checks_are_partitioned_into_phases() -> None:
    registry = load_builtin_checks()
    checks = list(registry)
    # Aggregates read what the other phases left behind, so they must run last.
    assert set(c.id for c in checks if c.aggregate) == {"rnlp_summary", "llm_context_coverage"}
    # Every model-backed check is async, so they can overlap.
    for check in checks:
        if check.requires == ("enable_llm",) and not check.aggregate:
            assert check.is_async, f"{check.id} would serialise the run"
    # The blocking bibliographic lookup opts into a worker thread.
    assert registry.checks["reference_hallucination"].runs_concurrently
    assert not registry.checks["reference_hallucination"].is_async


def test_deterministic_checks_do_not_run_concurrently() -> None:
    registry = load_builtin_checks()
    for check_id in ("paper_size", "margins", "page_limit", "limitations_present"):
        assert not registry.checks[check_id].runs_concurrently


def test_progress_reports_what_is_still_running(clean_paper: Path) -> None:
    seen: list[Progress] = []
    run_checks(clean_paper, "arr", "long", Settings.offline(), progress=seen.append)
    assert seen
    # Completion never goes backwards, and the run ends complete and idle.
    assert [p.done for p in seen] == sorted(p.done for p in seen)
    assert seen[-1].finished
    assert seen[-1].label() == "done"
    assert seen[-1].done == seen[-1].total
    # While a check is in flight it is named, rather than the last one to finish.
    in_flight = [p for p in seen if p.running]
    assert in_flight
    assert all(p.label() for p in in_flight)


def test_progress_label_prefers_sub_status_detail() -> None:
    p = Progress(done=3, total=10, running=("Unverifiable references",), details=("references 12/62",))
    assert p.label() == "references 12/62"
    assert Progress(done=3, total=10, running=("A", "B")).label() == "A, B"
    assert Progress(done=10, total=10).finished


def test_async_entry_point_matches_the_sync_one(clean_paper: Path) -> None:
    sync = run_checks(clean_paper, "arr", "long", Settings.offline())
    other = asyncio.run(run_checks_async(clean_paper, "arr", "long", Settings.offline()))
    assert sync.counts() == other.counts()
    assert {f.check_id for f in sync.findings} == {f.check_id for f in other.findings}


def test_a_crashing_check_does_not_sink_the_report(clean_paper: Path, monkeypatch) -> None:
    registry = load_builtin_checks()
    check = registry.checks["paper_size"]
    monkeypatch.setattr(check, "fn", lambda ctx: (_ for _ in ()).throw(RuntimeError("boom")))
    report = run_checks(clean_paper, "arr", "long", Settings.offline())
    finding = next(f for f in report.findings if f.check_id == "paper_size")
    assert finding.severity.value == "skipped"
    assert "boom" in finding.message
    # ...and the rest of the run still happened.
    assert len(report.findings) > 10


def test_every_check_says_whether_it_is_offline_online_or_a_model() -> None:
    from preflight.models import LLM, NETWORK

    registry = load_builtin_checks()
    for check in registry:
        if "enable_llm" in check.requires:
            assert LLM in check.uses, f"{check.id} calls a model but is not marked LLM"
        if {"enable_refcheck", "enable_hallucinator"} & set(check.requires):
            assert NETWORK in check.uses, f"{check.id} looks references up but is not marked online"
        if not check.requires:
            assert check.mode == "offline", check.id
    assert registry.checks["llm_anonymity"].mode == "LLM"
    assert registry.checks["reference_details"].mode == "online + LLM"
    assert registry.checks["margins"].mode == "offline"


def test_an_offline_run_tags_every_finding_offline(clean_paper: Path) -> None:
    report = run_checks(clean_paper, "arr", "long", Settings.offline())
    assert {f.mode for f in report.findings} == {"offline"}
    assert all(f.to_dict()["mode"] == "offline" for f in report.findings)


def test_a_model_the_check_can_do_without_is_not_claimed(clean_paper: Path) -> None:
    from preflight.context import CheckContext
    from preflight.document import Document
    from preflight.profile import load_profile

    registry = load_builtin_checks()
    profile = load_profile("arr")
    doc = Document(clean_paper)
    try:
        def uses(check_id: str, settings: Settings) -> str:
            ctx = CheckContext(doc=doc, profile=profile, track=profile.track("long"), settings=settings)
            return registry.checks[check_id].uses_in(ctx)

        assert uses("reference_details", Settings(openai_api_key="k")) == ("network", "llm")
        assert uses("reference_details", Settings(enable_llm=False)) == ("network",)
        assert uses("reference_details", Settings(openai_api_key=None)) == ("network",)
        # A check that is nothing but a model call stays one, even when it cannot run.
        assert uses("llm_anonymity", Settings(openai_api_key=None)) == ("llm",)
    finally:
        doc.close()
