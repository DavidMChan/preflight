"""The deterministic checks, end to end, against synthetic papers."""

from __future__ import annotations

from pathlib import Path

import pytest

from preflight.context import Settings
from preflight.models import Report, Severity
from preflight.runner import run_checks


def _run(path: Path, conference: str = "arr", track: str | None = "long") -> Report:
    return run_checks(path, conference, track, Settings.offline())


def _finding(report: Report, check_id: str):
    matches = [f for f in report.findings if f.check_id == check_id]
    assert matches, f"{check_id} did not run; ran: {sorted(f.check_id for f in report.findings)}"
    return matches[0]


def test_clean_paper_has_no_errors(clean_paper: Path) -> None:
    report = _run(clean_paper)
    assert report.errors == [], [f"{f.check_id}: {f.message}" for f in report.errors]
    assert report.passed


def test_page_limit_counts_content_pages_not_pdf_pages(clean_paper: Path) -> None:
    """12 PDF pages, but only 8 of content, so an 8-page limit passes."""
    finding = _finding(_run(clean_paper), "page_limit")
    assert finding.severity is Severity.PASS
    assert "8 content page" in finding.message


def test_page_limit_fires_when_content_overruns(overlong_paper: Path) -> None:
    finding = _finding(_run(overlong_paper), "page_limit")
    assert finding.severity is Severity.ERROR
    assert "page 11" in finding.message
    assert finding.evidence


def test_short_track_applies_a_tighter_limit(clean_paper: Path) -> None:
    """The same PDF passes as a long paper and fails as a short one."""
    assert _finding(_run(clean_paper, track="long"), "page_limit").severity is Severity.PASS
    assert _finding(_run(clean_paper, track="short"), "page_limit").severity is Severity.ERROR


def test_missing_limitations_is_an_error(no_limitations_paper: Path) -> None:
    finding = _finding(_run(no_limitations_paper), "limitations_present")
    assert finding.severity is Severity.ERROR
    assert "desk rejection" in finding.message
    assert finding.cfp_reference       # the CFP sentence is attached


def test_wrong_paper_size_is_an_error(letter_paper: Path) -> None:
    finding = _finding(_run(letter_paper), "paper_size")
    assert finding.severity is Severity.ERROR
    assert "A4" in finding.message


def test_letter_paper_passes_the_neurips_profile(letter_paper: Path) -> None:
    """The same file is correct for a US Letter venue — the rule lives in the profile."""
    finding = _finding(run_checks(letter_paper, "neurips", "main", Settings.offline()), "paper_size")
    assert finding.severity is Severity.PASS


def test_clean_paper_passes_geometry_and_typography(clean_paper: Path) -> None:
    report = _run(clean_paper)
    for check_id in ("paper_size", "margins", "column_layout", "body_font_size"):
        assert _finding(report, check_id).severity is Severity.PASS, check_id


def test_appendix_must_follow_references(tmp_path: Path) -> None:
    from conftest import build_paper

    # Appendix on page 8, references on page 9.
    path = build_paper(tmp_path / "bad_order.pdf", pages=10, limitations_page=None,
                       references_page=9, appendix_page=8)
    finding = _finding(_run(path), "appendix_position")
    assert finding.severity is Severity.ERROR


def test_unverifiable_items_are_always_reported(clean_paper: Path) -> None:
    report = _run(clean_paper)
    unverifiable = report.of(Severity.UNVERIFIABLE)
    assert len(unverifiable) >= 6
    ids = {f.check_id for f in unverifiable}
    assert "unverifiable.dual_submission" in ids
    assert "unverifiable.originality" in ids


def test_offline_mode_makes_no_network_calls(clean_paper: Path) -> None:
    """--offline must exclude every check that would reach out."""
    report = _run(clean_paper)
    assert not [f for f in report.findings if f.check_id.startswith(("llm_", "rnlp_", "reference_"))]
    assert report.meta["llm_model"] is None
    assert report.meta["hallucinator"] is False


def test_network_checks_are_on_by_default() -> None:
    """Model checks and the built-in reference checker run unless opted out."""
    settings = Settings()
    assert settings.enable_llm and settings.enable_scores and settings.enable_refcheck
    # The third-party backend is deprecated and opt-in.
    assert settings.enable_hallucinator is False
    offline = Settings.offline()
    assert not (offline.enable_llm or offline.enable_refcheck or offline.enable_hallucinator)


def test_profile_severity_override_is_applied(no_limitations_paper: Path) -> None:
    """NeurIPS softens a missing Limitations section to a warning; ARR does not."""
    arr = _finding(_run(no_limitations_paper), "limitations_present")
    neurips = _finding(run_checks(no_limitations_paper, "neurips", "main", Settings.offline()),
                       "limitations_present")
    assert arr.severity is Severity.ERROR
    assert neurips.severity is Severity.WARNING


def test_report_metadata_describes_the_run(clean_paper: Path) -> None:
    report = _run(clean_paper)
    assert report.meta["concurrent_checks"] == 0     # offline: nothing to overlap
    assert report.meta["page_count"] == 12
    assert report.meta["content_page_limit"] == 8
    assert "core.geometry" in report.meta["modules"]
    assert report.meta["profile_lineage"] == ["arr", "acl", "base"]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _run(tmp_path / "nope.pdf")


def test_real_paper_has_no_errors(real_paper: Path) -> None:
    report = _run(real_paper)
    assert report.errors == [], [f"{f.check_id}: {f.message}" for f in report.errors]


def test_no_llm_flag_wins_over_default_scoring() -> None:
    """--no-llm must disable scoring too; scoring is itself a model call."""
    from preflight.cli import _build_settings

    settings = _build_settings(llm=False, scores=True, hallucinator=False, model="m",
                               openai_key="k", max_refs=0, strict=False)
    assert not settings.enable_llm
    assert not settings.enable_scores
    assert settings.enable_refcheck        # deterministic + database work still runs


def test_offline_disables_every_network_path() -> None:
    from preflight.cli import _build_settings

    settings = _build_settings(llm=True, scores=True, hallucinator=True, model="m",
                               openai_key="k", max_refs=0, strict=False, offline=True)
    assert not (settings.enable_llm or settings.enable_scores
                or settings.enable_refcheck or settings.enable_hallucinator)
