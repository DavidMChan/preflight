"""Findings, evidence rendering, and report aggregation."""

from __future__ import annotations

import json

from preflight.models import Evidence, Finding, Report, Severity


def test_evidence_renders_measurement_and_expectation() -> None:
    ev = Evidence(page=5, detail="text bleeds into the LEFT margin", measured=34.0,
                  expected=">= 69.0 pt", quote="a  long   quote")
    rendered = ev.render()
    assert "page 5" in rendered
    assert "measured 34.0, expected >= 69.0 pt" in rendered
    assert '"a long quote"' in rendered      # whitespace is normalised


def test_evidence_truncates_long_quotes() -> None:
    ev = Evidence(quote="x" * 500)
    assert len(ev.render()) < 200
    assert ev.render().endswith('..."')


def test_report_counts_and_ordering() -> None:
    report = Report(pdf_path="p.pdf", conference="arr", track="long")
    report.add(Finding("a", "A", Severity.PASS, "ok"))
    report.add(Finding("b", "B", Severity.ERROR, "bad"))
    report.add(Finding("c", "C", Severity.WARNING, "hmm"))
    assert report.counts() == {"error": 1, "warning": 1, "pass": 1, "skipped": 0, "unverifiable": 0}
    assert not report.passed
    assert [f.check_id for f in report.sorted_findings()] == ["b", "c", "a"]


def test_report_passes_with_only_warnings() -> None:
    report = Report(pdf_path="p.pdf", conference="arr", track="long")
    report.add(Finding("c", "C", Severity.WARNING, "hmm"))
    assert report.passed          # warnings do not block
    assert report.warnings


def test_report_json_round_trips() -> None:
    report = Report(pdf_path="p.pdf", conference="arr", track="long")
    report.add(Finding("a", "A", Severity.ERROR, "bad", evidence=[Evidence(page=2, detail="d")]))
    data = json.loads(report.to_json())
    assert data["counts"]["error"] == 1
    assert data["findings"][0]["evidence"][0]["page"] == 2
    assert data["passed"] is False


def test_add_accepts_none_and_iterables() -> None:
    report = Report(pdf_path="p.pdf", conference="arr", track="long")
    report.add(None)
    report.add([Finding("a", "A", Severity.PASS, "ok"), Finding("b", "B", Severity.PASS, "ok")])
    assert len(report.findings) == 2
