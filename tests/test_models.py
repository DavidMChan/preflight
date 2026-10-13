"""Findings, evidence rendering, and report aggregation."""

from __future__ import annotations

import json

from preflight.models import Evidence, Finding, Report, Severity


def test_evidence_renders_measurement_and_expectation() -> None:
    ev = Evidence(page=5, detail="text bleeds into the LEFT margin", measured=34.5,
                  expected=">= 69.0 pt", quote="a  long   quote")
    rendered = ev.render()
    assert "page 5" in rendered
    assert "measured 34.5, expected >= 69.0 pt" in rendered
    # A whole number keeps no decimal: counts render beside integer expectations.
    assert "measured 34," in Evidence(measured=34.0, expected="34").render()
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


def test_the_report_tags_each_finding_offline_online_or_llm() -> None:
    from rich.console import Console

    from preflight.report import print_report, to_markdown

    report = Report(pdf_path="p.pdf", conference="iclr", track="main",
                    meta={"llm_model": "gpt-x", "refcheck": True})
    report.add(Finding("margins", "Margins", Severity.PASS, "fine", uses=()))
    report.add(Finding("reference_details", "Reference details", Severity.ERROR, "wrong",
                       uses=("network", "llm")))
    report.add(Finding("rnlp_c1", "C1", Severity.UNVERIFIABLE, "not checkable", uses=()))

    console = Console(record=True, width=200)
    print_report(report, console)
    text = console.export_text()
    assert "Reference details  [reference_details]  online + LLM" in text
    assert "Margins  [margins]  offline" in text
    assert "[rnlp_c1]  offline" not in text          # never checked, so no tag
    assert "LLM — read by gpt-x" in text

    markdown = to_markdown(report)
    assert "- **Reference details** (`reference_details`, online + LLM) — wrong" in markdown


def test_findings_show_everything_unless_minimal() -> None:
    from rich.console import Console

    from preflight.report import render_finding, to_markdown

    finding = Finding("reference_details", "Reference details", Severity.ERROR, "wrong",
                      evidence=[Evidence(detail=f"entry {i}") for i in range(10)],
                      cfp_reference="Citations must be accurate.")

    def shown(**kw) -> str:
        console = Console(record=True, width=200)
        console.print(render_finding(finding, **kw))
        return console.export_text()

    full = shown()
    assert "entry 9" in full and "CFP: Citations must be accurate." in full
    collapsed = shown(verbose=False)
    assert "entry 3" in collapsed and "entry 4" not in collapsed
    assert "and 6 more (preflight show reference_details)" in collapsed
    assert "CFP:" not in collapsed

    report = Report(pdf_path="p.pdf", conference="arr", track="long")
    report.add(finding)
    assert "entry 9" in to_markdown(report)
    assert "entry 9" not in to_markdown(report, verbose=False)


def test_the_html_report_escapes_and_tags_every_finding() -> None:
    from preflight.report import to_html

    report = Report(pdf_path="p.pdf", conference="arr", track="long",
                    meta={"llm_model": "gpt-x", "refcheck": True})
    report.add(Finding("reference_details", "Reference details", Severity.ERROR, "cites <script>",
                       evidence=[Evidence(detail=f"entry {i}") for i in range(10)],
                       remedy="fix it", uses=("network", "llm")))
    report.add(Finding("margins", "Margins", Severity.PASS, "fine", uses=()))
    report.scores = {"model": "gpt-x", "dimensions": {"clarity": {"score": 4, "justification": "ok"}}}

    page = to_html(report)
    assert page.startswith("<!DOCTYPE html>")
    assert "cites &lt;script&gt;" in page and "<script>" not in page
    assert '<span class="m-llm">online + LLM</span>' in page
    assert '<span class="m-offline">offline</span>' in page
    assert "Not ready to submit" in page and "fix the errors below" in page
    assert "entry 9" in page and "Clarity" in page
    collapsed = to_html(report, verbose=False)
    assert "entry 5" in collapsed and "entry 6" not in collapsed
    assert "and 4 more (preflight show reference_details)" in collapsed


def test_the_pdf_report_lays_out_every_finding() -> None:
    import pymupdf

    from preflight.report import to_pdf

    report = Report(pdf_path="p.pdf", conference="arr", track="long")
    report.add(Finding("page_limit", "Content page limit", Severity.ERROR, "too long",
                       evidence=[Evidence(page=9, detail=f"line {i}") for i in range(200)]))
    report.add(Finding("margins", "Margins", Severity.PASS, "fine", uses=()))

    doc = pymupdf.open("pdf", to_pdf(report))
    assert doc.page_count > 1                       # long evidence flows onto later pages
    assert doc.metadata["title"] == "ARR pre-flight — p.pdf"
    text = "".join(page.get_text(flags=0) for page in doc)
    assert "Content page limit" in text and "line 199" in text and "Margins" in text
    assert f"page {doc.page_count} of {doc.page_count}" in text
