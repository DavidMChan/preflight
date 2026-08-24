"""Run caching and re-reading."""

from __future__ import annotations

from pathlib import Path

from preflight import cache
from preflight.models import Evidence, Finding, Report, Severity


def _report() -> Report:
    report = Report(pdf_path="/tmp/paper.pdf", conference="arr", track="long")
    report.add(Finding("page_limit", "Content page limit", Severity.ERROR, "too long",
                       evidence=[Evidence(page=9, detail="first unlimited section", measured=9.0,
                                          expected="<= 8")], remedy="cut"))
    report.add(Finding("margins", "Margins", Severity.PASS, "fine"))
    report.scores = {"model": "m", "dimensions": {"clarity": {"score": 4, "justification": "ok"}}}
    return report


def test_save_and_reload_round_trips(tmp_path: Path) -> None:
    saved = cache.save(_report(), root=tmp_path)
    assert saved is not None and saved.is_file()

    loaded = cache.load(saved)
    assert loaded is not None
    assert loaded.counts() == {"error": 1, "warning": 0, "pass": 1, "skipped": 0, "unverifiable": 0}
    finding = next(f for f in loaded.findings if f.check_id == "page_limit")
    assert finding.severity is Severity.ERROR
    assert finding.remedy == "cut"
    assert finding.evidence[0].page == 9
    assert finding.evidence[0].measured == 9.0
    assert loaded.scores["dimensions"]["clarity"]["score"] == 4


def test_list_and_resolve(tmp_path: Path) -> None:
    saved = cache.save(_report(), root=tmp_path)
    assert saved is not None
    runs = cache.list_runs(root=tmp_path)
    assert len(runs) == 1
    entry = runs[0]
    assert entry.name == "paper.pdf"
    assert entry.counts["error"] == 1

    assert cache.resolve(entry.run_id, root=tmp_path) == saved
    assert cache.resolve("paper.pdf", root=tmp_path) == saved
    assert cache.resolve("nope", root=tmp_path) is None
    assert cache.resolve(None, root=tmp_path) == saved      # most recent


def test_prune_keeps_the_newest(tmp_path: Path) -> None:
    for _ in range(5):
        cache.save(_report(), root=tmp_path)
    assert len(cache.list_runs(root=tmp_path, limit=100)) == 5
    removed = cache.prune(keep=2, root=tmp_path)
    assert removed == 3
    assert len(cache.list_runs(root=tmp_path, limit=100)) == 2


def test_save_never_raises_on_a_bad_root(tmp_path: Path) -> None:
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory")
    assert cache.save(_report(), root=blocker) is None


def test_load_rejects_junk(tmp_path: Path) -> None:
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    assert cache.load(junk) is None


# ---------------------------------------------------------------------------
# Addressing a check inside an older run: `preflight show <run>/<check>`
# ---------------------------------------------------------------------------


def _cli(tmp_path: Path, monkeypatch, *args: str):
    from typer.testing import CliRunner

    from preflight.cli import app

    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path)
    return CliRunner().invoke(app, ["show", *args])


def _report_saying(message: str) -> Report:
    report = Report(pdf_path="/tmp/paper.pdf", conference="arr", track="long")
    report.add(Finding("margins", "Margins", Severity.PASS, message))
    return report


def test_show_reads_a_check_out_of_an_older_run(tmp_path: Path, monkeypatch) -> None:
    """The whole point: the cache keeps every run, so addressing should too."""
    older = cache.save(_report_saying("the older run"), root=tmp_path)
    cache.save(_report_saying("the newer run"), root=tmp_path)
    assert older is not None
    run_id = next(r.run_id for r in cache.list_runs(root=tmp_path) if r.path == older)

    result = _cli(tmp_path, monkeypatch, f"{run_id}/margins")
    assert result.exit_code == 0
    assert "the older run" in result.output
    assert "the newer run" not in result.output


def test_a_bare_check_id_still_means_the_latest_run(tmp_path: Path, monkeypatch) -> None:
    cache.save(_report_saying("the older run"), root=tmp_path)
    cache.save(_report_saying("the newer run"), root=tmp_path)

    result = _cli(tmp_path, monkeypatch, "margins")
    assert result.exit_code == 0
    assert "the newer run" in result.output


def test_an_unknown_run_says_so_rather_than_falling_back(tmp_path: Path, monkeypatch) -> None:
    """Falling back to the latest run would answer a question nobody asked."""
    cache.save(_report_saying("the only run"), root=tmp_path)

    result = _cli(tmp_path, monkeypatch, "deadbeef/margins")
    assert result.exit_code == 2
    assert "No cached run matching" in result.output
    assert "the only run" not in result.output


def test_an_unknown_check_names_the_run_it_looked_in(tmp_path: Path, monkeypatch) -> None:
    cache.save(_report_saying("the only run"), root=tmp_path)
    run_id = cache.list_runs(root=tmp_path)[0].run_id

    result = _cli(tmp_path, monkeypatch, f"{run_id}/no_such_check")
    assert result.exit_code == 2
    assert "no_such_check" in result.output
    assert run_id in result.output


def test_a_run_named_by_a_path_is_not_split_into_run_and_check(tmp_path: Path, monkeypatch) -> None:
    """A PDF path has slashes of its own; whatever resolves whole is a run."""
    cache.save(_report_saying("the only run"), root=tmp_path)

    result = _cli(tmp_path, monkeypatch, "/tmp/paper.pdf")
    assert result.exit_code == 0
    assert "the only run" in result.output
