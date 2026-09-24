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
    report.add(Finding("margins", "Margins", Severity.PASS, "fine", uses=()))
    report.add(Finding("llm_anonymity", "Anonymity (model)", Severity.PASS, "fine", uses=("llm",)))
    report.scores = {"model": "m", "dimensions": {"clarity": {"score": 4, "justification": "ok"}}}
    return report


def test_save_and_reload_round_trips(tmp_path: Path) -> None:
    saved = cache.save(_report(), root=tmp_path)
    assert saved is not None and saved.is_file()

    loaded = cache.load(saved)
    assert loaded is not None
    assert loaded.counts() == {"error": 1, "warning": 0, "pass": 2, "skipped": 0, "unverifiable": 0}
    modes = {f.check_id: f.mode for f in loaded.findings}
    # A finding recorded before tagging has no mode, rather than a guessed one.
    assert modes == {"page_limit": None, "margins": "offline", "llm_anonymity": "LLM"}
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


def test_check_is_verbose_unless_minimal(tmp_path: Path, clean_paper: Path, monkeypatch) -> None:
    from typer.testing import CliRunner

    from preflight import picker
    from preflight.cli import app

    monkeypatch.setattr(cache, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(picker, "LAST_CHOICE", tmp_path / "last-conference")
    run = ["check", str(clean_paper), "-c", "arr", "-t", "long", "--offline"]

    verbose = CliRunner().invoke(app, run)
    minimal = CliRunner().invoke(app, [*run, "--minimal"])
    assert verbose.exit_code == minimal.exit_code == 0
    # Collapsed output points at the saved run for the rest; the full output has nothing hidden.
    assert "Full evidence kept as run" in minimal.output
    assert "Full evidence kept as run" not in verbose.output
    assert CliRunner().invoke(app, ["show", "--minimal"]).exit_code == 0
