"""The terminal UI, driven headlessly."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from preflight.context import Settings
from preflight.models import Report
from preflight.profile import load_profile
from preflight.tui.app import PreflightApp


@dataclass
class Driven:
    """What the UI looked like while it was still alive."""

    report: Report | None
    sections: list[str]
    status: str
    settings: Settings
    verbose: bool


def _drive(pdf: Path, keys: tuple[str, ...] = ()) -> Driven:
    app = PreflightApp(paths=[pdf], profile=load_profile("arr"), track="long",
                       settings=Settings.offline())
    captured: dict[str, object] = {}

    async def run() -> None:
        async with app.run_test() as pilot:
            for _ in range(60):
                if app.report is not None:
                    break
                await pilot.pause(0.25)
            for key in keys:
                await pilot.press(key)
            await pilot.pause(0.1)
            # The DOM only exists inside this block, so snapshot it here.
            captured["sections"] = [str(node.label) for node in app.query_one("#tree").root.children]
            captured["detail"] = app.query_one("#detail") is not None

    asyncio.run(run())
    assert captured.get("detail"), "the detail pane was never mounted"
    return Driven(report=app.report, sections=list(captured["sections"]),  # type: ignore[arg-type]
                  status=app.status, settings=app.settings, verbose=app.verbose)


def test_tui_renders_a_report(clean_paper: Path) -> None:
    driven = _drive(clean_paper)
    assert driven.report is not None
    assert driven.report.counts()["error"] == 0
    assert any("Passed" in section for section in driven.sections)


def test_tui_groups_errors_first(overlong_paper: Path) -> None:
    driven = _drive(overlong_paper)
    assert driven.report is not None and driven.report.errors
    assert "Errors" in driven.sections[0]


def test_toggles_change_settings_without_rerunning(clean_paper: Path) -> None:
    driven = _drive(clean_paper, keys=("l", "b", "v"))
    assert driven.settings.enable_llm is True          # toggled on from offline
    assert driven.settings.enable_hallucinator is True
    assert driven.verbose is True
    assert "re-run" in driven.status
