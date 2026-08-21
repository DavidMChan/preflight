"""Choosing a conference: the picker, and what happens without a terminal."""

from __future__ import annotations

import io

import pytest
from rich.console import Console
from typer.testing import CliRunner

from preflight import picker
from preflight.cli import app
from preflight.profile import load_profile


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(picker, "LAST_CHOICE", tmp_path / "last-conference")


class _Answers:
    """Feed scripted replies to ``console.input`` (which calls ``input``)."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)

    def __call__(self, *_args) -> str:
        return self.replies.pop(0)


def _console() -> Console:
    return Console(file=io.StringIO(), width=100)


def test_bundled_profiles_are_offered():
    keys = [c.key for c in picker._choices()]
    assert "arr" in keys and "neurips" in keys
    assert all(c.name for c in picker._choices())


def test_numbered_picker_accepts_a_number(monkeypatch):
    monkeypatch.setattr("builtins.input", _Answers("2"))
    choices = picker._choices()
    assert picker._numbered_picker(_console(), choices, 0, None) == choices[1].key


def test_numbered_picker_accepts_a_key_and_reprompts_on_junk(monkeypatch):
    monkeypatch.setattr("builtins.input", _Answers("nope", "neurips"))
    choices = picker._choices()
    assert picker._numbered_picker(_console(), choices, 0, None) == "neurips"


def test_numbered_picker_empty_answer_takes_the_highlighted_row(monkeypatch):
    monkeypatch.setattr("builtins.input", _Answers(""))
    choices = picker._choices()
    assert picker._numbered_picker(_console(), choices, 1, None) == choices[1].key


def test_last_choice_round_trips_and_ignores_paths():
    assert picker.remembered() is None
    picker.remember("neurips")
    assert picker.remembered() == "neurips"
    picker.remember("./somewhere/custom.yaml")
    assert picker.remembered() == "neurips"


def test_picker_refuses_without_a_terminal(monkeypatch):
    monkeypatch.setattr(picker, "interactive", lambda: False)
    with pytest.raises(picker.PickerUnavailable):
        picker.pick_conference(Console(file=io.StringIO()))


def test_cli_without_a_conference_is_a_usage_error(tmp_path):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")
    result = CliRunner().invoke(app, ["check", str(pdf), "--offline"])
    assert result.exit_code == 2
    assert "--conference" in result.output


def test_cli_prompts_when_no_conference_is_given(tmp_path, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr("preflight.cli.pick_conference", lambda console: (asked.append("yes"), "arr")[1])
    result = CliRunner().invoke(app, ["check", str(tmp_path / "missing.pdf"), "--offline"])
    assert asked == ["yes"]
    assert "No such file" in result.output


def test_cli_remembers_an_explicit_conference(tmp_path):
    CliRunner().invoke(app, ["check", str(tmp_path / "missing.pdf"), "-c", "neurips", "--offline"])
    assert picker.remembered() == "neurips"


# -- track selection ------------------------------------------------------


def test_track_choices_are_labelled_by_page_limit():
    labels = {c.key: c.name for c in picker._track_choices(load_profile("arr"))}
    assert labels == {"long": "8 content pages", "short": "4 content pages", "demo": "6 content pages"}


def test_pick_track_returns_none_for_a_single_track_venue(monkeypatch):
    monkeypatch.setattr(picker, "interactive", lambda: True)
    assert picker.pick_track(_console(), load_profile("neurips")) is None


def test_pick_track_returns_none_without_a_terminal(monkeypatch):
    monkeypatch.setattr(picker, "interactive", lambda: False)
    assert picker.pick_track(_console(), load_profile("arr")) is None


def test_numbered_picker_selects_a_track(monkeypatch):
    monkeypatch.setattr("builtins.input", _Answers("2"))
    choices = picker._track_choices(load_profile("arr"))
    assert picker._numbered_picker(_console(), choices, 0, "long") == "short"


def test_numbered_picker_accepts_a_track_by_name(monkeypatch):
    monkeypatch.setattr("builtins.input", _Answers("demo"))
    choices = picker._track_choices(load_profile("arr"))
    assert picker._numbered_picker(_console(), choices, 0, "long") == "demo"


def test_picked_conference_is_followed_by_a_track_question(tmp_path, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr("preflight.cli.pick_conference", lambda console: "arr")
    monkeypatch.setattr("preflight.cli.pick_track",
                        lambda console, profile: (asked.append(profile.key), "short")[1])
    CliRunner().invoke(app, ["check", str(tmp_path / "missing.pdf"), "--offline"])
    assert asked == ["arr"]


def test_explicit_conference_does_not_ask_for_a_track(tmp_path, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr("preflight.cli.pick_track",
                        lambda console, profile: asked.append(profile.key))
    CliRunner().invoke(app, ["check", str(tmp_path / "missing.pdf"), "-c", "arr", "--offline"])
    assert asked == []


def test_explicit_track_is_not_second_guessed(tmp_path, monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr("preflight.cli.pick_conference", lambda console: "arr")
    monkeypatch.setattr("preflight.cli.pick_track",
                        lambda console, profile: asked.append(profile.key))
    CliRunner().invoke(app, ["check", str(tmp_path / "missing.pdf"), "-t", "short", "--offline"])
    assert asked == []


def test_picked_track_is_applied_to_the_run(clean_paper, monkeypatch):
    """8 content pages: within the ARR long limit, over the short one."""
    monkeypatch.setattr("preflight.cli.pick_conference", lambda console: "arr")

    monkeypatch.setattr("preflight.cli.pick_track", lambda console, profile: "long")
    passed = CliRunner().invoke(app, ["check", str(clean_paper), "--offline"])

    monkeypatch.setattr("preflight.cli.pick_track", lambda console, profile: "short")
    failed = CliRunner().invoke(app, ["check", str(clean_paper), "--offline"])

    assert passed.exit_code == 0, passed.output
    assert failed.exit_code == 1
    assert "page limit" in failed.output.lower()
