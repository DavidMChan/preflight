"""Interactive conference picker.

There is no default venue: the rules that matter (page limits, anonymity,
checklists) differ enough between conferences that guessing one is worse than
asking. When ``--conference`` is omitted and the terminal is interactive, we
ask here; otherwise the caller reports a usage error. A venue with more than one
track is a second question, since picking ARR without picking between long,
short and demo would silently apply the wrong page limit.

The picker is arrow-key driven where the terminal supports raw input, and falls
back to a numbered prompt when it does not (pipes, dumb terminals, Windows).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

from .profile import Profile, ProfileError, available_profiles, load_profile

LAST_CHOICE = Path(os.environ.get("PREFLIGHT_CACHE", "~/.cache/preflight")).expanduser() / "last-conference"


class PickerUnavailable(RuntimeError):
    """No interactive terminal, so the caller has to ask for a conference explicitly."""


@dataclass(slots=True)
class Choice:
    key: str
    name: str
    detail: str = ""


def _choices() -> list[Choice]:
    out: list[Choice] = []
    for key in available_profiles():
        try:
            profile = load_profile(key)
        except ProfileError:  # pragma: no cover - a broken bundled profile
            continue
        tracks = ", ".join(f"{n} ({s.content_page_limit}p)" for n, s in profile.tracks.items())
        out.append(Choice(key, profile.name, tracks))
    return out


def _track_choices(profile: Profile) -> list[Choice]:
    # The bundled descriptions run to a paragraph, which is too much for a row;
    # the page limit is the fact that decides the choice.
    return [
        Choice(name, f"{spec.content_page_limit} content pages")
        for name, spec in profile.tracks.items()
    ]


def remembered() -> str | None:
    """The venue picked last time, if it is still a profile we know about."""
    try:
        key = LAST_CHOICE.read_text("utf-8").strip()
    except OSError:
        return None
    return key or None


def remember(key: str) -> None:
    """Record the choice so the next picker starts on it. Never raises."""
    if key not in available_profiles():  # a one-off path is not worth remembering
        return
    try:
        LAST_CHOICE.parent.mkdir(parents=True, exist_ok=True)
        LAST_CHOICE.write_text(key, encoding="utf-8")
    except OSError:  # pragma: no cover - a read-only cache dir is not a failure
        pass


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def pick_conference(console: Console) -> str:
    """Ask which venue to check against. Returns a profile key.

    Raises ``PickerUnavailable`` when there is no terminal to ask in, and
    ``KeyboardInterrupt`` when the user backs out.
    """
    if not interactive():
        raise PickerUnavailable("not a terminal")

    choices = _choices()
    if not choices:
        raise PickerUnavailable("no bundled profiles")
    if len(choices) == 1:
        return choices[0].key

    last = remembered()
    start = next((i for i, c in enumerate(choices) if c.key == last), 0)

    try:
        key = _arrow_picker(console, choices, start, last, "Which conference?", "last used")
    except _RawModeUnavailable:
        key = _numbered_picker(console, choices, start, last, "Which conference?", "last used")
    remember(key)
    return key


def pick_track(console: Console, profile: Profile) -> str | None:
    """Ask which track within a venue. ``None`` means "use the profile default".

    Venues with a single track have nothing to ask about, and a terminal that
    cannot answer keeps the default rather than failing the run.
    """
    choices = _track_choices(profile)
    if len(choices) < 2 or not interactive():
        return None

    default = profile.default_track
    start = next((i for i, c in enumerate(choices) if c.key == default), 0)
    title = f"Which {profile.name} track?"
    try:
        return _arrow_picker(console, choices, start, default, title, "default")
    except _RawModeUnavailable:
        return _numbered_picker(console, choices, start, default, title, "default")


# -- arrow-key picker -----------------------------------------------------


class _RawModeUnavailable(RuntimeError):
    pass


def _render(choices: list[Choice], index: int, marked: str | None, title: str, marker: str) -> Panel:
    lines: list[Text] = []
    for i, choice in enumerate(choices):
        selected = i == index
        line = Text()
        line.append("  ❯ " if selected else "    ", style="bold cyan" if selected else "")
        line.append(f"{choice.key:<10}", style="bold cyan" if selected else "bold")
        line.append(choice.name, style="" if selected else "dim")
        if choice.key == marked:
            line.append(f"  ({marker})", style="dim italic")
        lines.append(line)
        if choice.detail:
            lines.append(Text(f"      {choice.detail}", style="dim"))
    lines.append(Text())
    lines.append(Text("  ↑/↓ or j/k to move · enter to choose · q to cancel", style="dim"))
    return Panel(
        Group(*lines),
        title=title,
        title_align="left",
        subtitle="[dim]pass -c next time to skip this[/dim]",
        subtitle_align="right",
        border_style="cyan",
    )


def _read_key() -> str:
    """One keypress from the terminal, with escape sequences collapsed to names."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":  # CSI: read the rest of the arrow sequence
            rest = sys.stdin.read(2)
            return {"[A": "up", "[B": "down"}.get(rest, "esc")
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _arrow_picker(console: Console, choices: list[Choice], index: int, marked: str | None,
                  title: str, marker: str) -> str:
    try:
        import termios  # noqa: F401
        import tty  # noqa: F401
    except ImportError as exc:  # pragma: no cover - Windows
        raise _RawModeUnavailable(str(exc)) from exc
    try:
        termios_ok = sys.stdin.fileno() >= 0
    except (OSError, ValueError) as exc:  # pragma: no cover - detached stdin
        raise _RawModeUnavailable(str(exc)) from exc
    if not termios_ok:  # pragma: no cover
        raise _RawModeUnavailable("no stdin")

    from rich.live import Live

    with Live(_render(choices, index, marked, title, marker), console=console, auto_refresh=False,
              transient=True, screen=False) as live:
        while True:
            try:
                key = _read_key()
            except Exception as exc:  # pragma: no cover - terminal without raw mode
                raise _RawModeUnavailable(str(exc)) from exc
            if key in ("up", "k"):
                index = (index - 1) % len(choices)
            elif key in ("down", "j"):
                index = (index + 1) % len(choices)
            elif key in ("\r", "\n"):
                return choices[index].key
            elif key in ("q", "\x03", "\x04", "esc"):
                raise KeyboardInterrupt
            elif key.isdigit() and 1 <= int(key) <= len(choices):
                return choices[int(key) - 1].key
            else:
                continue
            live.update(_render(choices, index, marked, title, marker), refresh=True)


# -- fallback -------------------------------------------------------------


def _numbered_picker(console: Console, choices: list[Choice], index: int, marked: str | None,
                     title: str = "Which conference?", marker: str = "last used") -> str:
    console.print(f"[bold]{title}[/bold]")
    for i, choice in enumerate(choices, start=1):
        suffix = f" [dim]({marker})[/dim]" if choice.key == marked else ""
        console.print(f"  [cyan]{i}[/cyan]. [bold]{choice.key}[/bold] — {choice.name}{suffix}")
    default = choices[index].key
    while True:
        answer = console.input(f"[dim]number or key [{default}]: [/dim]").strip()
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(choices):
            return choices[int(answer) - 1].key
        for choice in choices:
            if choice.key == answer.lower():
                return choice.key
        console.print(f"[yellow]Not one of the options: {answer!r}[/yellow]")
