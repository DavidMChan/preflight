"""Command line interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import cache
from .context import Settings
from .models import Progress, Severity
from .picker import PickerUnavailable, pick_conference, pick_track, remember
from .profile import Profile, ProfileError, available_profiles, load_profile
from .registry import describe, load_builtin_checks
from .report import print_report, render_finding, to_markdown
from .runner import run_checks

app = typer.Typer(
    name="preflight",
    help="Pre-flight checks for academic paper submissions, before a desk rejection finds them.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)

DEFAULT_MODEL = "gpt-5.6-luna"


def _load_env() -> None:
    """Read .env from the current directory, then the project root."""
    load_dotenv()
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break


def _build_settings(
    llm: bool, scores: bool, hallucinator: bool, model: str, openai_key: str | None,
    max_refs: int, strict: bool, offline: bool = False, concurrency: int = 8,
    refcheck: bool = True,
) -> Settings:
    _load_env()
    if offline:
        llm = scores = hallucinator = refcheck = False
    # Scoring is itself a model call, so --no-llm turns it off too rather than
    # quietly re-enabling the model behind the flag that disabled it.
    return Settings(
        enable_llm=llm,
        enable_scores=scores and llm,
        enable_refcheck=refcheck,
        enable_hallucinator=hallucinator,
        llm_concurrency=concurrency,
        openai_api_key=openai_key or os.environ.get("OPENAI_API_KEY"),
        llm_model=model or os.environ.get("PREFLIGHT_LLM_MODEL") or DEFAULT_MODEL,
        hallucinator_max_refs=max_refs,
        strict=strict,
    )


def _resolve_conference(conference: str | None) -> tuple[str, bool]:
    """A venue is required; ask for one when the terminal can answer.

    Returns the profile key and whether it came from the picker. A venue chosen
    interactively is followed by a track question: picking ARR and silently
    getting the long-paper limit is how a short paper gets measured against the
    wrong rule. An explicit ``-c`` keeps the profile default, so scripts that
    pass a venue and no track behave as before.
    """
    if conference:
        remember(conference)
        return conference, False
    try:
        return pick_conference(console), True
    except PickerUnavailable:
        err_console.print(
            "[red]No conference given.[/red] Pass [bold]-c/--conference[/bold] "
            "(e.g. [bold]-c arr[/bold]); `preflight conferences` lists the bundled profiles."
        )
        raise typer.Exit(2) from None
    except (KeyboardInterrupt, EOFError):
        err_console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(130) from None


def _resolve_track(profile: Profile) -> str | None:
    """Ask which track, once a venue has been chosen interactively."""
    try:
        return pick_track(console, profile)
    except (KeyboardInterrupt, EOFError):
        err_console.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(130) from None


@app.command()
def check(
    pdf: Annotated[list[Path], typer.Argument(help="One or more PDFs to check.")],
    conference: Annotated[str | None, typer.Option("--conference", "-c", help="Bundled profile key, or a path to a YAML profile. Omit to pick one interactively.")] = None,
    track: Annotated[str | None, typer.Option("--track", "-t", help="Submission track (e.g. long, short).")] = None,
    llm: Annotated[bool, typer.Option("--llm/--no-llm", help="LLM-backed semantic checks and the Responsible NLP checklist.")] = True,
    scores: Annotated[bool, typer.Option("--scores/--no-scores", help="Reviewer-style scores (needs --llm).")] = True,
    refcheck: Annotated[bool, typer.Option("--refcheck/--no-refcheck", help="Verify every reference against databases, its own host, and the live web.")] = True,
    hallucinator: Annotated[bool, typer.Option("--hallucinator/--no-hallucinator", help="Also run the third-party hallucinator backend (slower; superseded by --refcheck).")] = False,
    offline: Annotated[bool, typer.Option("--offline", help="Deterministic checks only: no model calls, no database lookups.")] = False,
    model: Annotated[str, typer.Option("--model", "-m", help="Model for the LLM-backed checks.")] = DEFAULT_MODEL,
    openai_key: Annotated[str | None, typer.Option("--openai-key", envvar="OPENAI_API_KEY", help="Overrides OPENAI_API_KEY.")] = None,
    max_refs: Annotated[int, typer.Option("--max-refs", help="Cap references sent to hallucinator (0 = all).")] = 0,
    json_out: Annotated[Path | None, typer.Option("--json", help="Write the full report as JSON.")] = None,
    markdown_out: Annotated[Path | None, typer.Option("--markdown", help="Write a Markdown summary.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show all evidence and CFP references.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only show errors and warnings.")] = False,
    strict: Annotated[bool, typer.Option("--strict", help="Exit non-zero on warnings as well as errors.")] = False,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Maximum simultaneous model calls.")] = 8,
) -> None:
    """Run the checks against one or more PDFs and print a report."""
    settings = _build_settings(llm, scores, hallucinator, model, openai_key, max_refs,
                               strict, offline, concurrency, refcheck)

    if settings.enable_llm and not settings.openai_api_key:
        err_console.print(
            "[yellow]No OPENAI_API_KEY found — the model-backed checks will be skipped. "
            "Set it in .env, pass --openai-key, or run with --offline to silence this.[/yellow]"
        )

    try:
        key, from_picker = _resolve_conference(conference)
        profile = load_profile(key)
        if track is None and from_picker:
            track = _resolve_track(profile)
    except ProfileError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    worst = 0
    for path in pdf:
        if not path.is_file():
            err_console.print(f"[red]No such file: {path}[/red]")
            worst = max(worst, 2)
            continue
        try:
            with console.status(f"Checking {path.name}...", spinner="dots") as status:
                def progress(snapshot: Progress) -> None:
                    status.update(f"[{snapshot.done}/{snapshot.total}] {snapshot.label()}")

                report = run_checks(path, profile, track, settings, progress=progress)
        except ProfileError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc
        except Exception as exc:
            err_console.print(f"[red]Failed to check {path}: {type(exc).__name__}: {exc}[/red]")
            worst = max(worst, 2)
            continue

        print_report(report, console, verbose=verbose, show_passed=not quiet)

        saved = cache.save(report)
        if saved is not None and not verbose:
            console.print(
                f"[dim]Full evidence kept as run {saved.stem} — "
                f"`preflight show` for everything, `preflight show <check_id>` for one finding.[/dim]"
            )

        if json_out is not None:
            target = _numbered(json_out, path, len(pdf))
            target.write_text(report.to_json(), encoding="utf-8")
            console.print(f"[dim]JSON written to {target}[/dim]")
        if markdown_out is not None:
            target = _numbered(markdown_out, path, len(pdf))
            target.write_text(to_markdown(report), encoding="utf-8")
            console.print(f"[dim]Markdown written to {target}[/dim]")

        if report.errors:
            worst = max(worst, 1)
        elif strict and report.of(Severity.WARNING):
            worst = max(worst, 2)

    raise typer.Exit(worst)


def _numbered(target: Path, pdf: Path, count: int) -> Path:
    """Keep multi-PDF runs from overwriting a single output file."""
    if count <= 1:
        return target
    return target.with_name(f"{target.stem}-{pdf.stem}{target.suffix}")


@app.command()
def tui(
    pdf: Annotated[list[Path] | None, typer.Argument(help="PDFs to load at startup.")] = None,
    conference: Annotated[str | None, typer.Option("--conference", "-c", help="Bundled profile key, or a path to a YAML profile. Omit to pick one interactively.")] = None,
    track: Annotated[str | None, typer.Option("--track", "-t")] = None,
    llm: Annotated[bool, typer.Option("--llm/--no-llm")] = True,
    scores: Annotated[bool, typer.Option("--scores/--no-scores")] = True,
    refcheck: Annotated[bool, typer.Option("--refcheck/--no-refcheck")] = True,
    hallucinator: Annotated[bool, typer.Option("--hallucinator/--no-hallucinator")] = False,
    offline: Annotated[bool, typer.Option("--offline")] = False,
    model: Annotated[str, typer.Option("--model", "-m")] = DEFAULT_MODEL,
) -> None:
    """Open the interactive terminal UI."""
    from .tui.app import PreflightApp

    settings = _build_settings(llm, scores, hallucinator, model, None, 0, False, offline,
                               refcheck=refcheck)
    try:
        key, from_picker = _resolve_conference(conference)
        profile = load_profile(key)
        if track is None and from_picker:
            track = _resolve_track(profile)
    except ProfileError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    PreflightApp(
        paths=[p for p in (pdf or []) if p.is_file()],
        profile=profile,
        track=track,
        settings=settings,
    ).run()


@app.command("show")
def show(
    target: Annotated[str | None, typer.Argument(help="A check id, or a run id / PDF name. Omit for the last run.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Which cached run to read (id or PDF name).")] = None,
    list_runs: Annotated[bool, typer.Option("--list", help="List cached runs instead of showing one.")] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Only errors and warnings.")] = False,
) -> None:
    """Re-read a finished run in full detail, without re-running anything."""
    if list_runs:
        runs = cache.list_runs()
        if not runs:
            console.print("[yellow]No cached runs yet.[/yellow]")
            raise typer.Exit(0)
        table = Table(title="Cached runs", title_justify="left")
        table.add_column("Run", style="bold")
        table.add_column("When")
        table.add_column("Paper")
        table.add_column("Venue")
        table.add_column("Result")
        for entry in runs:
            counts = entry.counts
            table.add_row(entry.run_id, entry.when, entry.name, f"{entry.conference}/{entry.track}",
                          f"{counts.get('error', 0)}E {counts.get('warning', 0)}W {counts.get('pass', 0)}P")
        console.print(table)
        raise typer.Exit(0)

    # `target` may name a run, or a check inside one. Try it as a run first; if
    # that fails, fall back to the most recent run and read it as a check id.
    path = cache.resolve(run) if run else None
    named_a_run = False
    if path is None and target:
        path = cache.resolve(target)
        named_a_run = path is not None
    if path is None:
        path = cache.resolve(None)
    if path is None:
        err_console.print("[yellow]No cached run found. Run `preflight <pdf>` first.[/yellow]")
        raise typer.Exit(2)

    report = cache.load(path)
    if report is None:
        err_console.print(f"[red]Could not read the cached run at {path}.[/red]")
        raise typer.Exit(2)

    if target and not named_a_run:
        wanted = [f for f in report.findings if f.check_id == target]
        if not wanted:
            # Offer near-misses rather than silently printing the whole report.
            close = sorted(f.check_id for f in report.findings if target.lower() in f.check_id.lower())
            hint = ", ".join(close or sorted(f.check_id for f in report.findings)[:8])
            err_console.print(
                f"[yellow]No check called {target!r} in run {report.meta.get('run_id')}. "
                f"Did you mean: {hint}[/yellow]"
            )
            raise typer.Exit(2)
        console.print()
        for finding in wanted:
            console.print(render_finding(finding, verbose=True))
        raise typer.Exit(0)

    print_report(report, console, verbose=True, show_passed=not quiet)


@app.command("conferences")
def list_conferences() -> None:
    """List the bundled conference profiles."""
    table = Table(title="Bundled profiles", title_justify="left")
    table.add_column("Key", style="bold")
    table.add_column("Name")
    table.add_column("Inherits")
    table.add_column("Tracks (content page limit)")
    for key in available_profiles():
        profile = load_profile(key)
        tracks = ", ".join(f"{name} ({spec.content_page_limit}p)" for name, spec in profile.tracks.items())
        table.add_row(key, profile.name, " <- ".join(profile.lineage[1:]) or "—", tracks)
    console.print(table)
    console.print("[dim]Use any of these with -c, or pass a path to your own YAML profile.[/dim]")


@app.command("checks")
def list_checks(
    conference: Annotated[str | None, typer.Option("--conference", "-c", help="Only show checks this profile runs.")] = None,
) -> None:
    """List the available checks, grouped by module."""
    load_builtin_checks()
    profile = load_profile(conference) if conference else None

    table = Table(title=f"Checks{f' for {profile.name}' if profile else ''}", title_justify="left")
    table.add_column("Module", style="cyan")
    table.add_column("Check", style="bold")
    table.add_column("Needs")
    table.add_column("What it does", overflow="fold")

    for entry in describe():
        if profile is not None:
            if not profile.module_enabled(entry["module"]) or not profile.check_enabled(entry["id"]):
                continue
        needs = ", ".join(r.replace("enable_", "--") for r in entry["requires"]) or "—"
        table.add_row(entry["module"], entry["id"], needs, entry["description"])
    console.print(table)


def main() -> None:
    """Entry point. A bare PDF path is treated as ``preflight check <path>``."""
    argv = sys.argv[1:]
    commands = {"check", "tui", "show", "conferences", "checks", "--help", "-h", "--version"}
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        sys.argv.insert(1, "check")
    app()


if __name__ == "__main__":
    main()
