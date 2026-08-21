"""A Textual UI for reading pre-flight reports.

The report is a tree of findings on the left and the full evidence for the
selected finding on the right, because the evidence is the part an author
actually acts on. Toggles for the expensive checks live on keys rather than in a
menu so a re-run is one keystroke.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static, Tree
from textual.widgets.tree import TreeNode

from ..context import Settings
from ..models import Finding, Progress, Report, Severity
from ..profile import Profile
from ..report import LABELS, STYLES, headline, render_scores, to_markdown
from ..runner import run_checks

_SECTIONS = [
    (Severity.ERROR, "Errors"),
    (Severity.WARNING, "Warnings"),
    (Severity.PASS, "Passed"),
    (Severity.SKIPPED, "Skipped"),
    (Severity.UNVERIFIABLE, "Not checkable"),
]


class PreflightApp(App[None]):
    """Interactive front end for :func:`preflight.runner.run_checks`."""

    CSS = """
    Screen { layers: base overlay; }
    #body { height: 1fr; }
    #sidebar { width: 46; border-right: solid $panel-darken-2; }
    #tree { height: 1fr; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    #detail { padding: 1 2; }
    #summary { height: auto; padding: 0 1; }
    .muted { color: $text-muted; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "rerun", "Re-run"),
        Binding("l", "toggle_llm", "LLM"),
        Binding("s", "toggle_scores", "Scores"),
        Binding("b", "toggle_hallucinator", "Bib check"),
        Binding("n", "next_file", "Next PDF"),
        Binding("e", "export", "Export .md"),
        Binding("v", "toggle_verbose", "Verbose"),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
    ]

    status: reactive[str] = reactive("")

    def __init__(
        self,
        paths: list[Path],
        profile: Profile,
        track: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.profile = profile
        self.track = track
        self.settings = settings or Settings()
        self.index = 0
        self.report: Report | None = None
        self.verbose = False
        self._findings: dict[str, Finding] = {}

    # -- layout -----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("", id="summary")
                yield Tree("Findings", id="tree")
            with VerticalScroll():
                yield Static("", id="detail")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "preflight"
        self.sub_title = self.profile.name
        tree = self.query_one("#tree", Tree)
        tree.show_root = False
        tree.guide_depth = 2
        if not self.paths:
            self._set_detail(Panel(
                Text("No PDF given.\n\nStart the UI with a file:\n  preflight tui paper.pdf",
                     justify="left"),
                title="Nothing loaded", border_style="yellow"))
            return
        self.run_report()

    # -- running ----------------------------------------------------------
    @property
    def current_path(self) -> Path | None:
        return self.paths[self.index] if self.paths else None

    def run_report(self) -> None:
        path = self.current_path
        if path is None:
            return
        self.status = f"Checking {path.name}..."
        self.query_one("#tree", Tree).clear()
        self._set_detail(Text("Running checks...", style="dim"))
        self._run_worker(path)

    @work(thread=True, exclusive=True)
    def _run_worker(self, path: Path) -> None:
        def progress(snapshot: Progress) -> None:
            self.call_from_thread(
                setattr, self, "status", f"[{snapshot.done}/{snapshot.total}] {snapshot.label(2)}"
            )

        try:
            report = run_checks(path, self.profile, self.track, self.settings, progress=progress)
        except Exception as exc:  # surfaced in the UI rather than crashing it
            self.call_from_thread(self._show_error, f"{type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self._show_report, report)

    def _show_error(self, message: str) -> None:
        self.status = "failed"
        self._set_detail(Panel(Text(message, style="red"), title="Check run failed", border_style="red"))

    def _show_report(self, report: Report) -> None:
        self.report = report
        self._findings = {f.check_id: f for f in report.findings}
        self.status = self._toggle_summary()

        counts = report.counts()
        summary = Text()
        summary.append(f"{report.pdf_path.split('/')[-1]}\n", style="bold")
        summary.append(f"{report.conference} · {report.track} · ", style="dim")
        summary.append(f"{counts['error']}E ", style="bold red" if counts["error"] else "dim")
        summary.append(f"{counts['warning']}W ", style="bold yellow" if counts["warning"] else "dim")
        summary.append(f"{counts['pass']}P", style="dim")
        self.query_one("#summary", Static).update(summary)

        tree = self.query_one("#tree", Tree)
        tree.clear()
        first: TreeNode[str] | None = None
        for severity, label in _SECTIONS:
            findings = [f for f in report.sorted_findings() if f.severity is severity]
            if not findings:
                continue
            branch = tree.root.add(
                Text(f"{severity.glyph} {label} ({len(findings)})", style=STYLES[severity]),
                expand=severity in (Severity.ERROR, Severity.WARNING),
            )
            for finding in findings:
                node = branch.add_leaf(Text(finding.title, style=STYLES[severity]), data=finding.check_id)
                if first is None and severity in (Severity.ERROR, Severity.WARNING):
                    first = node
        tree.focus()
        if first is not None:
            tree.select_node(first)
            self._render_detail(self._findings[str(first.data)])
        else:
            self._set_detail(self._overview(report))

    # -- detail pane ------------------------------------------------------
    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        check_id = event.node.data
        if check_id and check_id in self._findings:
            self._render_detail(self._findings[check_id])
        elif self.report is not None:
            self._set_detail(self._overview(self.report))

    def _render_detail(self, finding: Finding) -> None:
        style = STYLES[finding.severity]
        parts: list[object] = []

        head = Text()
        head.append(f"{finding.severity.glyph} {LABELS[finding.severity]}  ", style=style)
        head.append(finding.title, style="bold")
        parts.append(head)
        parts.append(Text(f"{finding.check_id} · {finding.category}\n", style="dim"))
        parts.append(Text(finding.message))

        if finding.evidence:
            parts.append(Text("\nEvidence", style="bold underline"))
            limit = None if self.verbose else 12
            for ev in finding.evidence[:limit]:
                rendered = ev.render()
                if rendered:
                    parts.append(Text(f"  · {rendered}", style="dim"))
            remaining = len(finding.evidence) - (limit or len(finding.evidence))
            if remaining > 0:
                parts.append(Text(f"  · ... {remaining} more — press v for verbose", style="dim italic"))

        if finding.confidence:
            parts.append(Text(f"\nConfidence: {finding.confidence}", style="italic"))
        if finding.remedy:
            parts.append(Text("\nHow to fix", style="bold underline"))
            parts.append(Text(f"  {finding.remedy}"))
        if finding.cfp_reference:
            parts.append(Text("\nWhat the CFP says", style="bold underline"))
            parts.append(Text(f"  {finding.cfp_reference}", style="dim italic"))

        self._set_detail(Panel(Group(*parts), border_style=style.split()[-1], padding=(1, 2)))

    def _overview(self, report: Report) -> Group:
        parts: list[object] = [Panel(headline(report), border_style="green" if not report.errors else "red")]
        if report.scores:
            parts.append(render_scores(report))
        meta = report.meta
        info = Text()
        for key in ("profile_name", "profile_lineage", "modules", "checks_run", "page_count",
                    "page_size_pt", "body_font_size_pt", "dominant_font", "content_page_limit"):
            if key in meta:
                info.append(f"{key.replace('_', ' ')}: ", style="dim")
                info.append(f"{meta[key]}\n")
        parts.append(Panel(info, title="Document", title_align="left", border_style="blue"))
        parts.append(Text("Select a finding on the left to see its evidence.", style="dim italic"))
        return Group(*parts)

    def _set_detail(self, renderable: object) -> None:
        self.query_one("#detail", Static).update(renderable)  # type: ignore[arg-type]

    def watch_status(self, value: str) -> None:
        try:
            self.query_one("#status", Static).update(Text(value, style="dim"))
        except Exception:
            pass

    def _toggle_summary(self) -> str:
        flags = [
            ("LLM", self.settings.enable_llm),
            ("scores", self.settings.enable_scores),
            ("bib check", self.settings.enable_hallucinator),
            ("verbose", self.verbose),
        ]
        on = ", ".join(name for name, enabled in flags if enabled) or "deterministic checks only"
        model = f" · {self.settings.llm_model}" if self.settings.enable_llm else ""
        counter = f" · {self.index + 1}/{len(self.paths)}" if len(self.paths) > 1 else ""
        return f"{on}{model}{counter} · r to re-run"

    # -- actions ----------------------------------------------------------
    def action_rerun(self) -> None:
        self.run_report()

    def action_toggle_llm(self) -> None:
        self.settings = replace(self.settings, enable_llm=not self.settings.enable_llm)
        if not self.settings.enable_llm:
            self.settings = replace(self.settings, enable_scores=False)
        self.status = self._toggle_summary()
        self.notify(f"LLM checks {'on' if self.settings.enable_llm else 'off'} — press r to re-run")

    def action_toggle_scores(self) -> None:
        enable = not self.settings.enable_scores
        self.settings = replace(self.settings, enable_scores=enable,
                                enable_llm=self.settings.enable_llm or enable)
        self.status = self._toggle_summary()
        self.notify(f"Scoring {'on' if enable else 'off'} — press r to re-run")

    def action_toggle_hallucinator(self) -> None:
        self.settings = replace(self.settings, enable_hallucinator=not self.settings.enable_hallucinator)
        self.status = self._toggle_summary()
        self.notify(
            f"Reference validation {'on' if self.settings.enable_hallucinator else 'off'} — press r to re-run"
        )

    def action_toggle_verbose(self) -> None:
        self.verbose = not self.verbose
        tree = self.query_one("#tree", Tree)
        node = tree.cursor_node
        if node is not None and node.data and str(node.data) in self._findings:
            self._render_detail(self._findings[str(node.data)])
        self.status = self._toggle_summary()

    def action_next_file(self) -> None:
        if len(self.paths) > 1:
            self.index = (self.index + 1) % len(self.paths)
            self.run_report()

    def action_export(self) -> None:
        if self.report is None:
            return
        target = Path(self.report.pdf_path).with_suffix(".preflight.md")
        target.write_text(to_markdown(self.report), encoding="utf-8")
        self.notify(f"Wrote {target}")

    def action_cursor_down(self) -> None:
        self.query_one("#tree", Tree).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#tree", Tree).action_cursor_up()
