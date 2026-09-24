"""Persist finished reports so they can be re-read without re-running anything.

A full run makes dozens of model calls and can spend minutes on bibliographic
lookups. Asking for more detail about one finding should not cost that again, so
every run is written to disk and ``preflight show`` reads it back.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Evidence, Finding, Report, Severity

DEFAULT_ROOT = Path(os.environ.get("PREFLIGHT_CACHE", "~/.cache/preflight")).expanduser()


def runs_dir(root: Path | None = None) -> Path:
    return (root or DEFAULT_ROOT) / "runs"


@dataclass(slots=True)
class CachedRun:
    run_id: str
    path: Path
    pdf: str
    conference: str
    track: str
    saved_at: float
    counts: dict[str, int]

    @property
    def when(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(self.saved_at))

    @property
    def name(self) -> str:
        return Path(self.pdf).name


def run_id(report: Report) -> str:
    """Stable-ish id: the paper, the venue, and when it was run."""
    seed = f"{report.pdf_path}|{report.conference}|{report.track}|{time.time()}"
    return hashlib.sha256(seed.encode()).hexdigest()[:8]


def save(report: Report, root: Path | None = None) -> Path | None:
    """Write a report to the cache. Never raises: a cache miss is not a failure."""
    try:
        directory = runs_dir(root)
        directory.mkdir(parents=True, exist_ok=True)
        rid = run_id(report)
        payload = report.to_dict()
        payload["run_id"] = rid
        payload["saved_at"] = time.time()
        target = directory / f"{rid}.json"
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (directory.parent / "last-run").write_text(str(target), encoding="utf-8")
        return target
    except OSError:
        return None


def _load_payload(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def list_runs(root: Path | None = None, limit: int = 20) -> list[CachedRun]:
    directory = runs_dir(root)
    if not directory.is_dir():
        return []
    out: list[CachedRun] = []
    for path in directory.glob("*.json"):
        data = _load_payload(path)
        if not data:
            continue
        out.append(
            CachedRun(
                run_id=str(data.get("run_id", path.stem)),
                path=path,
                pdf=str(data.get("pdf", "")),
                conference=str(data.get("conference", "")),
                track=str(data.get("track", "")),
                saved_at=float(data.get("saved_at", 0.0)),
                counts=dict(data.get("counts") or {}),
            )
        )
    out.sort(key=lambda r: r.saved_at, reverse=True)
    return out[:limit]


def resolve(reference: str | None, root: Path | None = None) -> Path | None:
    """Find a cached run by id, by PDF name, or the most recent one."""
    directory = runs_dir(root)
    if reference:
        exact = directory / f"{reference}.json"
        if exact.is_file():
            return exact
        for run in list_runs(root, limit=200):
            if reference in (run.run_id, run.name, run.pdf):
                return run.path
        return None

    pointer = directory.parent / "last-run"
    if pointer.is_file():
        candidate = Path(pointer.read_text(encoding="utf-8").strip())
        if candidate.is_file():
            return candidate
    runs = list_runs(root, limit=1)
    return runs[0].path if runs else None


def load(path: Path) -> Report | None:
    """Rebuild a :class:`Report` from cached JSON."""
    data = _load_payload(path)
    if not data:
        return None
    report = Report(
        pdf_path=str(data.get("pdf", "")),
        conference=str(data.get("conference", "")),
        track=str(data.get("track", "")),
        scores=dict(data.get("scores") or {}),
        meta=dict(data.get("meta") or {}),
    )
    for raw in data.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        try:
            severity = Severity(str(raw.get("severity", "skipped")))
        except ValueError:
            severity = Severity.SKIPPED
        report.findings.append(
            Finding(
                check_id=str(raw.get("check_id", "")),
                title=str(raw.get("title", "")),
                severity=severity,
                message=str(raw.get("message", "")),
                category=str(raw.get("category", "general")),
                remedy=raw.get("remedy"),
                confidence=raw.get("confidence"),
                cfp_reference=raw.get("cfp_reference"),
                uses=tuple(raw["uses"]) if isinstance(raw.get("uses"), list) else None,
                evidence=[
                    Evidence(
                        page=ev.get("page"),
                        detail=str(ev.get("detail") or ""),
                        quote=ev.get("quote"),
                        measured=ev.get("measured"),
                        expected=ev.get("expected"),
                        bbox=tuple(ev["bbox"]) if ev.get("bbox") else None,
                    )
                    for ev in (raw.get("evidence") or [])
                    if isinstance(ev, dict)
                ],
            )
        )
    report.meta["run_id"] = data.get("run_id")
    report.meta["saved_at"] = data.get("saved_at")
    return report


def prune(keep: int = 50, root: Path | None = None) -> int:
    """Drop the oldest cached runs; returns how many were removed."""
    runs = list_runs(root, limit=10_000)
    removed = 0
    for run in runs[keep:]:
        try:
            run.path.unlink()
            removed += 1
        except OSError:
            pass
    return removed
