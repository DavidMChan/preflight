"""Core result types shared by every check."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """How much a finding should worry the author.

    ``ERROR``    -- a likely desk-rejection condition, verifiable from the PDF.
    ``WARNING``  -- a heuristic hit that a human must adjudicate.
    ``PASS``     -- the check ran and the paper satisfied it.
    ``SKIPPED``  -- the check could not run (missing dependency, no API key, ...).
    ``UNVERIFIABLE`` -- a real requirement that a PDF checker cannot establish.
    """

    ERROR = "error"
    WARNING = "warning"
    PASS = "pass"
    SKIPPED = "skipped"
    UNVERIFIABLE = "unverifiable"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    @property
    def glyph(self) -> str:
        return _SEVERITY_GLYPH[self]


_SEVERITY_RANK = {
    Severity.ERROR: 0,
    Severity.WARNING: 1,
    Severity.UNVERIFIABLE: 2,
    Severity.SKIPPED: 3,
    Severity.PASS: 4,
}

_SEVERITY_GLYPH = {
    Severity.ERROR: "✗",       # ✗
    Severity.WARNING: "⚠",     # ⚠
    Severity.PASS: "✓",        # ✓
    Severity.SKIPPED: "○",     # ○
    Severity.UNVERIFIABLE: "–",  # –
}


@dataclass(slots=True)
class Evidence:
    """A precise, author-actionable pointer back into the PDF.

    The whole point of preferring a deterministic layout analyzer over one big
    LLM call is that we can say *page 5, x=34.0pt, expected >= 69.0pt* instead
    of "formatting may be wrong".
    """

    page: int | None = None
    detail: str = ""
    quote: str | None = None
    measured: float | None = None
    expected: str | None = None
    bbox: tuple[float, float, float, float] | None = None

    def render(self) -> str:
        bits: list[str] = []
        if self.page is not None:
            bits.append(f"page {self.page}")
        if self.detail:
            bits.append(self.detail)
        if self.measured is not None:
            m = f"measured {self.measured:.1f}"
            if self.expected:
                m += f", expected {self.expected}"
            bits.append(m)
        elif self.expected:
            bits.append(f"expected {self.expected}")
        if self.quote:
            q = " ".join(self.quote.split())
            if len(q) > 140:
                q = q[:137] + "..."
            bits.append(f'"{q}"')
        return " — ".join(bits) if bits else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "detail": self.detail or None,
            "quote": self.quote,
            "measured": self.measured,
            "expected": self.expected,
            "bbox": list(self.bbox) if self.bbox else None,
        }


@dataclass(slots=True)
class Progress:
    """A snapshot of a run in flight.

    With the network checks overlapping, "the last check that finished" is not
    what a waiting user wants to see. This reports what is still running, and
    lets a long check publish its own sub-status.
    """

    done: int
    total: int
    running: tuple[str, ...] = ()
    details: tuple[str, ...] = ()

    @property
    def finished(self) -> bool:
        return self.done >= self.total and not self.running

    def label(self, width: int = 3) -> str:
        """A one-line summary: what is still in flight, not what just finished."""
        if self.finished:
            return "done"
        if self.details:
            head = " · ".join(self.details[:2])
        elif self.running:
            head = ", ".join(self.running[:width])
            if len(self.running) > width:
                head += f" +{len(self.running) - width} more"
        else:
            head = "starting"
        return head


@dataclass(slots=True)
class Finding:
    """The outcome of a single check."""

    check_id: str
    title: str
    severity: Severity
    message: str
    category: str = "general"
    evidence: list[Evidence] = field(default_factory=list)
    remedy: str | None = None
    confidence: str | None = None
    cfp_reference: str | None = None

    @property
    def is_blocking(self) -> bool:
        return self.severity is Severity.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "title": self.title,
            "category": self.category,
            "severity": self.severity.value,
            "message": self.message,
            "remedy": self.remedy,
            "confidence": self.confidence,
            "cfp_reference": self.cfp_reference,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass(slots=True)
class Report:
    """Everything one run of the tool produced for one PDF."""

    pdf_path: str
    conference: str
    track: str
    findings: list[Finding] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def add(self, finding: Finding | Iterable[Finding] | None) -> None:
        if finding is None:
            return
        if isinstance(finding, Finding):
            self.findings.append(finding)
        else:
            self.findings.extend(finding)

    def of(self, *severities: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity in severities]

    @property
    def errors(self) -> list[Finding]:
        return self.of(Severity.ERROR)

    @property
    def warnings(self) -> list[Finding]:
        return self.of(Severity.WARNING)

    @property
    def passed(self) -> bool:
        return not self.errors

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (f.severity.rank, f.category, f.check_id))

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in Severity}
        for f in self.findings:
            out[f.severity.value] += 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf": self.pdf_path,
            "conference": self.conference,
            "track": self.track,
            "counts": self.counts(),
            "passed": self.passed,
            "meta": self.meta,
            "scores": self.scores,
            "findings": [f.to_dict() for f in self.sorted_findings()],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
