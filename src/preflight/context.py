"""Shared state handed to every check."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .document import Document
from .models import Evidence, Finding, Severity
from .profile import Profile, TrackSpec


@dataclass(slots=True)
class Settings:
    """Run-time toggles, mostly driven by CLI flags and the environment."""

    enable_refcheck: bool = True        # the built-in reference checker
    enable_hallucinator: bool = False   # the third-party backend, opt-in
    enable_llm: bool = True
    enable_scores: bool = True
    openai_api_key: str | None = None
    llm_model: str = "gpt-5.6-luna"
    llm_timeout: float = 120.0
    llm_concurrency: int = 8
    hallucinator_max_refs: int = 0
    hallucinator_cache: str | None = None
    strict: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def offline(cls, **overrides: Any) -> Settings:
        """Deterministic checks only: no model calls and no database lookups."""
        return cls(enable_llm=False, enable_scores=False, enable_hallucinator=False,
                   enable_refcheck=False, **overrides)


@dataclass(slots=True)
class CheckContext:
    doc: Document
    profile: Profile
    track: TrackSpec
    settings: Settings
    shared: dict[str, Any] = field(default_factory=dict)

    # -- profile helpers --------------------------------------------------
    def conf(self, path: str, default: Any = None) -> Any:
        return self.profile.get(path, default)

    def cfp(self, key: str) -> str | None:
        return self.profile.cfp_note(key)

    # -- finding constructors --------------------------------------------
    def finding(
        self,
        check_id: str,
        title: str,
        severity: Severity,
        message: str,
        *,
        category: str = "general",
        evidence: list[Evidence] | None = None,
        remedy: str | None = None,
        confidence: str | None = None,
        cfp_key: str | None = None,
    ) -> Finding:
        return Finding(
            check_id=check_id,
            title=title,
            severity=severity,
            message=message,
            category=category,
            evidence=evidence or [],
            remedy=remedy,
            confidence=confidence,
            cfp_reference=self.cfp(cfp_key or check_id),
        )

    def ok(self, check_id: str, title: str, message: str, **kw: Any) -> Finding:
        return self.finding(check_id, title, Severity.PASS, message, **kw)

    def error(self, check_id: str, title: str, message: str, **kw: Any) -> Finding:
        return self.finding(check_id, title, Severity.ERROR, message, **kw)

    def warn(self, check_id: str, title: str, message: str, **kw: Any) -> Finding:
        return self.finding(check_id, title, Severity.WARNING, message, **kw)

    def skip(self, check_id: str, title: str, message: str, **kw: Any) -> Finding:
        return self.finding(check_id, title, Severity.SKIPPED, message, **kw)
