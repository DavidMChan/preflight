"""The model-backed audit framework (no model calls)."""

from __future__ import annotations

from pathlib import Path

import pytest

from preflight.analysis import paper_context
from preflight.audit import (
    AUDIT_SCHEMA,
    CONTEXT_PROVIDERS,
    audit_prompt,
    load_audits,
    normalise_quote,
)
from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks


def _ctx(path: Path) -> CheckContext:
    profile = load_profile("arr")
    return CheckContext(doc=Document(path), profile=profile, track=profile.track("long"),
                        settings=Settings(enable_llm=True))


def test_audits_load_with_required_fields() -> None:
    audits = load_audits()
    assert audits, "no audits were bundled"
    for audit in audits:
        assert audit.id and audit.title and audit.prompt
        assert audit.context in CONTEXT_PROVIDERS
        assert audit.check_id.startswith("audit_")


def test_audit_ids_are_unique() -> None:
    ids = [a.id for a in load_audits()]
    assert len(ids) == len(set(ids))


def test_every_audit_registers_an_advisory_check() -> None:
    registry = load_builtin_checks()
    for audit in load_audits():
        check = registry.checks[audit.check_id]
        assert check.requires == ("enable_llm",)   # never runs without opting in
        assert check.is_async                       # so audits overlap
        assert not check.aggregate


@pytest.mark.parametrize("provider", sorted(CONTEXT_PROVIDERS))
def test_context_providers_return_text(clean_paper: Path, provider: str) -> None:
    """Each provider must work on a paper that has no figures, maths or long sentences."""
    ctx = _ctx(clean_paper)
    try:
        result = CONTEXT_PROVIDERS[provider](ctx)
        assert isinstance(result, str)
    finally:
        ctx.doc.close()


def test_prompts_share_the_paper_prefix(clean_paper: Path) -> None:
    """Every audit sends the same paper text, so the provider can cache the prefix."""
    ctx = _ctx(clean_paper)
    try:
        context = paper_context(ctx)
        for audit in load_audits():
            prompt = audit_prompt(ctx, audit)
            assert prompt.startswith(context)
            assert audit.title in prompt
            # The varying part comes last and is short relative to the paper.
            assert len(prompt) - len(context) < 20_000
    finally:
        ctx.doc.close()


def test_audit_prompt_includes_its_structured_context(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        captioned = next((a for a in load_audits() if a.context == "captions"), None)
        if captioned is None:
            pytest.skip("no caption-based audit bundled")
        assert "CAPTIONS" in audit_prompt(ctx, captioned)
    finally:
        ctx.doc.close()


def test_schema_is_the_shared_shape() -> None:
    for key in ("summary", "verdict", "confidence", "issues", "severity", "quote", "fix"):
        assert key in AUDIT_SCHEMA


def test_normalise_quote_collapses_and_truncates() -> None:
    assert normalise_quote("  a   b\nc  ") == "a b c"
    assert len(normalise_quote("x" * 500)) == 200


def test_scoring_reads_the_whole_paper(clean_paper: Path) -> None:
    """Scores must be formed over the full text, not a first-two-pages excerpt.

    The excerpt version produced exactly the hedge you would expect — "cannot be
    assessed from the excerpt" — and the context budget is now large enough that
    there is no reason to send anything less.
    """
    from preflight.llm.prompts import SCORE_PROMPT

    ctx = _ctx(clean_paper)
    try:
        body = paper_context(ctx)
        prompt = SCORE_PROMPT.format(body=body)
        assert body in prompt
        assert "BEGIN PAPER TEXT" in prompt
        # The prompt must not tell the model it is working from a fragment.
        assert "excerpt" not in prompt.split("--- PAPER ---")[0].lower()
    finally:
        ctx.doc.close()
