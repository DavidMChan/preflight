"""The Responsible NLP checklist framework (no model calls)."""

from __future__ import annotations

from pathlib import Path

from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks
from preflight.rnlp import SCOPE_ALWAYS, load_questions, paper_context, question_prompt


def _ctx(path: Path) -> CheckContext:
    profile = load_profile("arr")
    return CheckContext(doc=Document(path), profile=profile, track=profile.track("long"),
                        settings=Settings(enable_llm=True))


def test_questions_load_with_required_fields() -> None:
    questions = load_questions()
    assert questions, "no checklist questions were bundled"
    for q in questions:
        assert q.code and q.title and q.question
        assert q.section in {"A", "B", "C", "D", "E"}
        assert q.check_id.startswith("rnlp_")


def test_question_codes_are_unique() -> None:
    codes = [q.code for q in load_questions()]
    assert len(codes) == len(set(codes))


def test_every_question_registers_a_check() -> None:
    registry = load_builtin_checks()
    for q in load_questions():
        assert q.check_id in registry.checks
        check = registry.checks[q.check_id]
        assert check.requires == ("enable_llm",)     # never runs without opting in
        assert check.module == "arr.responsible_nlp"


def test_prompts_share_an_identical_prefix(clean_paper: Path) -> None:
    """The paper text must be byte-identical across questions, so it can be cached."""
    ctx = _ctx(clean_paper)
    try:
        questions = load_questions()
        context = paper_context(ctx)
        assert len(questions) >= 1
        for q in questions:
            assert question_prompt(ctx, q).startswith(context)
    finally:
        ctx.doc.close()


def test_context_strips_line_numbers_and_is_cached(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        first = paper_context(ctx)
        assert paper_context(ctx) is ctx.shared["llm_paper_context"]
        assert first == paper_context(ctx)
        assert "BEGIN PAPER TEXT" in first
    finally:
        ctx.doc.close()


def test_context_respects_the_length_budget(clean_paper: Path) -> None:
    ctx = _ctx(clean_paper)
    try:
        ctx.profile.raw.setdefault("llm", {})["max_context_chars"] = 500
        text = paper_context(ctx)
        assert len(text) < 1200          # header/footer plus the budget
        assert "omitted for length" in text
    finally:
        ctx.doc.close()


def test_applicability_groups_are_known() -> None:
    valid = {SCOPE_ALWAYS, "artifacts", "experiments", "human_subjects", "ai_assistants"}
    for q in load_questions():
        assert q.applies_when in valid, f"{q.code} has an unknown applies_when"
