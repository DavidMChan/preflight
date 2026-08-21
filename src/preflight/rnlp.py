"""The ARR Responsible NLP Research checklist, as data.

Every checklist question asks the same shape of thing — "does the paper contain
this information?" — so the questions are YAML files under
``preflight/conferences/rnlp/`` rather than eighteen near-identical Python
functions. Adding or amending a question means adding or editing one file.

Two things matter for how this runs:

* **The checklist itself is a submission-form artifact.** Since February 2024 it
  is filled in on OpenReview, not attached to the PDF, so nothing here inspects
  checklist *answers*. What these checks look at is whether the paper contains
  the material each question asks you to point at. Every finding is a warning:
  the CFP is explicit that answering "no" with a justification is acceptable and
  is not grounds for rejection.
* **Prompt shape is chosen for cache hits.** Each question sends the same system
  prompt and the same paper excerpt, with only a short question-specific block
  appended at the end. That keeps a long identical prefix across all eighteen
  calls so the provider can serve most of it from cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import Any

import yaml

from .analysis import paper_context
from .context import CheckContext

# Applicability groups. The CFP gates sections B, C and D on what the paper did.
SCOPE_ALWAYS = "always"
SCOPE_ARTIFACTS = "artifacts"
SCOPE_EXPERIMENTS = "experiments"
SCOPE_HUMAN_SUBJECTS = "human_subjects"
SCOPE_AI_ASSISTANTS = "ai_assistants"

SCOPE_QUESTIONS = {
    SCOPE_ARTIFACTS: "Did the paper use or create scientific artifacts (code, data, models, or other artifacts)?",
    SCOPE_EXPERIMENTS: "Did the paper run computational experiments?",
    SCOPE_HUMAN_SUBJECTS: "Did the paper use human annotators (e.g. crowdworkers) or conduct research with human participants?",
    SCOPE_AI_ASSISTANTS: "Does the paper indicate that AI assistants (e.g. ChatGPT, Copilot) were used in the research, coding, or writing?",
}


@dataclass(slots=True)
class Question:
    """One checklist question, loaded from a YAML file."""

    id: str
    code: str
    section: str
    title: str
    question: str
    applies_when: str = SCOPE_ALWAYS
    guidance: str = ""
    prompt: str = ""
    evidence_hint: str = ""
    order: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def check_id(self) -> str:
        return f"rnlp_{self.id.lower()}"

    @property
    def display(self) -> str:
        return f"{self.code}. {self.title}"


def _coerce(data: dict[str, Any], source: str) -> Question:
    missing = [k for k in ("id", "code", "section", "title", "question") if not data.get(k)]
    if missing:
        raise ValueError(f"{source}: checklist question is missing {', '.join(missing)}")
    return Question(
        id=str(data["id"]),
        code=str(data["code"]),
        section=str(data["section"]).upper(),
        title=str(data["title"]),
        question=str(data["question"]),
        applies_when=str(data.get("applies_when", SCOPE_ALWAYS)),
        guidance=str(data.get("guidance", "")).strip(),
        prompt=str(data.get("prompt", "")).strip(),
        evidence_hint=str(data.get("evidence_hint", "")).strip(),
        order=int(data.get("order", 0)),
        raw=data,
    )


def load_questions() -> list[Question]:
    """Every bundled checklist question, in checklist order."""
    out: list[Question] = []
    root = resources.files("preflight.conferences").joinpath("rnlp")
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if not entry.name.endswith((".yaml", ".yml")):
            continue
        data = yaml.safe_load(entry.read_text("utf-8")) or {}
        out.append(_coerce(data, entry.name))
    return sorted(out, key=lambda q: (q.section, q.order, q.code))


# ---------------------------------------------------------------------------
# Shared prompt scaffolding
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are checking an NLP paper against the ACL Rolling Review Responsible NLP Research "
    "checklist, on behalf of the paper's own authors before they submit.\n\n"
    "You are asked, for one checklist item at a time, whether the paper contains the information "
    "that item asks about. You are NOT reviewing the paper's quality and NOT judging whether the "
    "research was done well.\n\n"
    "Rules you must follow:\n"
    "- Decide only from the paper text provided. Never invent quotations, section numbers, or facts.\n"
    "- Quote verbatim. Every quote you give must appear in the text exactly as you write it.\n"
    "- Partial information counts as 'partial', not 'yes' and not 'no'.\n"
    "- If an item is genuinely not applicable to this paper, say so rather than reporting it missing.\n"
    "- Authors are allowed to answer 'no' with a justification; your job is to tell them what a "
    "reviewer will and will not find, not to scold them.\n"
    "- Be conservative: if you are unsure whether something is present, lower your confidence."
)

ANSWER_SCHEMA = """{
  "status": "yes"|"partial"|"no"|"not_applicable",
  "confidence": "low"|"medium"|"high",
  "where": "section name or number where the information appears, or null",
  "quotes": ["short verbatim quote from the paper", "..."],
  "explanation": "one or two sentences",
  "missing": ["specific piece of information that is absent", "..."]
}"""

SCOPE_SCHEMA = """{
  "artifacts": {"answer": true|false, "why": "one sentence"},
  "experiments": {"answer": true|false, "why": "one sentence"},
  "human_subjects": {"answer": true|false, "why": "one sentence"},
  "ai_assistants": {"answer": true|false, "why": "one sentence"}
}"""

def question_prompt(ctx: CheckContext, question: Question) -> str:
    """Shared context first, question-specific instructions last."""
    parts = [paper_context(ctx), "\n=== CHECKLIST ITEM ===\n", f"Item {question.code}: {question.question}\n"]
    if question.guidance:
        parts.append(f"\nWhat the ARR guidelines say this item covers:\n{question.guidance}\n")
    if question.prompt:
        parts.append(f"\n{question.prompt}\n")
    if question.evidence_hint:
        parts.append(f"\nIn `where`, name {question.evidence_hint}.\n")
    parts.append(
        "\nAnswer for THIS ITEM ONLY. Use status 'yes' only if a reviewer could point at the "
        "information in the paper, 'partial' if some but not all of it is there, 'no' if it is "
        "absent, and 'not_applicable' if the item does not apply to this kind of paper.\n"
    )
    return "".join(parts)


def scope_prompt(ctx: CheckContext) -> str:
    """One call that decides which checklist sections apply to this paper."""
    items = "\n".join(f"- {key}: {text}" for key, text in SCOPE_QUESTIONS.items())
    return (
        paper_context(ctx)
        + "\n=== APPLICABILITY ===\n"
        + "The ARR checklist gates whole sections on what the paper actually did. Answer each of "
        "these about the paper above:\n\n"
        + items
        + "\n\nMost NLP papers use scientific artifacts (any code, dataset, or model, including "
        "ones they merely used rather than created), so default to true for `artifacts` unless the "
        "paper is purely theoretical. Answer `human_subjects` true only for human annotation, "
        "crowdsourcing, or human participants — not for text written by humans in a dataset. "
        "Answer `ai_assistants` true only if the paper says AI assistants were used in the "
        "research, coding, or writing; a paper that merely evaluates language models does not "
        "count.\n"
    )
