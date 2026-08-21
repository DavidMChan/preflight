"""Run a single Responsible NLP checklist item against a PDF.

    uv run python scripts/try_question.py A1 /path/to/paper.pdf

Used while authoring question specs: it exercises exactly one item so you are
not paying for the whole checklist on every iteration.
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks
from preflight.rnlp import load_questions


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    code, pdf = sys.argv[1].upper(), Path(sys.argv[2])
    conference = sys.argv[3] if len(sys.argv) > 3 else "arr"

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    registry = load_builtin_checks()

    question = next((q for q in load_questions() if q.code.upper() == code), None)
    if question is None:
        print(f"No checklist item {code}. Available: {', '.join(q.code for q in load_questions())}")
        return 2

    profile = load_profile(conference)
    doc = Document(pdf)
    try:
        ctx = CheckContext(doc=doc, profile=profile, track=profile.track(None),
                           settings=Settings(enable_llm=True, llm_model="gpt-5.6-luna"))
        check = registry.checks[question.check_id]
        print(f"=== {question.display} ({question.applies_when}) ===\n")
        for finding in check.run(ctx):
            print(f"[{finding.severity.value.upper()}] {finding.message}\n")
            for ev in finding.evidence:
                rendered = ev.render()
                if rendered:
                    print(f"  - {rendered}")
            if finding.confidence:
                print(f"  confidence: {finding.confidence}")
            if finding.remedy:
                print(f"  fix: {finding.remedy}")
        scope = ctx.shared.get("rnlp_scope")
        if scope:
            print("\nscope:", {k: v.get("answer") for k, v in scope.items() if isinstance(v, dict)})
    finally:
        doc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
