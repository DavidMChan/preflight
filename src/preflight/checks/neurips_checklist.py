"""NeurIPS paper-checklist checks: presence, position, completeness, and honesty.

NeurIPS ships the checklist inside its LaTeX style file and desk-rejects papers
that submit without it, so the questions here are cheap to answer from text and
expensive to get wrong. Every threshold, alias and regex is read from the
profile; this file only knows the *shape* of the checklist, never a venue's
numbers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "neurips.checklist"

CATEGORY = "checklist"

#: The 16 questions in template order. Index + 1 is the question number.
QUESTION_KEYS: tuple[str, ...] = (
    "claims",
    "limitations",
    "theory_assumptions_proofs",
    "experimental_result_reproducibility",
    "open_access_data_code",
    "experimental_setting_details",
    "experiment_statistical_significance",
    "experiments_compute_resources",
    "code_of_ethics",
    "broader_impacts",
    "safeguards",
    "licenses",
    "assets",
    "crowdsourcing_human_subjects",
    "irb_approvals",
    "llm_usage_declaration",
)

DEFAULT_TITLE_ALIASES: tuple[str, ...] = (
    "NeurIPS Paper Checklist",
    "NeurIPS paper checklist",
    "Paper Checklist",
    "NeurIPS Checklist",
)

DEFAULT_ANSWER_REGEX = r"Answer\s*:\s*\[\s*([^\]\[]{0,12}?)\s*\]"
DEFAULT_QUESTION_REGEX = r"Question\s*:"
DEFAULT_JUSTIFICATION_REGEX = r"Justification\s*:"
DEFAULT_GUIDELINES_REGEX = r"Guidelines\s*:"
DEFAULT_PLACEHOLDER_REGEX = r"(?i)\b(TODO|answerTODO|answerYes|answerNo|answerNA|FILL IN|XXX+)\b|\\answer"


@dataclass(slots=True)
class Answer:
    """One parsed checklist item."""

    number: int
    key: str
    raw: str | None = None            # bracket contents exactly as extracted
    value: str | None = None          # normalised: "yes" / "no" / "na"
    justification: str = ""
    quote: str = ""

    @property
    def answered(self) -> bool:
        return self.value in {"yes", "no", "na"}


@dataclass(slots=True)
class Checklist:
    """The result of locating and parsing the checklist inside the PDF."""

    present: bool = False
    title_page: int | None = None
    start_page: int | None = None
    end_page: int | None = None
    answer_marker_count: int = 0
    question_marker_count: int = 0
    answers: list[Answer] = field(default_factory=list)
    placeholders: list[Evidence] = field(default_factory=list)
    text: str = ""

    @property
    def answered(self) -> list[Answer]:
        return [a for a in self.answers if a.answered]

    @property
    def unanswered(self) -> list[Answer]:
        return [a for a in self.answers if not a.answered]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join(text.split())


def _normalise_answer(raw: str) -> str | None:
    token = re.sub(r"[^a-z/]", "", raw.lower())
    if token in {"yes", "y"}:
        return "yes"
    if token in {"no", "n"}:
        return "no"
    if token in {"na", "n/a", "notapplicable"}:
        return "na"
    return None


def _title_page(ctx: CheckContext, aliases: list[str]) -> int | None:
    """First page whose heading or text carries the checklist title."""
    wanted = [re.sub(r"[^a-z ]", " ", a.lower()) for a in aliases]
    wanted = [_norm(a) for a in wanted if _norm(a)]
    for heading in ctx.doc.headings:
        if any(w in heading.normalized for w in wanted):
            return heading.page
    for page in ctx.doc.pages:
        flat = _norm(re.sub(r"[^A-Za-z ]", " ", page.text)).lower()
        if any(w in flat for w in wanted):
            return page.number
    return None


def _parse(ctx: CheckContext) -> Checklist:
    """Locate the checklist and parse its answers; cached on ``ctx.shared``."""
    cached = ctx.shared.get("neurips_checklist")
    if isinstance(cached, Checklist):
        return cached

    aliases = [str(a) for a in (ctx.conf("neurips.checklist_title_aliases", list(DEFAULT_TITLE_ALIASES)) or [])]
    answer_re = re.compile(str(ctx.conf("neurips.answer_regex", DEFAULT_ANSWER_REGEX)))
    question_re = re.compile(str(ctx.conf("neurips.question_regex", DEFAULT_QUESTION_REGEX)))
    just_re = re.compile(str(ctx.conf("neurips.justification_regex", DEFAULT_JUSTIFICATION_REGEX)))
    guide_re = re.compile(str(ctx.conf("neurips.guidelines_regex", DEFAULT_GUIDELINES_REGEX)))
    placeholder_re = re.compile(str(ctx.conf("neurips.placeholder_regex", DEFAULT_PLACEHOLDER_REGEX)))
    min_markers = int(ctx.conf("neurips.min_answer_markers", 4))
    expected = int(ctx.conf("neurips.expected_question_count", len(QUESTION_KEYS)))
    keys = [str(k) for k in (ctx.conf("neurips.question_keys", list(QUESTION_KEYS)) or [])] or list(QUESTION_KEYS)

    result = Checklist()
    result.title_page = _title_page(ctx, aliases)

    page_texts = {p.number: _norm(p.text) for p in ctx.doc.pages}
    answer_pages = [n for n, t in page_texts.items() if answer_re.search(t)]

    # The checklist is the last contiguous run of answer-bearing pages, optionally
    # opened by the title page.
    candidates = [p for p in (result.title_page, min(answer_pages) if answer_pages else None) if p]
    if not answer_pages and result.title_page is None:
        ctx.shared["neurips_checklist"] = result
        return result
    result.start_page = min(candidates) if candidates else None
    result.end_page = max(answer_pages) if answer_pages else result.start_page

    start = result.start_page or 1
    region = _norm(" ".join(page_texts.get(n, "") for n in range(start, ctx.doc.page_count + 1)))
    result.text = region
    result.answer_marker_count = len(answer_re.findall(region))
    result.question_marker_count = len(question_re.findall(region))
    result.present = bool(result.title_page) or result.answer_marker_count >= min_markers

    if not result.present:
        ctx.shared["neurips_checklist"] = result
        return result

    # Anchors delimit one question each. "Question:" is the template's own marker;
    # fall back to the answers themselves when extraction mangled it.
    anchors = [m.start() for m in question_re.finditer(region)]
    if len(anchors) < result.answer_marker_count:
        anchors = [m.start() for m in answer_re.finditer(region)]
    bounds = [*anchors, len(region)]
    count = max(len(anchors), 0)

    for idx in range(max(count, expected)):
        number = idx + 1
        key = keys[idx] if idx < len(keys) else f"question_{number}"
        item = Answer(number=number, key=key)
        if idx < count:
            segment = region[bounds[idx] : bounds[idx + 1]]
            m = answer_re.search(segment)
            if m:
                item.raw = m.group(1)
                item.value = _normalise_answer(m.group(1))
                item.quote = _norm(segment[max(0, m.start() - 60) : m.end() + 90])
            else:
                item.quote = _norm(segment[:120])
            jm = just_re.search(segment)
            if jm:
                tail = segment[jm.end() :]
                gm = guide_re.search(tail)
                item.justification = _norm(tail[: gm.start()] if gm else tail[:400])
        result.answers.append(item)

    for m in placeholder_re.finditer(region):
        page = _page_of(ctx, m.group(0), start)
        result.placeholders.append(
            Evidence(page=page, detail="template placeholder left in the checklist",
                     quote=_norm(region[max(0, m.start() - 50) : m.end() + 50]))
        )
    for m in re.finditer(r"Answer\s*:\s*\[\s*\]", region):
        result.placeholders.append(
            Evidence(page=result.start_page, detail="empty answer bracket",
                     quote=_norm(region[max(0, m.start() - 60) : m.end() + 40]))
        )

    ctx.shared["neurips_checklist"] = result
    return result


def _page_of(ctx: CheckContext, needle: str, start: int) -> int | None:
    """Best-effort page number for a snippet inside the checklist region."""
    for page in ctx.doc.pages[start - 1 :]:
        if needle and needle in _norm(page.text):
            return page.number
    return start if start <= ctx.doc.page_count else None


def _is_placeholder(text: str, placeholder_re: re.Pattern[str], min_chars: int) -> bool:
    stripped = text.strip(" .:-—_")
    if len(stripped) < min_chars:
        return True
    return bool(placeholder_re.search(stripped))


def _label(item: Answer) -> str:
    return f"Q{item.number} ({item.key})"


def _absent_skip(ctx: CheckContext, check_id: str, title: str) -> Finding:
    return ctx.skip(
        check_id, title,
        "No NeurIPS paper checklist was found in this PDF, so this check has nothing to inspect "
        "(see neurips_checklist_present).",
        category=CATEGORY,
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@register("neurips_checklist_present", "NeurIPS checklist present", module=MODULE,
          category=CATEGORY, order=60)
def check_checklist_present(ctx: CheckContext) -> Finding:
    """The checklist must be in the submitted PDF; omitting it is a desk rejection."""
    cl = _parse(ctx)
    if not cl.present:
        return ctx.error(
            "neurips_checklist_present",
            "NeurIPS checklist present",
            "No NeurIPS paper checklist was found in this PDF. Papers submitted without the checklist "
            "are desk-rejected without review.",
            category=CATEGORY,
            evidence=[Evidence(detail=f"searched all {ctx.doc.page_count} pages for a checklist title "
                                      f"and for 'Answer: [...]' markers", measured=0.0,
                               expected="a checklist section after the references")],
            remedy="Paste the checklist from the official NeurIPS style file at the end of the PDF "
                   "(after the paper and any technical appendices) and answer every question.",
        )
    evidence = [
        Evidence(page=cl.start_page, detail="checklist begins here",
                 quote=_norm(ctx.doc.page_text(cl.start_page or 1))[:120] if cl.start_page else None),
        Evidence(detail=f"{cl.answer_marker_count} 'Answer: [...]' marker(s), "
                        f"{cl.question_marker_count} 'Question:' marker(s)"),
    ]
    return ctx.ok(
        "neurips_checklist_present",
        "NeurIPS checklist present",
        f"The NeurIPS paper checklist is present, starting on page {cl.start_page} "
        f"and running through page {cl.end_page}.",
        category=CATEGORY,
        evidence=evidence,
    )


@register("neurips_checklist_position", "NeurIPS checklist position", module=MODULE,
          category=CATEGORY, order=61)
def check_checklist_position(ctx: CheckContext) -> Finding:
    """The checklist must come last: after the references and after any appendix."""
    cl = _parse(ctx)
    if not cl.present or cl.start_page is None:
        return _absent_skip(ctx, "neurips_checklist_position", "NeurIPS checklist position")

    refs = ctx.doc.find_heading([str(a) for a in (ctx.conf("structure.references_aliases",
                                                          ["References", "Bibliography"]) or [])])
    appendix = ctx.doc.find_heading([str(a) for a in (ctx.conf("structure.appendix_aliases",
                                                              ["Appendix", "Appendices",
                                                               "Supplementary Material"]) or [])])
    evidence: list[Evidence] = [Evidence(page=cl.start_page, detail="checklist starts on this page")]
    problems: list[str] = []

    if refs is not None:
        evidence.append(Evidence(page=refs.page, detail="References heading", quote=refs.text))
        if cl.start_page < refs.page:
            problems.append(f"it starts on page {cl.start_page}, before the references on page {refs.page}")
    if appendix is not None:
        evidence.append(Evidence(page=appendix.page, detail="Appendix heading", quote=appendix.text))
        if cl.start_page < appendix.page:
            problems.append(f"it starts on page {cl.start_page}, before the appendix on page {appendix.page}")

    if problems:
        return ctx.error(
            "neurips_checklist_position",
            "NeurIPS checklist position",
            "The checklist is out of order: " + "; ".join(problems) + ". The required order in the single "
            "PDF is paper, then optional technical appendices, then the checklist.",
            category=CATEGORY,
            evidence=evidence,
            remedy="Move the \\input of the checklist to the very end of the document, after the "
                   "bibliography and all appendices.",
        )
    if refs is None and appendix is None:
        return ctx.warn(
            "neurips_checklist_position",
            "NeurIPS checklist position",
            f"The checklist starts on page {cl.start_page}, but no References or Appendix heading could be "
            "located, so its position could not be confirmed.",
            category=CATEGORY,
            evidence=evidence,
            confidence="low — heading detection found no references section",
        )
    return ctx.ok(
        "neurips_checklist_position",
        "NeurIPS checklist position",
        f"The checklist starts on page {cl.start_page}, after everything it has to follow.",
        category=CATEGORY,
        evidence=evidence,
    )


@register("neurips_checklist_complete", "NeurIPS checklist completeness", module=MODULE,
          category=CATEGORY, order=62)
def check_checklist_complete(ctx: CheckContext) -> Finding:
    """Every one of the questions needs a parseable [Yes] / [No] / [NA] answer."""
    cl = _parse(ctx)
    if not cl.present:
        return _absent_skip(ctx, "neurips_checklist_complete", "NeurIPS checklist completeness")

    expected = int(ctx.conf("neurips.expected_question_count", len(QUESTION_KEYS)))
    found = min(len(cl.answers), max(cl.question_marker_count, len(cl.answered)))
    missing = [a for a in cl.answers if a.number > found]
    unanswered = [a for a in cl.unanswered if a.number <= found]

    evidence: list[Evidence] = [
        Evidence(page=cl.start_page, detail="questions located in the checklist",
                 measured=float(found), expected=f"{expected}"),
        Evidence(page=cl.start_page, detail="questions carrying a parseable answer",
                 measured=float(len(cl.answered)), expected=f"{expected}"),
    ]
    for item in unanswered[:6]:
        evidence.append(Evidence(page=cl.start_page, detail=f"{_label(item)} has no [Yes]/[No]/[NA] answer",
                                 quote=item.quote or None))
    evidence.extend(cl.placeholders[:4])

    tally = {v: sum(1 for a in cl.answered if a.value == v) for v in ("yes", "no", "na")}
    summary = f"{tally['yes']} Yes, {tally['no']} No, {tally['na']} NA"

    if missing or unanswered or cl.placeholders:
        bits: list[str] = []
        if missing:
            bits.append("missing question(s) " + ", ".join(str(a.number) for a in missing))
        if unanswered:
            bits.append("unanswered question(s) " + ", ".join(str(a.number) for a in unanswered))
        if cl.placeholders:
            bits.append(f"{len(cl.placeholders)} leftover template placeholder(s)")
        return ctx.error(
            "neurips_checklist_complete",
            "NeurIPS checklist completeness",
            f"The checklist is incomplete ({len(cl.answered)}/{expected} answered; {summary}): "
            + "; ".join(bits) + ". Every question must be answered Yes, No, or NA.",
            category=CATEGORY,
            evidence=evidence[:12],
            remedy="Answering No is fine when justified — leaving a question blank or on its \\answerTODO "
                   "placeholder is not.",
        )
    return ctx.ok(
        "neurips_checklist_complete",
        "NeurIPS checklist completeness",
        f"All {len(cl.answered)} of the {expected} checklist questions carry an answer ({summary}).",
        category=CATEGORY,
        evidence=evidence[:4],
    )


@register("neurips_checklist_justifications", "NeurIPS checklist justifications", module=MODULE,
          category=CATEGORY, order=63)
def check_checklist_justifications(ctx: CheckContext) -> Finding:
    """A No or NA answer is acceptable, but only when it is actually justified."""
    cl = _parse(ctx)
    if not cl.present:
        return _absent_skip(ctx, "neurips_checklist_justifications", "NeurIPS checklist justifications")

    need = {str(v).lower() for v in (ctx.conf("neurips.justification_required_answers", ["no", "na"]) or [])}
    min_chars = int(ctx.conf("neurips.min_justification_chars", 15))
    placeholder_re = re.compile(str(ctx.conf("neurips.placeholder_regex", DEFAULT_PLACEHOLDER_REGEX)))

    targets = [a for a in cl.answered if a.value in need]
    thin = [a for a in targets if _is_placeholder(a.justification, placeholder_re, min_chars)]

    if not targets:
        return ctx.ok(
            "neurips_checklist_justifications",
            "NeurIPS checklist justifications",
            f"No answers of {'/'.join(sorted(need)).upper()} to justify.",
            category=CATEGORY,
        )
    if thin:
        evidence = [
            Evidence(page=cl.start_page,
                     detail=f"{_label(a)} answered [{(a.raw or '').strip()}] with a "
                            f"{len(a.justification.strip())}-character justification",
                     quote=a.justification or a.quote or None,
                     measured=float(len(a.justification.strip())),
                     expected=f">= {min_chars} characters of real text")
            for a in thin[:8]
        ]
        return ctx.warn(
            "neurips_checklist_justifications",
            "NeurIPS checklist justifications",
            f"{len(thin)} of {len(targets)} No/NA answer(s) have an empty or placeholder justification: "
            + ", ".join(f"Q{a.number}" for a in thin)
            + ". Answering No is explicitly not grounds for rejection, but it must be justified.",
            category=CATEGORY,
            evidence=evidence,
            remedy="Write one or two sentences per No/NA answer explaining why.",
            confidence="medium — justification text is recovered from extracted PDF text",
        )
    return ctx.ok(
        "neurips_checklist_justifications",
        "NeurIPS checklist justifications",
        f"All {len(targets)} No/NA answer(s) carry a written justification.",
        category=CATEGORY,
    )


@register("neurips_checklist_consistency", "NeurIPS checklist consistency", module=MODULE,
          category=CATEGORY, order=64)
def check_checklist_consistency(ctx: CheckContext) -> Finding:
    """Cross-check a few Yes answers against what the paper itself contains."""
    cl = _parse(ctx)
    if not cl.present:
        return _absent_skip(ctx, "neurips_checklist_consistency", "NeurIPS checklist consistency")

    by_key = {a.key: a for a in cl.answers}
    body = ctx.doc.text
    if cl.start_page and cl.start_page > 1:
        body = "\n".join(ctx.doc.page_text(n) for n in range(1, cl.start_page))

    conflicts: list[Evidence] = []

    limitations = by_key.get("limitations")
    if limitations is not None and limitations.value == "yes":
        aliases = [str(a) for a in (ctx.conf("structure.limitations_aliases", ["Limitations"]) or [])]
        heading = ctx.doc.find_heading(aliases)
        if heading is None:
            conflicts.append(Evidence(page=cl.start_page,
                                      detail="Q2 (limitations) answered [Yes] but no Limitations section "
                                             "heading was found in the paper",
                                      expected=f"a heading matching one of: {', '.join(aliases[:4])}"))

    data_code = by_key.get("open_access_data_code")
    if data_code is not None and data_code.value == "yes":
        urls = ctx.doc.textual_urls + ctx.doc.hyperlinks
        if not urls:
            conflicts.append(Evidence(page=cl.start_page,
                                      detail="Q5 (open access to data and code) answered [Yes] but the "
                                             "document contains no URL or hyperlink at all",
                                      expected="a repository or data link (an anonymised one is fine)"))

    compute = by_key.get("experiments_compute_resources")
    if compute is not None and compute.value == "yes":
        keywords = [str(k) for k in (ctx.conf("neurips.compute_keywords",
                                              ["GPU", "CPU", "TPU", "A100", "V100", "H100", "compute",
                                               "GPU-hours", "cluster", "runtime"]) or [])]
        low = body.lower()
        if not any(k.lower() in low for k in keywords):
            conflicts.append(Evidence(page=cl.start_page,
                                      detail="Q8 (compute resources) answered [Yes] but the paper body "
                                             "never mentions any compute hardware or budget",
                                      expected=f"one of: {', '.join(keywords[:6])}"))

    if conflicts:
        return ctx.warn(
            "neurips_checklist_consistency",
            "NeurIPS checklist consistency",
            f"{len(conflicts)} checklist answer(s) look inconsistent with the paper body. "
            "Reviewers check these against the text, so make sure the paper backs up each Yes.",
            category=CATEGORY,
            evidence=conflicts,
            confidence="medium — heuristic cross-reference against extracted text",
        )
    return ctx.ok(
        "neurips_checklist_consistency",
        "NeurIPS checklist consistency",
        "The checklist answers we can cross-check are consistent with the paper body.",
        category=CATEGORY,
    )
