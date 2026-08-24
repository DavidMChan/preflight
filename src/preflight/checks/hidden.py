"""Hidden-content and prompt-injection checks.

The CFP requires a submission to be directed at human readers and to be clearly
visible; text that is rendered invisibly, microscopically, in the page colour,
or outside the CropBox is aimed at an automated reader instead, and instructions
addressed to an automated reviewer are a desk-rejection condition. Every
threshold and pattern comes from the profile, never from this file.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..context import CheckContext
from ..document import INVISIBLE_RENDER_MODES, BBox, PageInfo, Span
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.hidden"

CATEGORY = "integrity"

DEFAULT_INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|preceding|earlier)\s+"
    r"(?:instructions|prompts|guidance)",
    r"disregard\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|prior|above|preceding)\s+"
    r"(?:instructions|prompts|guidance)",
    r"override\s+(?:your|the)\s+(?:previous\s+)?(?:instructions|system\s+prompt)",
    r"if\s+you\s+are\s+(?:an?\s+)?(?:ai|llm|large\s+language\s+model|language\s+model|"
    r"automated\s+reviewer|machine\s+reviewer)",
    r"as\s+an\s+ai\s+(?:language\s+model|assistant|reviewer)",
    r"(?:this\s+paper|this\s+submission)\s+(?:must|should)\s+be\s+accepted",
    r"(?:recommend|give|assign|write)[^.\n]{0,40}(?:accept|acceptance|positive\s+review|"
    r"strong\s+accept)",
    r"(?:give|assign|award)[^.\n]{0,40}(?:highest|maximum|top|perfect)[^.\n]{0,20}"
    r"(?:score|rating|grade)",
    r"do\s+not\s+(?:mention|reveal|disclose|report|output)\s+(?:this|these|the\s+following)",
    r"only\s+(?:highlight|emphasi[sz]e|discuss)\s+(?:the\s+)?(?:strengths|positives)",
    # Steering the review's wording rather than its verdict. Papers that study
    # prompting can phrase things this way too, which is what the model
    # adjudication in `llm_injection` is there to sort out.
    r"in\s+your\s+(?:output|response|review|answer|summary)[^.\n]{0,60}"
    r"(?:must|should|make\s+sure)",
    r"(?:must|should)\s+(?:include|contain|use|repeat)\s+(?:all\s+of\s+)?the\s+following"
    r"\s+(?:phrases|phrase|sentences|words|terms)",
    r"include\s+(?:all\s+of\s+)?the\s+following\s+phrases",
)


@dataclass(slots=True, frozen=True)
class _Run:
    """One run of glyphs as the page's content stream actually drew it.

    The parsed :class:`Span` view is built from ``get_text("dict")``, which drops
    glyphs painted outside the page rectangle and does not always carry the text
    rendering mode. The raw trace keeps both, which is exactly what concealment
    tricks exploit.
    """

    text: str
    bbox: BBox
    font: str
    size: float
    opacity: float
    mode: int
    page: int

    @property
    def invisible(self) -> bool:
        return self.mode in INVISIBLE_RENDER_MODES or self.opacity <= 0.01


def _char_union(chars: list) -> BBox | None:
    boxes = [c[3] for c in chars if len(c) > 3 and c[3]]
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _runs(ctx: CheckContext, page: PageInfo) -> list[_Run]:
    """Raw glyph runs for one page, cached on the context, spans as a fallback."""
    cache: dict[int, list[_Run]] = ctx.shared.setdefault("hidden.runs", {})
    if page.number in cache:
        return cache[page.number]

    try:
        trace = ctx.doc.doc[page.number - 1].get_texttrace()
    except Exception:  # pragma: no cover - older PyMuPDF or a damaged page
        trace = []

    runs: list[_Run] = []
    for item in trace:
        chars = list(item.get("chars") or [])
        text = "".join(chr(c[0]) for c in chars if isinstance(c[0], int) and 0 < c[0] < 0x110000)
        if not text.strip():
            continue
        bbox = item.get("bbox") or _char_union(chars)
        if not bbox:
            continue
        runs.append(
            _Run(
                text=text,
                bbox=tuple(float(v) for v in bbox),  # type: ignore[arg-type]
                font=str(item.get("font", "")),
                size=float(item.get("size", 0.0)),
                opacity=float(item.get("opacity", 1.0)),
                mode=int(item.get("type", 0)),
                page=page.number,
            )
        )

    if not runs:
        runs = [
            _Run(s.text, s.bbox, s.font, s.size, s.opacity, s.render_mode, page.number)
            for s in page.spans
        ]
    cache[page.number] = runs
    return runs


def _enabled(ctx: CheckContext) -> bool:
    return bool(ctx.conf("hidden.enabled", True))


def _ignore_pattern(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


def _alpha_len(text: str) -> int:
    return sum(1 for c in text if c.isalpha())


def _quote(item: Span | _Run, limit: int = 160) -> str:
    text = " ".join(item.text.split())
    return text[:limit]


def _inside_graphic(page: PageInfo, bbox: BBox) -> bool:
    """True if an image or filled drawing sits behind this bbox.

    Text over artwork may legitimately be white, and text inside a figure may
    legitimately be tiny, so both cases are treated conservatively.
    """
    return page.inside_graphic(bbox)


def _is_ocr_layer_page(ctx: CheckContext, page: PageInfo) -> bool:
    """A page dominated by one image is a scan; its invisible text is an OCR layer."""
    return page.image_coverage >= float(ctx.conf("hidden.ocr_image_coverage", 0.6))


def _page_list(pages: Iterable[int | None]) -> str:
    seen = sorted({p for p in pages if p})
    head = ", ".join(str(p) for p in seen[:10])
    return f"{head}{'...' if len(seen) > 10 else ''}"


@register("hidden_invisible_text", "Invisible text", module=MODULE, category=CATEGORY, order=60)
def check_invisible_text(ctx: CheckContext) -> Finding | None:
    """Flag text drawn in a no-mark render mode or at zero opacity."""
    if not _enabled(ctx):
        return None
    cap = int(ctx.conf("hidden.max_reported_spans", 10))
    error_chars = int(ctx.conf("hidden.invisible_text_error_chars", 40))

    offenders: list[Evidence] = []
    total = 0
    skipped_ocr: list[int] = []
    for page in ctx.doc.pages:
        hidden = [r for r in _runs(ctx, page) if r.invisible and _alpha_len(r.text)]
        if not hidden:
            continue
        if _is_ocr_layer_page(ctx, page):
            skipped_ocr.append(page.number)
            continue
        for run in hidden:
            total += _alpha_len(run.text)
            offenders.append(
                Evidence(
                    page=run.page,
                    detail=f"render mode {run.mode}, opacity {run.opacity:.2f}, "
                    f"font {run.font} at {run.size:.1f} pt",
                    quote=_quote(run),
                    bbox=run.bbox,
                )
            )

    if not offenders:
        note = (
            f" (invisible text on page(s) {_page_list(skipped_ocr)} was treated as a scanned OCR layer)"
            if skipped_ocr
            else ""
        )
        return ctx.ok(
            "hidden_invisible_text",
            "Invisible text",
            f"No text is drawn invisibly{note}.",
            category=CATEGORY,
        )

    pages = _page_list(e.page for e in offenders)
    message = (
        f"{len(offenders)} text run(s) ({total} alphabetic characters) are rendered invisibly "
        f"on page(s) {pages}. Invisible text is read by automated tools but not by human "
        "reviewers, and the CFP requires the paper to be directed at human readers."
    )
    severity_fn = ctx.error if total > error_chars else ctx.warn
    return severity_fn(
        "hidden_invisible_text",
        "Invisible text",
        message,
        category=CATEGORY,
        evidence=sorted(offenders, key=lambda e: -len(e.quote or ""))[:cap],
        remedy="Remove the invisible runs entirely. If they came from a figure export or a "
        "redaction, re-export the graphic so the text is not carried into the PDF.",
        confidence="high — read from the PDF text-rendering mode and alpha, not inferred",
    )


@register("hidden_tiny_text", "Microscopic text", module=MODULE, category=CATEGORY, order=61)
def check_tiny_text(ctx: CheckContext) -> Finding | None:
    """Flag visible text set so small that no human reader can read it."""
    if not _enabled(ctx):
        return None
    limit = float(ctx.conf("hidden.hidden_font_size_pt", 4.0))
    cap = int(ctx.conf("hidden.max_reported_spans", 10))
    ignore = _ignore_pattern(ctx)

    body_offenders: list[Evidence] = []
    figure_offenders: list[Evidence] = []
    body_chars = 0
    figure_chars = 0

    for page in ctx.doc.pages:
        for span in page.spans:
            text = span.text.strip()
            if not text or span.invisible or ignore.match(text) or not _alpha_len(text):
                continue
            if span.size > limit:
                continue
            evidence = Evidence(
                page=span.page,
                detail=f"font {span.font}",
                quote=_quote(span),
                measured=span.size,
                expected=f"> {limit:.1f} pt",
                bbox=span.bbox,
            )
            if _inside_graphic(page, span.bbox):
                figure_offenders.append(evidence)
                figure_chars += _alpha_len(text)
            else:
                body_offenders.append(evidence)
                body_chars += _alpha_len(text)

    if not body_offenders and not figure_offenders:
        return ctx.ok(
            "hidden_tiny_text",
            "Microscopic text",
            f"No visible text is set at or below {limit:.1f} pt.",
            category=CATEGORY,
        )

    offenders = body_offenders + figure_offenders
    evidence = sorted(offenders, key=lambda e: e.measured or 0.0)[:cap]
    if body_offenders:
        return ctx.error(
            "hidden_tiny_text",
            "Microscopic text",
            f"{len(body_offenders)} text run(s) ({body_chars} alphabetic characters) in the text "
            f"flow are set at or below {limit:.1f} pt on page(s) "
            f"{_page_list(e.page for e in body_offenders)}. Text this small is not addressed to a "
            "human reader.",
            category=CATEGORY,
            evidence=evidence,
            remedy="Delete the microscopic runs, or set them at the style's normal sizes.",
            confidence="high — measured from span font sizes",
        )
    return ctx.warn(
        "hidden_tiny_text",
        "Microscopic text",
        f"{len(figure_offenders)} text run(s) ({figure_chars} alphabetic characters) at or below "
        f"{limit:.1f} pt sit inside figures on page(s) "
        f"{_page_list(e.page for e in figure_offenders)}. This is most likely an illegible screenshot "
        "or plot label rather than hidden content, but a reviewer cannot read it either.",
        category=CATEGORY,
        evidence=evidence,
        remedy="Re-export the figure at a larger label size, or crop it so the labels are legible.",
        confidence="medium — text inside artwork is usually a small figure label, not concealment",
    )


@register("hidden_low_contrast", "Text in the page colour", module=MODULE, category=CATEGORY, order=62)
def check_low_contrast(ctx: CheckContext) -> Finding | None:
    """Flag text whose colour matches the (assumed white) page background."""
    if not _enabled(ctx):
        return None
    delta = float(ctx.conf("hidden.white_text_luminance_delta", 0.06))
    background = float(ctx.conf("hidden.page_background_luminance", 1.0))
    error_chars = int(ctx.conf("hidden.low_contrast_error_chars", 40))
    cap = int(ctx.conf("hidden.max_reported_spans", 10))

    offenders: list[Evidence] = []
    total = 0
    for page in ctx.doc.pages:
        for span in page.spans:
            text = span.text.strip()
            if not text or span.invisible or not _alpha_len(text):
                continue
            if abs(span.luminance - background) > delta:
                continue
            # White on artwork is normal; only bare-page text is concealment.
            if _inside_graphic(page, span.bbox):
                continue
            total += _alpha_len(text)
            offenders.append(
                Evidence(
                    page=span.page,
                    detail=f"colour #{span.color:06x} on an assumed white page",
                    quote=_quote(span),
                    measured=span.luminance,
                    expected=f"luminance differing from {background:.2f} by > {delta:.2f}",
                    bbox=span.bbox,
                )
            )

    if not offenders:
        return ctx.ok(
            "hidden_low_contrast",
            "Text in the page colour",
            f"No text is set within {delta:.2f} luminance of the page background.",
            category=CATEGORY,
        )

    severity_fn = ctx.error if total > error_chars else ctx.warn
    return severity_fn(
        "hidden_low_contrast",
        "Text in the page colour",
        f"{len(offenders)} text run(s) ({total} alphabetic characters) are set in the page's own "
        f"colour on page(s) {_page_list(e.page for e in offenders)}, so they are invisible on the "
        "printed page but still extracted by automated tools.",
        category=CATEGORY,
        evidence=sorted(offenders, key=lambda e: -len(e.quote or ""))[:cap],
        remedy="Delete the runs, or colour them so a human reviewer can read them.",
        confidence="medium — the page background is assumed white where no drawing or image covers the text",
    )


@register("hidden_offpage_text", "Text outside the page", module=MODULE, category=CATEGORY, order=63)
def check_offpage_text(ctx: CheckContext) -> Finding | None:
    """Flag text whose box falls outside the CropBox a reader actually sees."""
    if not _enabled(ctx):
        return None
    slack = float(ctx.conf("hidden.outside_crop_slack_pt", 3.0))
    error_chars = int(ctx.conf("hidden.offpage_error_chars", 40))
    cap = int(ctx.conf("hidden.max_reported_spans", 10))

    offenders: list[Evidence] = []
    total = 0
    for page in ctx.doc.pages:
        crop = page.cropbox or page.mediabox
        cx0, cy0, cx1, cy1 = crop
        for run in _runs(ctx, page):
            if not _alpha_len(run.text):
                continue
            x0, y0, x1, y1 = run.bbox
            overshoot = max(cx0 - x1, x0 - cx1, cy0 - y1, y0 - cy1)
            if overshoot <= slack:
                continue
            total += _alpha_len(run.text)
            offenders.append(
                Evidence(
                    page=run.page,
                    detail=f"CropBox is ({cx0:.1f}, {cy0:.1f}, {cx1:.1f}, {cy1:.1f})",
                    quote=_quote(run),
                    measured=overshoot,
                    expected=f"<= {slack:.1f} pt outside the CropBox",
                    bbox=run.bbox,
                )
            )

    if not offenders:
        return ctx.ok(
            "hidden_offpage_text",
            "Text outside the page",
            "All text sits inside the visible page area.",
            category=CATEGORY,
        )

    severity_fn = ctx.error if total > error_chars else ctx.warn
    return severity_fn(
        "hidden_offpage_text",
        "Text outside the page",
        f"{len(offenders)} text run(s) ({total} alphabetic characters) sit outside the CropBox on "
        f"page(s) {_page_list(e.page for e in offenders)}. Such text never renders for a reader but "
        "is still extracted from the file.",
        category=CATEGORY,
        evidence=sorted(offenders, key=lambda e: -(e.measured or 0.0))[:cap],
        remedy="Remove the off-page content; do not park text beyond the trim edge.",
        confidence="high — compared against the page CropBox",
    )


def _compiled_patterns(ctx: CheckContext) -> list[tuple[str, re.Pattern[str]]]:
    raw = ctx.conf("hidden.prompt_injection_patterns", list(DEFAULT_INJECTION_PATTERNS)) or []
    out: list[tuple[str, re.Pattern[str]]] = []
    for entry in raw:
        try:
            out.append((str(entry), re.compile(str(entry), re.IGNORECASE)))
        except re.error:  # a malformed profile pattern must not sink the check
            continue
    return out


def _concealed_text(ctx: CheckContext, page: PageInfo) -> str:
    """All text on the page a human reader cannot read, as one blob."""
    limit = float(ctx.conf("hidden.hidden_font_size_pt", 4.0))
    delta = float(ctx.conf("hidden.white_text_luminance_delta", 0.06))
    background = float(ctx.conf("hidden.page_background_luminance", 1.0))
    parts: list[str] = [r.text for r in _runs(ctx, page) if r.invisible and r.text.strip()]
    for span in page.spans:
        if not span.text.strip():
            continue
        concealed = (
            span.invisible
            or span.size <= limit
            or (abs(span.luminance - background) <= delta and not _inside_graphic(page, span.bbox))
        )
        if concealed:
            parts.append(span.text)
    return " ".join(parts)


@register("hidden_prompt_injection", "Prompt injection", module=MODULE, category=CATEGORY, order=64)
def check_prompt_injection(ctx: CheckContext) -> Finding | None:
    """Search all text, visible and hidden, for instructions aimed at an automated reviewer."""
    if not _enabled(ctx):
        return None
    patterns = _compiled_patterns(ctx)
    if not patterns:
        return ctx.skip(
            "hidden_prompt_injection",
            "Prompt injection",
            "The profile defines no prompt-injection patterns to search for.",
            category=CATEGORY,
        )
    window = int(ctx.conf("hidden.injection_context_chars", 160))
    cap = int(ctx.conf("hidden.max_reported_matches", 10))

    visible_hits: list[Evidence] = []
    hidden_hits: list[Evidence] = []
    for page in ctx.doc.pages:
        page_text = " ".join(page.text.split())
        concealed = " ".join(_concealed_text(ctx, page).split())
        for source, pattern in patterns:
            for match in pattern.finditer(page_text):
                start = max(0, match.start() - window // 2)
                context = page_text[start : match.end() + window // 2]
                in_hidden = bool(pattern.search(concealed))
                evidence = Evidence(
                    page=page.number,
                    detail=f"matched pattern {source!r}"
                    + (" in text that is itself hidden or microscopic" if in_hidden else ""),
                    quote=context,
                )
                (hidden_hits if in_hidden else visible_hits).append(evidence)

    if not visible_hits and not hidden_hits:
        return ctx.ok(
            "hidden_prompt_injection",
            "Prompt injection",
            f"None of the {len(patterns)} prompt-injection patterns appear anywhere in the document "
            "text, visible or hidden.",
            category=CATEGORY,
        )

    if hidden_hits:
        return ctx.error(
            "hidden_prompt_injection",
            "Prompt injection",
            f"{len(hidden_hits)} prompt-injection pattern match(es) occur in text that is itself "
            f"hidden, microscopic, or in the page colour, on page(s) "
            f"{_page_list(e.page for e in hidden_hits)}. Concealed instructions addressed to an "
            "automated reviewer are unambiguous manipulation and the CFP allows desk rejection for "
            "them.",
            category=CATEGORY,
            evidence=(hidden_hits + visible_hits)[:cap],
            remedy="Remove the concealed text from the source and rebuild the PDF.",
            confidence="high — the matching text is both injection-shaped and unreadable by a human",
        )

    return ctx.warn(
        "hidden_prompt_injection",
        "Prompt injection",
        f"{len(visible_hits)} prompt-injection pattern match(es) appear in normally visible text on "
        f"page(s) {_page_list(e.page for e in visible_hits)}. A paper studying prompt injection will "
        "legitimately quote these strings, so check the context before acting; this is only a "
        "violation if the text is instructing an automated reviewer rather than describing one.",
        category=CATEGORY,
        evidence=visible_hits[:cap],
        remedy="If the strings are study material, quote them inside a figure, listing, or clearly "
        "marked example so no reader mistakes them for instructions.",
        confidence="low — visible matches are frequently legitimate subject matter",
    )
