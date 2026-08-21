"""Document-structure checks: page limits, Limitations, references, appendices.

The page-limit rule is the reason this module cannot just count PDF pages. The
CFP grants unlimited space after the conclusion for Limitations, references and
appendices, so the limit applies to *content* pages only — the ones before the
first unlimited section begins.
"""

from __future__ import annotations

import re

from ..context import CheckContext
from ..document import Heading, Line
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.structure"

# Section kinds that end the page-limited part of the paper.
_KIND_ALIAS_KEYS = {
    "limitations": "structure.limitations_aliases",
    "references": "structure.references_aliases",
    "acknowledgments": "structure.acknowledgments_aliases",
    "ethics": "structure.ethics_aliases",
    "appendix": "structure.appendix_aliases",
}


def _aliases(ctx: CheckContext, kind: str) -> list[str]:
    key = _KIND_ALIAS_KEYS[kind]
    default = [kind.capitalize()]
    return [str(a) for a in (ctx.conf(key, default) or default)]


def _headings(ctx: CheckContext) -> dict[str, Heading | None]:
    """Locate each structural landmark once and cache it for the whole run."""
    cached = ctx.shared.get("structure_headings")
    if cached is not None:
        return cached
    found = {kind: ctx.doc.find_heading(_aliases(ctx, kind)) for kind in _KIND_ALIAS_KEYS}
    ctx.shared["structure_headings"] = found
    return found


def _first_unlimited(ctx: CheckContext) -> Heading | None:
    """The earliest heading after which pages stop counting against the limit."""
    kinds = [str(k) for k in (ctx.conf("structure.unlimited_after", list(_KIND_ALIAS_KEYS)) or [])]
    found = _headings(ctx)
    candidates = [found[k] for k in kinds if found.get(k) is not None]
    return min(candidates, key=lambda h: h.order) if candidates else None  # type: ignore[arg-type]


def _line_index(ctx: CheckContext, heading: Heading) -> int | None:
    for i, line in enumerate(ctx.doc.reading_order):
        if line.page == heading.page and abs(line.bbox[1] - heading.bbox[1]) < 0.6:
            return i
    return None


def _is_content_line(line: Line) -> bool:
    text = line.text.strip()
    return bool(text) and not text.isdigit() and any(c.isalpha() for c in text)


@register("page_limit", "Content page limit", module=MODULE, category="structure", order=30)
def check_page_limit(ctx: CheckContext) -> Finding:
    """Count content pages up to the first unlimited section, not total PDF pages."""
    limit = ctx.track.content_page_limit
    marker = _first_unlimited(ctx)

    if marker is None:
        # No Limitations/References/appendix landmark at all: every page counts.
        content_pages = ctx.doc.page_count
        note = (
            "No Limitations, references or appendix heading was found, so every page in the PDF "
            "counts as content."
        )
        evidence = [Evidence(detail="no unlimited-section landmark detected",
                             measured=float(content_pages), expected=f"<= {limit} content pages")]
    else:
        idx = _line_index(ctx, marker)
        content_pages = marker.page
        if idx is not None:
            preceding = [ln for ln in ctx.doc.reading_order[:idx] if _is_content_line(ln)]
            if preceding:
                # If nothing precedes the landmark on its own page, content ended
                # on the previous page and this page is entirely unlimited material.
                content_pages = preceding[-1].page
        note = f"Content runs up to the {marker.text!r} heading on page {marker.page}."
        evidence = [
            Evidence(page=marker.page, detail=f"first unlimited section: {marker.text!r}",
                     measured=float(content_pages), expected=f"<= {limit} content pages"),
            Evidence(detail=f"{ctx.doc.page_count} pages total in the PDF "
                            f"({ctx.doc.page_count - content_pages} of them unlimited material)"),
        ]

    if content_pages > limit:
        return ctx.error(
            "page_limit",
            "Content page limit",
            f"Main content appears to run through page {content_pages}; {ctx.track.name} papers allow "
            f"{limit}. {note} Exceeding the page limit is explicitly a desk-rejection condition.",
            category="structure",
            evidence=evidence,
            remedy="Move material into the appendix (which is unlimited and sits after the references), "
            "or cut content. Do not shrink fonts or margins to fit.",
            confidence="high — derived from the position of the first unlimited section, not raw page count",
        )
    return ctx.ok(
        "page_limit",
        "Content page limit",
        f"{content_pages} content page(s) against a limit of {limit} for {ctx.track.name} papers. {note}",
        category="structure",
        evidence=evidence,
    )


@register("limitations_present", "Limitations section", module=MODULE, category="structure", order=31)
def check_limitations_present(ctx: CheckContext) -> Finding:
    """A section titled exactly 'Limitations' is mandatory and desk-rejectable if absent."""
    if not ctx.conf("structure.limitations_required", True):
        return ctx.skip("limitations_present", "Limitations section",
                        "This venue does not require a Limitations section.", category="structure")

    heading = _headings(ctx)["limitations"]
    if heading is None:
        return ctx.error(
            "limitations_present",
            "Limitations section",
            "No 'Limitations' heading was detected. A section with this exact title is mandatory and "
            "omitting it is explicitly grounds for desk rejection.",
            category="structure",
            remedy="Add an unnumbered section titled exactly 'Limitations' after the conclusion and before "
            "the references.",
            confidence="high — but verify manually if your heading uses unusual formatting",
        )

    expected = str(ctx.conf("structure.limitations_heading", "Limitations"))
    evidence = [Evidence(page=heading.page, detail=f"heading {heading.text!r}", quote=heading.text)]
    if heading.text.strip().lower() != expected.lower():
        return ctx.warn(
            "limitations_present",
            "Limitations section",
            f"Found a section titled {heading.text!r}; the CFP requires the title {expected!r} exactly.",
            category="structure", evidence=evidence,
            remedy=f"Rename the heading to {expected!r}.",
        )
    if heading.numbering:
        return ctx.warn(
            "limitations_present",
            "Limitations section",
            f"The Limitations section is numbered ({heading.numbering}). The ACL template leaves it unnumbered.",
            category="structure", evidence=evidence,
            remedy="Use \\section*{Limitations} so it is not numbered.",
        )
    return ctx.ok("limitations_present", "Limitations section",
                  f"'Limitations' section found on page {heading.page}.",
                  category="structure", evidence=evidence)


@register("limitations_position", "Limitations placement", module=MODULE, category="structure", order=32)
def check_limitations_position(ctx: CheckContext) -> Finding | None:
    """Limitations must sit at the end of the paper, before the references."""
    found = _headings(ctx)
    limitations, references = found["limitations"], found["references"]
    if limitations is None:
        return None  # already reported by limitations_present

    if references is None:
        return ctx.warn(
            "limitations_position", "Limitations placement",
            "Could not locate a References heading, so the position of Limitations could not be confirmed.",
            category="structure", confidence="low — heading detection may have missed the bibliography",
        )

    evidence = [
        Evidence(page=limitations.page, detail=f"Limitations on page {limitations.page}"),
        Evidence(page=references.page, detail=f"References on page {references.page}"),
    ]
    if limitations.order > references.order:
        return ctx.error(
            "limitations_position", "Limitations placement",
            f"The Limitations section (page {limitations.page}) appears after the references "
            f"(page {references.page}). It must come at the end of the paper but before the references.",
            category="structure", evidence=evidence,
            remedy="Move the Limitations section above \\bibliography.",
        )

    # Anything numbered after Limitations means it is not actually at the end.
    trailing = [
        h for h in ctx.doc.headings
        if limitations.order < h.order < references.order and h.numbering and re.match(r"^\d", h.numbering)
    ]
    if trailing:
        return ctx.warn(
            "limitations_position", "Limitations placement",
            f"{len(trailing)} numbered section(s) appear between Limitations and the references, so "
            "Limitations is not the last section of the paper.",
            category="structure",
            evidence=[*evidence, *[Evidence(page=h.page, detail=f"section {h.numbering} {h.text!r}")
                                   for h in trailing[:5]]],
        )
    return ctx.ok("limitations_position", "Limitations placement",
                  f"Limitations (page {limitations.page}) sits after the main content and before the "
                  f"references (page {references.page}).",
                  category="structure", evidence=evidence)


@register("limitations_scope", "Limitations content", module=MODULE, category="structure", order=33)
def check_limitations_scope(ctx: CheckContext) -> Finding | None:
    """Heuristics for new methods/results smuggled into the Limitations section.

    Whether the section introduces new contributions is a semantic question, so
    this only ever warns and names what triggered it.
    """
    found = _headings(ctx)
    limitations, references = found["limitations"], found["references"]
    if limitations is None:
        return None

    body = ctx.doc.text_between(limitations, references)
    if not body.strip():
        return ctx.warn("limitations_scope", "Limitations content",
                        "The Limitations section appears to be empty.", category="structure",
                        cfp_key="limitations_scope")

    signals: list[Evidence] = []
    patterns = [str(p) for p in (ctx.conf("structure.limitations_new_content_patterns", []) or [])]
    lowered = body.lower()
    for pattern in patterns:
        pos = lowered.find(pattern.lower())
        if pos >= 0:
            signals.append(
                Evidence(page=limitations.page, detail=f"phrase {pattern!r}",
                         quote=body[max(0, pos - 60): pos + 100])
            )

    # A new subsection or a figure/table first introduced here is a stronger tell.
    inner = [h for h in ctx.doc.headings
             if references is not None and limitations.order < h.order < references.order]
    if inner:
        signals.append(Evidence(page=inner[0].page,
                                detail=f"{len(inner)} heading(s) inside Limitations, e.g. {inner[0].text!r}"))
    for label in ("Figure", "Table"):
        m = re.search(rf"\b{label}\s+\d+\s*:", body)
        if m:
            signals.append(Evidence(page=limitations.page, detail=f"a {label.lower()} caption appears here",
                                    quote=m.group(0)))

    words = len(body.split())
    if not signals:
        return ctx.ok("limitations_scope", "Limitations content",
                      f"The Limitations section reads as discussion ({words} words) with no signals of new "
                      "methods, analyses or results.",
                      category="structure", cfp_key="limitations_scope")
    return ctx.warn(
        "limitations_scope", "Limitations content",
        f"The Limitations section ({words} words) contains {len(signals)} signal(s) that may indicate new "
        "methods, analyses or results, which the CFP does not permit here. Review manually.",
        category="structure", evidence=signals[:6],
        confidence="low — these are lexical signals, not a judgement about your content",
        cfp_key="limitations_scope",
    )


@register("references_position", "References placement", module=MODULE, category="structure", order=34)
def check_references_position(ctx: CheckContext) -> Finding:
    """References should follow the main content and any Limitations section."""
    references = _headings(ctx)["references"]
    if references is None:
        return ctx.warn("references_position", "References placement",
                        "No References or Bibliography heading was detected.", category="structure",
                        confidence="low — heading detection may have missed it",
                        remedy="Verify the bibliography heading is present and formatted by the style file.")
    return ctx.ok("references_position", "References placement",
                  f"References begin on page {references.page} and are excluded from the content page limit.",
                  category="structure",
                  evidence=[Evidence(page=references.page, detail=f"heading {references.text!r}")])


@register("appendix_position", "Appendix placement", module=MODULE, category="structure", order=35)
def check_appendix_position(ctx: CheckContext) -> Finding | None:
    """Appendices must appear after the references."""
    if not ctx.conf("structure.appendix_must_follow_references", True):
        return None
    found = _headings(ctx)
    appendix, references = found["appendix"], found["references"]
    if appendix is None:
        return ctx.ok("appendix_position", "Appendix placement",
                      "No appendix detected; nothing to place.", category="structure")
    if references is None:
        return ctx.warn("appendix_position", "Appendix placement",
                        f"An appendix was found on page {appendix.page} but the references heading was not, "
                        "so their order could not be confirmed.",
                        category="structure", confidence="low")

    evidence = [Evidence(page=appendix.page, detail=f"appendix starts on page {appendix.page}"),
                Evidence(page=references.page, detail=f"references start on page {references.page}")]
    if appendix.order < references.order:
        return ctx.error(
            "appendix_position", "Appendix placement",
            f"The appendix (page {appendix.page}) appears before the references (page {references.page}). "
            "Appendices must come after the references.",
            category="structure", evidence=evidence,
            remedy="Move \\appendix below the bibliography.",
        )
    return ctx.ok("appendix_position", "Appendix placement",
                  f"The appendix (page {appendix.page}) correctly follows the references (page {references.page}).",
                  category="structure", evidence=evidence)


@register("appendix_columns", "Appendix column format", module=MODULE, category="structure", order=36)
def check_appendix_columns(ctx: CheckContext) -> Finding | None:
    """Appendix pages must keep the two-column format, barring approved exceptions."""
    if not ctx.conf("structure.appendix_must_be_two_column", True):
        return None
    appendix = _headings(ctx)["appendix"]
    if appendix is None:
        return None

    threshold = float(ctx.conf("columns.two_column_confidence", 0.6))
    if len(ctx.doc.column_bands) < 2:
        return ctx.skip("appendix_columns", "Appendix column format",
                        "The document's column model could not be established.", category="structure")

    offenders: list[Evidence] = []
    checked = 0
    for page in ctx.doc.pages[appendix.page - 1:]:
        fit = ctx.doc.column_fit(page)
        if fit <= 0.0:
            continue  # a page with no body text (e.g. a full-page figure)
        checked += 1
        if fit < threshold:
            offenders.append(
                Evidence(page=page.number, detail="text does not fit the two-column bands",
                         measured=fit * 100, expected=f">= {threshold * 100:.0f}% of characters in-column")
            )

    if offenders:
        pages = ", ".join(str(e.page) for e in offenders[:10])
        return ctx.warn(
            "appendix_columns", "Appendix column format",
            f"{len(offenders)} of {checked} appendix page(s) do not look double-column ({pages}). "
            "Incorrect appendix formatting is a desk-rejection condition, though full-width tables and "
            "figures are approved exceptions.",
            category="structure", evidence=offenders[:10],
            remedy="Keep appendix prose in two columns; use table*/figure* only where a wide float is needed.",
            confidence="medium — wide floats legitimately span both columns",
        )
    return ctx.ok("appendix_columns", "Appendix column format",
                  f"All {checked} appendix page(s) keep the two-column format.", category="structure")
