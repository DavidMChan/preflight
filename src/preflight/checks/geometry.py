"""Page-geometry checks: paper size, margins, and column layout.

Reusable by any two-column, fixed-paper-size venue; all numbers come from the
profile rather than this file.
"""

from __future__ import annotations

import re
from typing import Any

from ..context import CheckContext
from ..document import Line, PageInfo
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.geometry"


def _paper_formats(ctx: CheckContext) -> list[dict[str, Any]]:
    """Every paper size the venue accepts, primary first, each with its own text block.

    Most venues accept one size. ICASSP accepts US Letter or A4, and on A4 the
    same 178 x 229 mm block keeps its top-left position, so the right and bottom
    margins differ with the sheet.
    """
    primary: dict[str, Any] = {
        "page_width_pt": float(ctx.conf("geometry.page_width_pt", 595.0)),
        "page_height_pt": float(ctx.conf("geometry.page_height_pt", 842.0)),
        "paper_size_label": str(ctx.conf("geometry.paper_size_label", "A4")),
        "margin_left_pt": float(ctx.conf("geometry.margin_left_pt", 71.0)),
        "margin_right_pt": float(ctx.conf("geometry.margin_right_pt", 71.0)),
        "margin_top_pt": float(ctx.conf("geometry.margin_top_pt", 57.0)),
        "margin_bottom_pt": float(ctx.conf("geometry.margin_bottom_pt", 62.0)),
    }
    formats = [primary]
    for alt in (ctx.conf("geometry.alternate_page_sizes", {}) or {}).values():
        if isinstance(alt, dict):
            formats.append({**primary, **{k: type(primary[k])(v) for k, v in alt.items() if k in primary}})
    return formats


def _format_of(ctx: CheckContext, page: PageInfo) -> dict[str, Any] | None:
    tol = float(ctx.conf("geometry.page_size_tolerance_pt", 3.0))
    return next(
        (f for f in _paper_formats(ctx)
         if abs(page.width - f["page_width_pt"]) <= tol and abs(page.height - f["page_height_pt"]) <= tol),
        None,
    )


def page_format(ctx: CheckContext, page: PageInfo) -> dict[str, Any]:
    """The accepted size a page matches, or the primary one, with its text block."""
    return _format_of(ctx, page) or _paper_formats(ctx)[0]


def _describe(fmt: dict[str, Any]) -> str:
    return f"{fmt['page_width_pt']:.0f} x {fmt['page_height_pt']:.0f} pt ({fmt['paper_size_label']})"


@register("paper_size", "Paper size", module=MODULE, category="format", order=10)
def check_paper_size(ctx: CheckContext) -> Finding:
    """Compare every page's MediaBox against the style's required dimensions."""
    formats = _paper_formats(ctx)
    label = " or ".join(f["paper_size_label"] for f in formats)
    expected = " or ".join(_describe(f) for f in formats)

    bad: list[Evidence] = []
    used: list[str] = []
    for page in ctx.doc.pages:
        fmt = _format_of(ctx, page)
        if fmt is None:
            bad.append(Evidence(page=page.number, detail=f"{page.width:.1f} x {page.height:.1f} pt",
                                expected=expected))
        elif fmt["paper_size_label"] not in used:
            used.append(fmt["paper_size_label"])

    if bad:
        return ctx.error(
            "paper_size",
            "Paper size",
            f"{len(bad)} page(s) are not {label}. Paper-size violations are a "
            "desk-rejection condition and can be rejected without review.",
            category="format",
            evidence=bad[:8],
            remedy=f"Rebuild the PDF at {expected}. "
            "In pdflatex this usually means passing the right paper option to the class or geometry package.",
        )
    fmt = next(f for f in formats if f["paper_size_label"] == used[0])
    accepted = f", one of the accepted sizes ({label})" if len(formats) > 1 else ""
    return ctx.ok(
        "paper_size",
        "Paper size",
        f"All {ctx.doc.page_count} pages are {' and '.join(used)} "
        f"({fmt['page_width_pt']:.0f} x {fmt['page_height_pt']:.0f} pt){accepted}.",
        category="format",
    )


def _line_number_re(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


def _margin_furniture(ctx: CheckContext) -> list[re.Pattern[str]]:
    """Regexes for lines that belong in the margin because nobody chose to put them there.

    A style file's own preprint footer sits below the text block by design, and
    a submission system stamps its confidentiality banner across the foot of
    every page after the author has stopped editing. Neither is something an
    author can fix, so reporting them buries the wide table that they can.

    Matching on the text rather than on the band it occupies is deliberate:
    anything *else* that turns up down there -- a stray line pushed off the
    block, or text planted in the banner's place to catch an automated
    reviewer -- is still worth reporting, and an exempt band would hide it.
    """
    spec = ctx.conf("geometry.margin_furniture", {}) or {}
    return [re.compile(str(pattern)) for pattern in spec.values()]


def _is_margin_exempt(line: Line, pattern: re.Pattern[str],
                      furniture: list[re.Pattern[str]]) -> bool:
    """Line numbers from the anonymous template legitimately sit in the margin."""
    text = " ".join(line.text.split())
    return bool(pattern.match(text)) or any(f.search(text) for f in furniture)


def _systematic_side(violations: list[Evidence]) -> str | None:
    """The side whose overshoot repeats at one position on three or more pages.

    A float that overflows does so once, by its own amount. The same line
    position over and over is the text block: the margins themselves differ
    from the template's.
    """
    seen: dict[tuple[str, int], set[int]] = {}
    for e in violations:
        if e.page is None or e.measured is None or "text bleeds" not in e.detail:
            continue
        side = e.detail.split()[-2].lower()
        seen.setdefault((side, round(e.measured)), set()).add(e.page)
    for (side, _), pages in seen.items():
        if len(pages) >= 3:
            return side
    return None


@register("margins", "Margins", module=MODULE, category="format", order=11)
def check_margins(ctx: CheckContext) -> Finding:
    """Measure the extreme text/image boxes on each page against the style margins."""
    tl = float(ctx.conf("geometry.tolerance_left_pt", 2.0))
    tr = float(ctx.conf("geometry.tolerance_right_pt", 4.5))
    tt = float(ctx.conf("geometry.tolerance_top_pt", 1.0))
    tb = float(ctx.conf("geometry.tolerance_bottom_pt", 1.0))
    min_violation = float(ctx.conf("geometry.min_violation_pt", 1.0))
    cap = int(ctx.conf("geometry.max_reported_violations", 12))

    pattern = _line_number_re(ctx)
    furniture = _margin_furniture(ctx)
    violations: list[Evidence] = []

    for page in ctx.doc.pages:
        # A page at an unaccepted size is the paper-size check's to report;
        # measure it against the primary text block.
        fmt = page_format(ctx, page)
        left, right = fmt["margin_left_pt"], fmt["margin_right_pt"]
        top, bottom = fmt["margin_top_pt"], fmt["margin_bottom_pt"]
        limit_left = left - tl
        limit_right = page.width - (right - tr)
        limit_top = top - tt
        limit_bottom = page.height - (bottom - tb)

        for line in page.lines:
            if not line.text.strip() or _is_margin_exempt(line, pattern, furniture):
                continue
            if all(s.invisible for s in line.spans):
                continue  # reported by the hidden-text check instead
            x0, y0, x1, y1 = line.bbox
            quote = " ".join(line.text.split())[:60]
            if limit_left - x0 > min_violation:
                violations.append(Evidence(page.number, "text bleeds into the LEFT margin",
                                           quote, measured=x0, expected=f">= {limit_left:.1f} pt", bbox=line.bbox))
            if x1 - limit_right > min_violation:
                violations.append(Evidence(page.number, "text bleeds into the RIGHT margin",
                                           quote, measured=x1, expected=f"<= {limit_right:.1f} pt", bbox=line.bbox))
            if limit_top - y0 > min_violation:
                violations.append(Evidence(page.number, "text bleeds into the TOP margin",
                                           quote, measured=y0, expected=f">= {limit_top:.1f} pt", bbox=line.bbox))
            if y1 - limit_bottom > min_violation:
                violations.append(Evidence(page.number, "text bleeds into the BOTTOM margin",
                                           quote, measured=y1, expected=f"<= {limit_bottom:.1f} pt", bbox=line.bbox))

        for bbox in page.images:
            x0, y0, x1, y1 = bbox
            if limit_left - x0 > min_violation:
                violations.append(Evidence(page.number, "an image bleeds into the LEFT margin",
                                           measured=x0, expected=f">= {limit_left:.1f} pt", bbox=bbox))
            if x1 - limit_right > min_violation:
                violations.append(Evidence(page.number, "an image bleeds into the RIGHT margin",
                                           measured=x1, expected=f"<= {limit_right:.1f} pt", bbox=bbox))
            if limit_top - y0 > min_violation:
                violations.append(Evidence(page.number, "an image bleeds into the TOP margin",
                                           measured=y0, expected=f">= {limit_top:.1f} pt", bbox=bbox))

    if violations:
        pages = sorted({e.page for e in violations if e.page})
        systematic = _systematic_side(violations)
        note = (
            f" The {systematic} overshoot recurs at the same position on most pages, so it is "
            "the text block itself that sits outside the template's, not a float spilling "
            "over: the paper was built with a different class file or altered margins."
            if systematic else ""
        )
        remedy = (
            "Rebuild with the venue's official class file and its default margins; do not "
            "override the text block."
            if systematic else
            "Usually caused by wide tables, unbroken URLs, or oversized figures. "
            "Wrap long URLs, shrink or rotate wide tables, and let the venue's style file "
            "set the text block."
        )
        return ctx.error(
            "margins",
            "Margins",
            f"{len(violations)} margin violation(s) across {len(pages)} page(s): "
            f"{', '.join(str(p) for p in pages[:10])}"
            f"{'...' if len(pages) > 10 else ''}. Margin violations can be rejected without review."
            + note,
            category="format",
            evidence=violations[:cap],
            remedy=remedy,
            confidence="high — measured from text bounding boxes; images with white backgrounds may be spurious",
        )
    fmt = page_format(ctx, ctx.doc.pages[0])
    left, right = fmt["margin_left_pt"], fmt["margin_right_pt"]
    top, bottom = fmt["margin_top_pt"], fmt["margin_bottom_pt"]
    return ctx.ok(
        "margins",
        "Margins",
        f"No text or images intrude into the margins "
        f"(left/right >= {left:.0f} pt, top >= {top:.0f} pt, bottom {bottom:.0f} pt reserved).",
        category="format",
    )


@register("column_layout", "Column layout", module=MODULE, category="format", order=12)
def check_column_layout(ctx: CheckContext) -> Finding:
    """Confirm the body of the paper is set in the expected number of columns."""
    expected = int(ctx.conf("columns.expected_columns", 2))
    min_gap = float(ctx.conf("columns.min_gap_pt", 8.0))
    bands = ctx.doc.column_bands

    if expected < 2:
        return _check_single_column(ctx)

    if len(bands) < expected:
        return ctx.error(
            "column_layout",
            "Two-column layout",
            f"Detected {len(bands)} text column(s); the required style is {expected}-column. "
            "This usually means the paper was not built with the official style file.",
            category="format",
            evidence=[Evidence(detail=f"column bands: {[(round(a, 1), round(b, 1)) for a, b in bands]}")],
            remedy="Build with the venue's official LaTeX class or Word template.",
            cfp_key="paper_size",
        )

    gap = bands[1][0] - bands[0][1]
    evidence = [
        Evidence(
            detail=f"column 1 x=[{bands[0][0]:.1f}, {bands[0][1]:.1f}], "
            f"column 2 x=[{bands[1][0]:.1f}, {bands[1][1]:.1f}]",
            measured=gap,
            expected=f">= {min_gap:.1f} pt gutter",
        )
    ]
    if gap < min_gap:
        return ctx.warn(
            "column_layout",
            "Two-column layout",
            f"The gutter between columns measures {gap:.1f} pt, narrower than the "
            f"{min_gap:.1f} pt the style implies. Check that \\columnsep was not overridden.",
            category="format",
            evidence=evidence,
            cfp_key="paper_size",
        )
    return ctx.ok(
        "column_layout",
        "Two-column layout",
        f"Body text is set in {expected} columns with a {gap:.1f} pt gutter.",
        category="format",
        evidence=evidence,
    )


def _check_single_column(ctx: CheckContext) -> Finding:
    """Single-column venues: what matters is that the body is NOT columnised.

    The band model happily finds two "columns" in a wide notation table or a
    two-block appendix, so the gutter is meaningless here; measure instead how
    much of the body text actually sits inside those bands.
    """
    threshold = float(ctx.conf("columns.two_column_confidence", 0.6))
    pages = [p for p in ctx.doc.pages if any(ln.text.strip() for ln in p.lines)]
    columnised = [p for p in pages if page_is_two_column(ctx, p)[0]]

    if pages and len(columnised) > len(pages) / 2:
        return ctx.error(
            "column_layout",
            "Column layout",
            f"{len(columnised)} of {len(pages)} pages read as two columns; the required style is "
            "single-column. This usually means the paper was not built with the official style file.",
            category="format",
            evidence=[
                Evidence(page=p.number, detail="page reads as two columns",
                         measured=page_is_two_column(ctx, p)[1], expected=f"< {threshold:.2f} column fit")
                for p in columnised[:8]
            ],
            remedy="Build with the official single-column style file rather than a two-column class.",
            cfp_key="paper_size",
        )
    return ctx.ok(
        "column_layout",
        "Column layout",
        f"Body text is set in a single column across {len(pages)} page(s).",
        category="format",
    )


def page_is_two_column(ctx: CheckContext, page: PageInfo) -> tuple[bool, float]:
    """Helper shared with the appendix check."""
    threshold = float(ctx.conf("columns.two_column_confidence", 0.6))
    fit = ctx.doc.column_fit(page)
    return (fit >= threshold, fit)
