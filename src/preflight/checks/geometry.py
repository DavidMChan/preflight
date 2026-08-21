"""Page-geometry checks: paper size, margins, and column layout.

Reusable by any two-column, fixed-paper-size venue; all numbers come from the
profile rather than this file.
"""

from __future__ import annotations

import re

from ..context import CheckContext
from ..document import Line, PageInfo
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.geometry"


@register("paper_size", "Paper size", module=MODULE, category="format", order=10)
def check_paper_size(ctx: CheckContext) -> Finding:
    """Compare every page's MediaBox against the style's required dimensions."""
    want_w = float(ctx.conf("geometry.page_width_pt", 595.0))
    want_h = float(ctx.conf("geometry.page_height_pt", 842.0))
    tol = float(ctx.conf("geometry.page_size_tolerance_pt", 3.0))
    label = ctx.conf("geometry.paper_size_label", "A4")

    bad: list[Evidence] = []
    for page in ctx.doc.pages:
        if abs(page.width - want_w) > tol or abs(page.height - want_h) > tol:
            bad.append(
                Evidence(
                    page=page.number,
                    detail=f"{page.width:.1f} x {page.height:.1f} pt",
                    expected=f"{want_w:.0f} x {want_h:.0f} pt ({label})",
                )
            )

    if bad:
        return ctx.error(
            "paper_size",
            "Paper size",
            f"{len(bad)} page(s) are not {label}. Paper-size violations are a "
            "desk-rejection condition and can be rejected without review.",
            category="format",
            evidence=bad[:8],
            remedy=f"Rebuild the PDF at {label} ({want_w:.0f} x {want_h:.0f} pt). "
            "In pdflatex this usually means passing the right paper option to the class or geometry package.",
        )
    return ctx.ok(
        "paper_size",
        "Paper size",
        f"All {ctx.doc.page_count} pages are {label} ({want_w:.0f} x {want_h:.0f} pt).",
        category="format",
    )


def _line_number_re(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


def _is_margin_exempt(line: Line, pattern: re.Pattern[str]) -> bool:
    """Line numbers from the anonymous template legitimately sit in the margin."""
    return bool(pattern.match(line.text.strip()))


@register("margins", "Margins", module=MODULE, category="format", order=11)
def check_margins(ctx: CheckContext) -> Finding:
    """Measure the extreme text/image boxes on each page against the style margins."""
    left = float(ctx.conf("geometry.margin_left_pt", 71.0))
    right = float(ctx.conf("geometry.margin_right_pt", 71.0))
    top = float(ctx.conf("geometry.margin_top_pt", 57.0))
    bottom = float(ctx.conf("geometry.margin_bottom_pt", 62.0))
    tl = float(ctx.conf("geometry.tolerance_left_pt", 2.0))
    tr = float(ctx.conf("geometry.tolerance_right_pt", 4.5))
    tt = float(ctx.conf("geometry.tolerance_top_pt", 1.0))
    tb = float(ctx.conf("geometry.tolerance_bottom_pt", 1.0))
    min_violation = float(ctx.conf("geometry.min_violation_pt", 1.0))
    cap = int(ctx.conf("geometry.max_reported_violations", 12))

    pattern = _line_number_re(ctx)
    violations: list[Evidence] = []

    for page in ctx.doc.pages:
        limit_left = left - tl
        limit_right = page.width - (right - tr)
        limit_top = top - tt
        limit_bottom = page.height - (bottom - tb)

        for line in page.lines:
            if not line.text.strip() or _is_margin_exempt(line, pattern):
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
        return ctx.error(
            "margins",
            "Margins",
            f"{len(violations)} margin violation(s) across {len(pages)} page(s): "
            f"{', '.join(str(p) for p in pages[:10])}"
            f"{'...' if len(pages) > 10 else ''}. Margin violations can be rejected without review.",
            category="format",
            evidence=violations[:cap],
            remedy="Usually caused by wide tables, unbroken URLs, or oversized figures. "
            "Wrap long URLs, shrink or rotate wide tables, and let the ACL style set the text block.",
            confidence="high — measured from text bounding boxes; images with white backgrounds may be spurious",
        )
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
            remedy="Build with the official ACL LaTeX class or Word template.",
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
