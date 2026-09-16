"""Typography checks: body font size, minimum legible size, and font family."""

from __future__ import annotations

import re
from collections import Counter

from ..context import CheckContext
from ..document import Line
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.fonts"


# Computer Modern and AMS fonts whose name ends in the design size 5, 6 or 7:
# CMR7, CMMI7, CMSY5, MSBM7... These are the script and scriptscript fonts that
# TeX uses for first- and second-level sub/superscripts, and nothing else.
_DEFAULT_SCRIPT_FONT_RE = r"(?:^|\+)(?:CM[A-Z]*|MS[AB]M|EU[A-Z]{2}|RSFS|LASY|LCMSS[A-Z]*)[5-7]$"


def _ignore_pattern(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


def _is_body_flow(ctx: CheckContext, line: Line) -> bool:
    """True if a line is running paragraph text rather than a table cell or float.

    This is the discriminator that decides severity. Shrinking type only buys
    space in running prose, so undersized *paragraph* text is the signal that
    someone squeezed the paper to fit; undersized table and caption text is
    ordinary practice at every venue.
    """
    bands = ctx.doc.column_bands
    if not bands:
        return False
    column_width = bands[0][1] - bands[0][0]
    if column_width <= 0:
        return False
    if not any(abs(line.bbox[0] - b0) < 6.0 for b0, _ in bands):
        return False
    ratio = (line.bbox[2] - line.bbox[0]) / column_width
    # Below 0.6 it is a cell or a short last line; above 1.15 it is a wide float.
    return 0.6 <= ratio <= 1.15


@register("body_font_size", "Body font size", module=MODULE, category="format", order=20)
def check_body_font_size(ctx: CheckContext) -> Finding:
    """The modal font size over the document should match the style's body size."""
    expected = float(ctx.conf("fonts.body_size_pt", 11.0))
    tol = float(ctx.conf("fonts.body_size_tolerance_pt", 0.6))
    measured = ctx.doc.body_font_size

    if measured <= 0:
        return ctx.skip("body_font_size", "Body font size", "No extractable text to measure.", category="format")

    evidence = [Evidence(detail="most common font size across the document",
                         measured=measured, expected=f"{expected:.1f} pt +/- {tol:.1f}")]
    if abs(measured - expected) > tol:
        return ctx.error(
            "body_font_size",
            "Body font size",
            f"Body text is set at {measured:.1f} pt but the style requires {expected:.1f} pt. "
            "Font-size violations can be rejected without review.",
            category="format",
            evidence=evidence,
            remedy="Do not override the class font size (e.g. no \\fontsize, \\small, or shrink-to-fit tricks on body text).",
            cfp_key="font_size",
        )
    return ctx.ok("body_font_size", "Body font size",
                  f"Body text measures {measured:.1f} pt, matching the required {expected:.1f} pt.",
                  category="format", evidence=evidence)


@register("min_font_size", "Minimum font size", module=MODULE, category="format", order=21)
def check_min_font_size(ctx: CheckContext) -> Finding:
    """Flag runs of real text set below the smallest legitimate size."""
    minimum = float(ctx.conf("fonts.min_font_size_pt", 9.0))
    ignore = _ignore_pattern(ctx)
    script_fonts = re.compile(str(ctx.conf("fonts.script_font_regex", _DEFAULT_SCRIPT_FONT_RE)))

    offenders: list[Evidence] = []
    body_offenders: list[Evidence] = []
    total_small = 0
    body_small = 0
    in_artwork = 0
    in_scripts = 0
    for line in ctx.doc.reading_order:
        body_flow = _is_body_flow(ctx, line)
        # Small capitals are cut at 80% of the size the line is set at: an 8pt
        # IEEE table caption ("LIBERO RESULTS") carries 6.4pt glyphs by design,
        # and the reader sees 8pt type. Judge the line at its full-size initial.
        if line.small_caps and line.heading_size >= minimum - 0.15:
            continue
        for span in line.spans:
            text = span.text.strip()
            if len(text) < 3 or ignore.match(text) or span.invisible:
                continue
            if span.size >= minimum - 0.15:
                continue
            # TeX's script and scriptscript fonts exist only for sub- and
            # superscripts; a 7pt "match" under a sigma is maths notation set
            # exactly as the style file intends, not shrunken text.
            if script_fonts.search(span.font):
                in_scripts += len(text)
                continue
            # Labels baked into a figure are a legibility problem, not a font-size
            # violation of the body text; the hidden-text module reports those.
            if ctx.doc.pages[span.page - 1].inside_graphic(span.bbox):
                in_artwork += len(text)
                continue
            total_small += len(text)
            evidence = Evidence(
                page=span.page,
                detail=f"font {span.font}" + (" in running body text" if body_flow else " in a table or float"),
                quote=text, measured=span.size, expected=f">= {minimum:.1f} pt", bbox=span.bbox,
            )
            offenders.append(evidence)
            if body_flow:
                body_small += len(text)
                body_offenders.append(evidence)

    artwork_note = (
        f" ({in_artwork} undersized characters inside figures were ignored here; "
        "see the microscopic-text check)" if in_artwork else ""
    )
    if in_scripts:
        artwork_note += (
            f" ({in_scripts} characters in TeX sub/superscript fonts were ignored: "
            "mathematical notation, not shrunken text)"
        )
    if not offenders:
        return ctx.ok("min_font_size", "Minimum font size",
                      f"No body text below the {minimum:.1f} pt minimum.{artwork_note}",
                      category="format")

    pages = sorted({e.page for e in offenders if e.page})
    page_list = ", ".join(str(p) for p in pages[:10]) + ("..." if len(pages) > 10 else "")
    smallest = min((e.measured or 0.0) for e in offenders)

    # This check never escalates to an error. Separating running prose from
    # table cells and example boxes is not reliable from geometry alone, and the
    # unambiguous version of this violation -- the document's own body size being
    # wrong -- is already an error in `body_font_size`. Reporting a possible desk
    # rejection on a heuristic this soft would be worse than useless to an author.
    if body_small > int(ctx.conf("fonts.body_flow_notice_chars", 200)):
        body_pages = ", ".join(str(p) for p in sorted({e.page for e in body_offenders if e.page})[:10])
        detail = (
            f" {body_small} of those characters sit on full-width lines that start at a column edge "
            f"(pages {body_pages}), which is the shape of running prose rather than a table cell — "
            "worth checking that the text was not shrunk to fit the page limit."
        )
        evidence = sorted(body_offenders, key=lambda e: e.measured or 0)[:10]
        confidence = "medium — full-width undersized lines are usually, but not always, running prose"
    else:
        detail = " All of it sits in tables, captions or floats rather than running prose."
        evidence = sorted(offenders, key=lambda e: e.measured or 0)[:10]
        confidence = "medium — table and caption text is exempted from the body-size rule at most venues"

    return ctx.warn(
        "min_font_size", "Minimum font size",
        f"{len(offenders)} text run(s) ({total_small} characters) are set below {minimum:.1f} pt on "
        f"page(s) {page_list}; the smallest measures {smallest:.1f} pt.{detail}{artwork_note}",
        category="format",
        evidence=evidence,
        remedy="\\small and \\footnotesize in tables are usually accepted, but text this small may be "
        "judged illegible. Never shrink running body text to fit the page limit.",
        confidence=confidence,
        cfp_key="font_size",
    )


@register("font_family", "Font family", module=MODULE, category="format", order=22)
def check_font_family(ctx: CheckContext) -> Finding:
    """The dominant font should be one the official style actually ships."""
    allowed = [str(s) for s in (ctx.conf("fonts.allowed_font_substrings", []) or [])]
    min_share = float(ctx.conf("fonts.dominant_font_min_share", 0.35))
    name, share = ctx.doc.dominant_font

    if not name:
        return ctx.skip("font_family", "Font family", "No extractable text to measure.", category="format")

    evidence = [Evidence(detail=f"dominant font {name!r}", measured=share * 100,
                         expected=f">= {min_share * 100:.0f}% of characters")]

    if share < min_share:
        counts = Counter(s.font for s in ctx.doc.visible_spans)
        top = ", ".join(f"{f} ({c})" for f, c in counts.most_common(4))
        return ctx.warn(
            "font_family", "Font family",
            f"No single font accounts for {min_share:.0%} of the text (top font {name!r} at {share:.0%}). "
            "This often means the document mixes fonts or embeds text as outlines.",
            category="format", evidence=[*evidence, Evidence(detail=f"most used: {top}")],
            cfp_key="font_size",
        )

    if allowed and not any(sub.lower() in name.lower() for sub in allowed):
        return ctx.warn(
            "font_family", "Font family",
            f"The dominant font is {name!r}, which is not one of the families the official style uses. "
            "Verify the paper was built with the official template.",
            category="format", evidence=evidence,
            remedy=f"Expected a family matching one of: {', '.join(allowed[:6])}.",
            confidence="medium — font naming varies between PDF producers",
            cfp_key="font_size",
        )

    return ctx.ok("font_family", "Font family",
                  f"Dominant font {name!r} covers {share:.0%} of characters and matches the expected family.",
                  category="format", evidence=evidence)
