"""Editing-markup checks: leftover PDF annotations, highlights, and tracked changes.

Editing markup left in a submitted PDF is embarrassing and occasionally discloses
identity (an annotation's author name, a co-author's inline comment). None of this
is a documented desk-rejection condition on its own, so every finding here is a
warning, never an error -- a venue that wants to treat it harder can promote it via
the profile's `severity:` map.
"""

from __future__ import annotations

import colorsys

from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.markup"

# Annotation subtypes that carry reviewer/author commentary rather than being
# structural (Link, Widget, FileAttachment icons are not commentary).
_COMMENT_SUBTYPES = {
    "Text", "FreeText", "Popup", "Caret", "Ink",
    "Square", "Circle", "Polygon", "StrikeOut", "Underline", "Squiggly",
}
_REVISION_SUBTYPES = {"StrikeOut", "Insert"}


def _iter_annots(ctx: CheckContext):
    """Yield (page_number, subtype, author, content) for every annotation in the doc."""
    for i in range(ctx.doc.doc.page_count):
        page = ctx.doc.doc[i]
        try:
            annots = page.annots()
        except Exception:  # pragma: no cover - defensive, mirrors document.py
            annots = None
        if not annots:
            continue
        for annot in annots:
            try:
                subtype = annot.type[1]
                info = annot.info or {}
                author = (info.get("title") or "").strip()
                content = (info.get("content") or "").strip()
            except Exception:  # pragma: no cover - malformed annot dict
                continue
            yield (i + 1, subtype, author, content)


@register("review_comments", "Review comments", module=MODULE, category="markup", order=60)
def check_review_comments(ctx: CheckContext) -> Finding:
    """Flag PDF comment annotations (sticky notes, freetext, markup) left in the file."""
    cap = int(ctx.conf("markup.max_reported_comments", 10))

    found: list[Evidence] = []
    has_author = False
    for page_no, subtype, author, content in _iter_annots(ctx):
        if subtype not in _COMMENT_SUBTYPES:
            continue
        if author:
            has_author = True
        bits = []
        if author:
            bits.append(f"author {author!r}")
        if content:
            bits.append(f"content {content!r}")
        detail = f"{subtype} annotation" + (f" ({'; '.join(bits)})" if bits else "")
        found.append(Evidence(page=page_no, detail=detail, quote=content or None))

    if not found:
        return ctx.ok(
            "review_comments",
            "Review comments",
            "No PDF comment annotations (sticky notes, freetext, markup shapes) found in the file.",
            category="markup",
        )

    author_note = (
        " At least one annotation carries an author name in its 'title' field, which is itself "
        "an anonymity leak independent of the comment's content."
        if has_author else ""
    )
    return ctx.warn(
        "review_comments",
        "Review comments",
        f"{len(found)} PDF comment annotation(s) are still embedded in the file across "
        f"{len({e.page for e in found})} page(s).{author_note}",
        category="markup",
        evidence=found[:cap],
        remedy="Flatten or remove all annotations before submitting (in Acrobat: Comment > "
        "flatten/delete; in most PDF producers, export or 'print to PDF' instead of saving the "
        "annotated file directly).",
        confidence="high — these are annotation objects in the PDF, not rendered content",
    )


def _rgb_from_fill(fill: object) -> tuple[float, float, float] | None:
    if not fill:
        return None
    try:
        r, g, b = fill  # type: ignore[misc]
        return (float(r), float(g), float(b))
    except Exception:
        return None


def _is_highlight_color(rgb: tuple[float, float, float]) -> bool:
    """Strong, light highlight colours (yellow/green/cyan/pink) -- not zebra grey."""
    r, g, b = rgb
    _hue, lightness, sat = colorsys.rgb_to_hls(r, g, b)
    if sat < 0.45:
        return False
    if not (0.5 <= lightness <= 0.92):
        return False
    return True


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _line_height_for(ctx: CheckContext) -> float:
    size = ctx.doc.body_font_size or 11.0
    return size * 1.6


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0:
        return 0.0
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _find_colorbox_hits(ctx: CheckContext, max_height: float, min_area: float) -> list[Evidence]:
    hits: list[Evidence] = []
    for page_index, page in enumerate(ctx.doc.pages):
        try:
            drawings = ctx.doc.doc[page_index].get_drawings()
        except Exception:  # pragma: no cover
            continue
        filled = [
            (tuple(float(v) for v in d["rect"]), _rgb_from_fill(d.get("fill")))
            for d in drawings
            if d.get("type") in ("f", "fs") and d.get("rect") is not None and d.get("fill")
        ]
        # How many other filled shapes share (roughly) this colour, anywhere on the
        # page: a genuine accidental highlight is one or two ad-hoc marks, while a
        # repeated colour across many rects is a table's zebra/header shading or a
        # bar chart's data series -- a deliberate, systematic use of colour.
        def _color_repeat_count(rgb: tuple[float, float, float], shapes=filled) -> int:
            return sum(
                1 for _, other in shapes
                if other is not None
                and abs(other[0] - rgb[0]) < 0.03 and abs(other[1] - rgb[1]) < 0.03 and abs(other[2] - rgb[2]) < 0.03
            )

        for bbox, rgb in filled:
            if rgb is None or not _is_highlight_color(rgb):
                continue
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if height <= 0 or height > max_height or width <= 0:
                continue
            area = _bbox_area(bbox)
            if area < min_area:
                continue
            if _color_repeat_count(rgb) >= 3:
                continue  # systematic use of this colour: table shading or a chart series
            # A shape stacked or nested under/over another differently-coloured
            # shape at (nearly) the same spot is a rendered border or drop-shadow
            # -- the signature of a diagram node, not a flat highlight rectangle.
            if any(
                other_rgb is not None and other_rgb != rgb and _iou(bbox, other_bbox) > 0.5
                for other_bbox, other_rgb in filled
                if other_bbox != bbox
            ):
                continue
            # Must actually have text sitting on top of it, and be small (roughly
            # one line): this rules out zebra-striped tables, header-row fills,
            # figure backgrounds and full-width rules, all of which either span
            # a whole table/column width with no single line of text, or are
            # much taller than one text line.
            covered_lines = [
                ln for ln in page.lines
                if ln.text.strip()
                and ln.bbox[0] >= bbox[0] - 2.0 and ln.bbox[2] <= bbox[2] + 2.0
                and ln.bbox[1] >= bbox[1] - 2.0 and ln.bbox[3] <= bbox[3] + 2.0
            ]
            if not covered_lines:
                continue
            quote = " ".join(covered_lines[0].text.split())[:80]
            hits.append(
                Evidence(
                    page=page_index + 1,
                    detail=f"filled rect (rgb {tuple(round(c, 2) for c in rgb)}) behind text",
                    quote=quote,
                    measured=height,
                    expected=f"<= {max_height:.1f} pt tall (one text line)",
                    bbox=bbox,
                )
            )
    return hits


@register("highlighted_text", "Highlighted text", module=MODULE, category="markup", order=61)
def check_highlighted_text(ctx: CheckContext) -> Finding:
    """Flag Highlight annotations and colour-box-behind-text (colorbox / Word highlight)."""
    cap = int(ctx.conf("markup.max_reported_highlights", 10))
    max_height = float(ctx.conf("markup.highlight_max_height_pt", _line_height_for(ctx)))
    min_area = float(ctx.conf("markup.highlight_min_area_pt2", 20.0))

    annot_hits: list[Evidence] = []
    for page_no, subtype, author, content in _iter_annots(ctx):
        if subtype != "Highlight":
            continue
        detail = "Highlight annotation" + (f" (author {author!r})" if author else "")
        annot_hits.append(Evidence(page=page_no, detail=detail, quote=content or None))

    colorbox_hits = _find_colorbox_hits(ctx, max_height, min_area)

    total = len(annot_hits) + len(colorbox_hits)
    if not total:
        return ctx.ok(
            "highlighted_text",
            "Highlighted text",
            "No highlight annotations and no small saturated colour boxes behind text "
            "(the signature of \\colorbox or Word highlighting).",
            category="markup",
        )

    parts = []
    if annot_hits:
        parts.append(f"{len(annot_hits)} Highlight annotation(s)")
    if colorbox_hits:
        parts.append(f"{len(colorbox_hits)} coloured-background span(s) that look like manual highlighting")
    return ctx.warn(
        "highlighted_text",
        "Highlighted text",
        f"Found {' and '.join(parts)}. Highlighting is usually a private editing note left in by "
        "accident, not intended content.",
        category="markup",
        evidence=(annot_hits + colorbox_hits)[:cap],
        remedy="Remove Highlight annotations and any \\colorbox/\\hl markup before submitting; "
        "check table zebra-striping and header fills were not mistakenly flagged before assuming "
        "this is real.",
        confidence="medium — a small saturated fill behind one line of text is inferred from "
        "geometry and colour, not from a semantic 'this is a highlight' marker",
    )


@register("revision_markup", "Revision markup", module=MODULE, category="markup", order=62)
def check_revision_markup(ctx: CheckContext) -> Finding:
    """Flag StrikeOut/Insert annotations, the residue of an 'All Markup' Word export."""
    cap = int(ctx.conf("markup.max_reported_revisions", 10))

    found: list[Evidence] = []
    for page_no, subtype, author, content in _iter_annots(ctx):
        if subtype not in _REVISION_SUBTYPES:
            continue
        detail = f"{subtype} annotation" + (f" (author {author!r})" if author else "")
        found.append(Evidence(page=page_no, detail=detail, quote=content or None))

    weak_evidence_note = (
        "Note this is weak evidence either way: tracked changes usually do not survive conversion "
        "to PDF at all -- a Word document with 'All Markup' still turned on generally exports as "
        "clean final text with no trace, and a PASS here means only 'no markup is visible in this "
        "PDF', not 'the source file has no unresolved tracked changes'."
    )

    if not found:
        return ctx.ok(
            "revision_markup",
            "Revision markup",
            f"No StrikeOut/Insert revision annotations found in the file. {weak_evidence_note}",
            category="markup",
        )

    return ctx.warn(
        "revision_markup",
        "Revision markup",
        f"{len(found)} revision-marking annotation(s) (StrikeOut/Insert) found, consistent with "
        f"Word's 'All Markup' export or LaTeX changes.sty left switched on. {weak_evidence_note}",
        category="markup",
        evidence=found[:cap],
        remedy="Accept or reject all tracked changes and re-export ('Final' view, not 'All Markup'); "
        "for LaTeX, run the \\final variant of changes.sty or strip \\added/\\deleted markup.",
        confidence="high on presence when found; a clean result is not proof the source is clean",
    )
