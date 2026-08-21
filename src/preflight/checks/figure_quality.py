"""Figure-quality checks: raster resolution and colour readability.

These are measurements, not desk-rejection rules, so every finding here is a
WARNING regardless of how bad the number looks -- a venue that wants one of
them enforced harder can promote it through the profile's severity map.
"""

from __future__ import annotations

import colorsys
from collections import Counter
from dataclasses import dataclass

from .. import analysis
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.figure_quality"

# Deuteranopia (red-green colour-blindness) simulation matrix, applied directly
# to sRGB. This is the widely used simplified Machado-style matrix; it is an
# approximation of one form of one kind of colour-vision deficiency, not a
# calibrated accessibility audit.
_DEUTERANOPIA_MATRIX = (
    (0.625, 0.375, 0.0),
    (0.7, 0.3, 0.0),
    (0.0, 0.3, 0.7),
)


def _luminance(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02x}" for c in rgb)


def _apply_matrix(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    r, g, b = rgb
    return tuple(
        max(0.0, min(1.0, m[0] * r + m[1] * g + m[2] * b))  # type: ignore[misc]
        for m in _DEUTERANOPIA_MATRIX
    )


# ---------------------------------------------------------------------------
# 1. Raster resolution
# ---------------------------------------------------------------------------


@register("figure_resolution", "Figure resolution", module=MODULE, category="figures", order=60)
def check_figure_resolution(ctx: CheckContext) -> Finding:
    """Flag raster images placed at low effective DPI.

    Only raster images appear in ``get_image_info`` at all -- a vector figure
    (a PDF-native plot, most matplotlib/TikZ output) has no pixels to measure
    and is perfect at any size, so it is automatically exempt rather than
    something this check has to special-case.
    """
    min_dpi = float(ctx.conf("figure_quality.min_dpi", 150.0))
    min_area = float(ctx.conf("figure_quality.min_icon_area_pt2", 1600.0))  # ~40x40pt
    max_repeats = int(ctx.conf("figure_quality.max_repeat_occurrences", 3))
    cap = int(ctx.conf("figure_quality.max_reported_violations", 10))

    # First pass: count how often each distinct image (by content digest)
    # recurs across the document. A logo or watermark stamped on every page,
    # or a glyph repeated dozens of times in one figure, is not "a figure" --
    # flagging it produces exactly the noise this check must avoid.
    digest_counts: Counter[bytes] = Counter()
    all_infos: list[tuple[int, dict]] = []
    for page in ctx.doc.pages:
        try:
            infos = ctx.doc.doc[page.number - 1].get_image_info(xrefs=True)
        except Exception:
            infos = []
        for info in infos:
            digest_counts[info.get("digest", b"")] += 1
            all_infos.append((page.number, info))

    offenders: list[Evidence] = []
    considered = 0
    skipped_repeated = 0
    skipped_tiny = 0
    for page_no, info in all_infos:
        bbox = info.get("bbox")
        w_px, h_px = info.get("width", 0), info.get("height", 0)
        if not bbox or w_px <= 0 or h_px <= 0:
            continue
        x0, y0, x1, y1 = bbox
        w_pt, h_pt = x1 - x0, y1 - y0
        if w_pt <= 0 or h_pt <= 0:
            continue
        if w_pt * h_pt < min_area:
            skipped_tiny += 1
            continue
        digest = info.get("digest", b"")
        if digest and digest_counts[digest] > max_repeats:
            skipped_repeated += 1
            continue
        considered += 1
        dpi_x = w_px / (w_pt / 72.0)
        dpi_y = h_px / (h_pt / 72.0)
        eff_dpi = min(dpi_x, dpi_y)
        if eff_dpi < min_dpi:
            offenders.append(
                Evidence(
                    page=page_no,
                    detail=f"raster image {w_px}x{h_px}px placed at {w_pt:.0f}x{h_pt:.0f}pt",
                    measured=eff_dpi,
                    expected=f">= {min_dpi:.0f} DPI",
                    bbox=bbox,
                )
            )

    note = (
        f" ({skipped_repeated} repeated/decorative and {skipped_tiny} sub-threshold images "
        "were excluded; vector figures are exempt by construction)"
        if (skipped_repeated or skipped_tiny)
        else " (vector figures are exempt by construction)"
    )

    if not offenders:
        return ctx.ok(
            "figure_resolution",
            "Figure resolution",
            f"All {considered} raster image(s) large enough to be a figure meet the "
            f"{min_dpi:.0f} DPI minimum.{note}",
            category="figures",
        )

    pages = sorted({e.page for e in offenders if e.page})
    worst = min(e.measured or 0.0 for e in offenders)
    return ctx.warn(
        "figure_resolution",
        "Figure resolution",
        f"{len(offenders)} raster image(s) on page(s) {', '.join(str(p) for p in pages[:10])} "
        f"are placed below {min_dpi:.0f} effective DPI (worst: {worst:.0f} DPI); at print size these "
        f"will read as low-resolution or blurry.{note}",
        category="figures",
        evidence=sorted(offenders, key=lambda e: e.measured or 0.0)[:cap],
        remedy="Re-export the source figure at a higher resolution (or as a vector PDF/SVG) "
        "rather than scaling up a small raster image.",
        confidence="medium — a screenshot or photograph legitimately has finite native resolution",
    )


# ---------------------------------------------------------------------------
# Shared machinery for the two colour checks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _FigureColors:
    caption: analysis.Caption
    bbox: tuple[float, float, float, float]
    colors: list[tuple[tuple[float, float, float], float]]  # (rgb 0..1, share)


def _figure_region(ctx: CheckContext, cap: analysis.Caption) -> tuple[float, float, float, float] | None:
    """Best-effort bounding box for the figure a caption belongs to.

    Finds the caption's own line, then unions every image and drawing on the
    same page that sits above it (captions follow their figure in ACL-style
    papers) and horizontally overlaps it or its column.
    """
    page = ctx.doc.pages[cap.page - 1]
    label_low = cap.label.lower()
    cap_line = None
    for line in page.lines:
        flat = " ".join(line.text.split()).lower()
        if flat.startswith(f"figure {label_low}") or flat.startswith(f"fig. {label_low}") or flat.startswith(
            f"fig {label_low}"
        ):
            cap_line = line
            break
    if cap_line is None:
        return None

    x0, y0, x1, y1 = cap_line.bbox
    boxes: list[tuple[float, float, float, float]] = []
    for bx in list(page.images) + list(page.drawings_bbox):
        bx0, by0, bx1, by1 = bx
        if by1 > y0 + 4.0:  # must sit at or above the caption
            continue
        if bx1 <= x0 - 4.0 or bx0 >= x1 + 4.0:
            continue  # no horizontal overlap with the caption line/column
        boxes.append(bx)
    if not boxes:
        return None
    rx0 = min(b[0] for b in boxes)
    ry0 = min(b[1] for b in boxes)
    rx1 = max(b[2] for b in boxes)
    ry1 = max(b[3] for b in boxes)
    return (rx0, ry0, rx1, ry1)


def _dominant_colors(
    ctx: CheckContext, page_no: int, bbox: tuple[float, float, float, float], dpi: float
) -> list[tuple[tuple[float, float, float], float]]:
    """Saturated dominant colours in a page region, as [(rgb 0..1, share), ...].

    Near-white, near-black and near-gray pixels are dropped before counting so
    axes, gridlines and text do not swamp the histogram; colours are quantised
    to coarse buckets so anti-aliasing does not fragment one visual colour into
    dozens of near-identical ones.
    """
    page = ctx.doc.doc[page_no - 1]
    rect = ctx.doc.doc[page_no - 1].rect
    clip = (
        max(bbox[0], rect.x0),
        max(bbox[1], rect.y0),
        min(bbox[2], rect.x1),
        min(bbox[3], rect.y1),
    )
    if clip[2] - clip[0] < 4 or clip[3] - clip[1] < 4:
        return []
    pix = page.get_pixmap(clip=clip, dpi=int(dpi), colorspace="rgb", alpha=False)
    n = pix.n
    samples = pix.samples
    total = 0
    buckets: Counter[tuple[int, int, int]] = Counter()
    # Quantise to 32-wide buckets (8 levels/channel) -- plenty for "is this
    # visually a distinct colour", cheap to tally.
    step = max(1, (pix.width * pix.height) // 20000)  # cap work on big regions
    for i in range(0, pix.width * pix.height, step):
        off = i * n
        r, g, b = samples[off], samples[off + 1], samples[off + 2]
        mx, mn = max(r, g, b), min(r, g, b)
        if mx > 235 and mn > 210:  # near-white
            continue
        if mx < 35:  # near-black
            continue
        if mx - mn < 22:  # near-gray / low saturation
            continue
        total += 1
        buckets[(r // 32 * 32 + 16, g // 32 * 32 + 16, b // 32 * 32 + 16)] += 1
    if total == 0:
        return []
    min_share = float(ctx.conf("figure_quality.min_color_share", 0.04))
    out = []
    for (r, g, b), count in buckets.most_common(8):
        share = count / total
        if share < min_share:
            continue
        out.append(((r / 255.0, g / 255.0, b / 255.0), share))
    return out


def _figure_colors(ctx: CheckContext) -> list[_FigureColors]:
    cached = ctx.shared.get("figure_quality_colors")
    if isinstance(cached, list):
        return cached

    max_figures = int(ctx.conf("figure_quality.max_figures", 12))
    render_dpi = float(ctx.conf("figure_quality.render_dpi", 72.0))
    out: list[_FigureColors] = []
    for cap in analysis.captions(ctx):
        if cap.kind != "figure":
            continue
        if len(out) >= max_figures:
            break
        bbox = _figure_region(ctx, cap)
        if bbox is None:
            continue
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < 400:  # too small to be a rendered chart
            continue
        colors = _dominant_colors(ctx, cap.page, bbox, render_dpi)
        out.append(_FigureColors(caption=cap, bbox=bbox, colors=colors))

    ctx.shared["figure_quality_colors"] = out
    return out


def _hue_degrees(rgb: tuple[float, float, float]) -> float:
    h, _s, _v = colorsys.rgb_to_hsv(*rgb)
    return h * 360.0


def _hue_distance(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _rgb_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


# ---------------------------------------------------------------------------
# 2. Grayscale printing
# ---------------------------------------------------------------------------


@register("figure_grayscale", "Figures in grayscale", module=MODULE, category="figures", order=61)
def check_figure_grayscale(ctx: CheckContext) -> Finding:
    """Flag figures whose colour-coded series would collapse together in black and white."""
    figures = analysis.captions(ctx)
    figure_count = sum(1 for c in figures if c.kind == "figure")
    if figure_count == 0:
        return ctx.skip("figure_grayscale", "Figures in grayscale", "No figures found in the paper.",
                        category="figures")

    min_hue_sep = float(ctx.conf("figure_quality.min_hue_separation_deg", 30.0))
    min_gray_sep = float(ctx.conf("figure_quality.min_gray_separation", 0.10))

    rendered = _figure_colors(ctx)
    if not rendered:
        return ctx.skip(
            "figure_grayscale", "Figures in grayscale",
            f"Could not isolate a renderable region for any of the {figure_count} figure(s) "
            "(no images/drawings found above their captions).",
            category="figures",
        )

    hits: list[Evidence] = []
    for fig in rendered:
        colors = fig.colors
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                (rgb_a, share_a), (rgb_b, share_b) = colors[i], colors[j]
                hue_sep = _hue_distance(_hue_degrees(rgb_a), _hue_degrees(rgb_b))
                if hue_sep < min_hue_sep:
                    continue  # not "clearly distinct" colours to begin with
                lum_a, lum_b = _luminance(rgb_a), _luminance(rgb_b)
                if abs(lum_a - lum_b) < min_gray_sep:
                    hits.append(
                        Evidence(
                            page=fig.caption.page,
                            detail=f"{fig.caption.name}: {_hex(rgb_a)} (L={lum_a:.2f}, share {share_a:.0%}) "
                            f"vs {_hex(rgb_b)} (L={lum_b:.2f}, share {share_b:.0%}), hue apart {hue_sep:.0f}°",
                            measured=abs(lum_a - lum_b),
                            expected=f">= {min_gray_sep:.2f} luminance separation",
                            bbox=fig.bbox,
                        )
                    )
                    break  # one pair is enough evidence per figure
            else:
                continue
            break

    if not hits:
        return ctx.ok(
            "figure_grayscale", "Figures in grayscale",
            f"Checked {len(rendered)} of {figure_count} figure(s); dominant colours stay separable "
            "by luminance, so the figures should still read in black-and-white printing.",
            category="figures",
        )

    return ctx.warn(
        "figure_grayscale", "Figures in grayscale",
        f"{len(hits)} figure(s) use colours that are visually distinct but nearly identical in "
        "luminance, so a black-and-white printout or a grayscale scan would make their series hard "
        "to tell apart.",
        category="figures",
        evidence=hits[:10],
        remedy="Add distinct line styles/markers/hatching in addition to colour, or pick a palette "
        "with separated luminance (e.g. ColorBrewer's colorblind-safe sets).",
        confidence="medium — measured from dominant rendered colours in the figure's bounding box; "
        "photographs and screenshots can trip this without being a real readability problem",
    )


# ---------------------------------------------------------------------------
# 3. Colour-vision deficiency
# ---------------------------------------------------------------------------


@register("figure_color_vision", "Figures under colour-vision deficiency",
          module=MODULE, category="figures", order=62)
def check_figure_color_vision(ctx: CheckContext) -> Finding:
    """Flag figures where distinct colours become confusable under a deuteranopia simulation.

    This applies one approximate simulation of one common form of red-green
    colour blindness. It is a useful smoke test, not an accessibility audit.
    """
    figures = analysis.captions(ctx)
    figure_count = sum(1 for c in figures if c.kind == "figure")
    if figure_count == 0:
        return ctx.skip("figure_color_vision", "Figures under colour-vision deficiency",
                        "No figures found in the paper.", category="figures")

    min_cvd_dist = float(ctx.conf("figure_quality.min_cvd_distance", 0.12))
    min_pre_dist = float(ctx.conf("figure_quality.min_pretransform_distance", 0.25))

    rendered = _figure_colors(ctx)
    if not rendered:
        return ctx.skip(
            "figure_color_vision", "Figures under colour-vision deficiency",
            f"Could not isolate a renderable region for any of the {figure_count} figure(s).",
            category="figures",
        )

    hits: list[Evidence] = []
    for fig in rendered:
        colors = fig.colors
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                (rgb_a, share_a), (rgb_b, share_b) = colors[i], colors[j]
                pre_dist = _rgb_distance(rgb_a, rgb_b)
                if pre_dist < min_pre_dist:
                    continue  # not clearly distinct before the transform
                sim_a, sim_b = _apply_matrix(rgb_a), _apply_matrix(rgb_b)
                post_dist = _rgb_distance(sim_a, sim_b)
                if post_dist < min_cvd_dist:
                    hits.append(
                        Evidence(
                            page=fig.caption.page,
                            detail=f"{fig.caption.name}: {_hex(rgb_a)} (share {share_a:.0%}) vs "
                            f"{_hex(rgb_b)} (share {share_b:.0%}) — simulated as {_hex(sim_a)} vs "
                            f"{_hex(sim_b)} under deuteranopia",
                            measured=post_dist,
                            expected=f">= {min_cvd_dist:.2f} simulated RGB distance",
                            bbox=fig.bbox,
                        )
                    )
                    break
            else:
                continue
            break

    if not hits:
        return ctx.ok(
            "figure_color_vision", "Figures under colour-vision deficiency",
            f"Checked {len(rendered)} of {figure_count} figure(s) against a deuteranopia simulation; "
            "no distinct colour pairs became confusable. This is an approximation of one form of "
            "colour-vision deficiency, not an accessibility audit.",
            category="figures",
        )

    return ctx.warn(
        "figure_color_vision", "Figures under colour-vision deficiency",
        f"{len(hits)} figure(s) have colour pairs that are distinct in normal vision but converge "
        "under a simulated deuteranopia (red-green colour-blindness) transform, the classic failure "
        "being red/green series encodings. This is an approximation of one form of one kind of "
        "colour-vision deficiency, not a full accessibility audit.",
        category="figures",
        evidence=hits[:10],
        remedy="Avoid red/green as the only distinguishing signal; use a colour-blind-safe palette "
        "and redundant encodings (markers, line styles, direct labels).",
        confidence="medium — one linear simulation of one CVD type on a dominant-colour sample",
    )
