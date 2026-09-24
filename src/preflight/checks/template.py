"""Was the paper built on this year's style file, in the mode the track needs?

Templates that print a running head say both at once: ICLR's reads "Under
review as a conference paper at ICLR 2027" at submission and "Published as ..."
once ``\\iclrfinalcopy`` is switched on. A stale year means last cycle's style
files; the camera-ready wording on a submission means the switch that prints
the author block was left on.

Venues that paginate their own proceedings also want no page numbers on the
submission, and some want the title set in capitals as their style file does.
"""

from __future__ import annotations

import re
from collections import Counter

from ..analysis import title_text
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register
from .geometry import page_format

MODULE = "core.template"


@register("running_head", "Template running head", module=MODULE, category="format", order=15)
def check_running_head(ctx: CheckContext) -> Finding | None:
    """The running head names the current template year and the track's mode."""
    pattern = ctx.conf("template.running_head.pattern", None)
    expected = (ctx.conf("template.running_head.expected", {}) or {}).get(ctx.track.name)
    if not pattern or not expected:
        return None
    family = re.compile(str(pattern))
    found: dict[str, list[int]] = {}
    for page in ctx.doc.pages:
        for line in page.lines:
            text = " ".join(line.text.split())
            if family.fullmatch(text):
                found.setdefault(text, []).append(page.number)

    expected = str(expected)
    if not found:
        return ctx.error(
            "running_head", "Template running head",
            f"No running head of the form {expected!r} was found on any page. The official style "
            "file prints it above the text block, so its absence suggests the template was modified "
            "or not used — which the venue treats as grounds for rejection.",
            category="format", cfp_key="running_head",
            remedy="Build the paper with the current, unmodified style files.",
            confidence="medium — a running head drawn as an image or outlined text would be missed",
        )

    wrong = {text: pages for text, pages in found.items() if text != expected}
    if not wrong:
        pages = found[expected]
        return ctx.ok("running_head", "Template running head",
                      f"Running head {expected!r} found on {len(pages)} page(s).",
                      category="format", cfp_key="running_head")

    text, pages = next(iter(wrong.items()))
    evidence = [Evidence(page=p, detail="running head", quote=t, expected=expected)
                for t, ps in wrong.items() for p in ps[:3]]
    return ctx.error(
        "running_head", "Template running head",
        f"The running head reads {text!r} (page {pages[0]}{' and others' if len(pages) > 1 else ''}), "
        f"but the {ctx.track.name} track expects {expected!r}. "
        "A different year means an earlier cycle's style files; the wrong review status means the "
        "final-copy switch is set incorrectly for this track (at submission it also prints the "
        "author names).",
        category="format", evidence=evidence, cfp_key="running_head",
        remedy="Use the current style files and set the final-copy switch to match the track.",
    )


# "3", "- 3 -", "Page 3", "Page 3 of 5": a folio, and nothing else on the line.
_FOLIO_RE = re.compile(r"(?i)(?:page\s+)?[-–—]?\s*(\d{1,3})\s*[-–—]?(?:\s+of\s+\d{1,3})?")


@register("page_numbers", "Page numbers", module=MODULE, category="format", order=16)
def check_page_numbers(ctx: CheckContext) -> Finding | None:
    """No folios above or below the text block, for venues that paginate the proceedings.

    A folio is a bare number outside the text block whose value climbs with the
    page. Requiring the climb keeps a stray digit (an equation tag pushed into the
    foot, a figure label) from reading as pagination.
    """
    if not ctx.conf("template.forbid_page_numbers", False):
        return None
    folios: list[tuple[int, int, Evidence]] = []
    for page in ctx.doc.pages:
        fmt = page_format(ctx, page)
        top, bottom = fmt["margin_top_pt"], page.height - fmt["margin_bottom_pt"]
        candidates = []
        for line in page.lines:
            text = " ".join(line.text.split())
            m = _FOLIO_RE.fullmatch(text)
            if m is None or not (line.bbox[3] <= top or line.bbox[1] >= bottom):
                continue
            off_centre = abs((line.bbox[0] + line.bbox[2]) / 2 - page.width / 2)
            candidates.append((off_centre, int(m.group(1)), line.bbox, text))
        if candidates:
            _, value, bbox, text = min(candidates, key=lambda c: c[0])
            folios.append((page.number, value, Evidence(
                page=page.number, detail="page number outside the text block", quote=text,
                bbox=bbox, expected="no page numbers")))

    offsets = Counter(value - number for number, value, _ in folios)
    offset, count = offsets.most_common(1)[0] if offsets else (0, 0)
    if count < min(2, ctx.doc.page_count):
        return ctx.ok("page_numbers", "Page numbers",
                      "No page numbers found above or below the text block.",
                      category="format", cfp_key="page_numbers")
    numbered = [e for number, value, e in folios if value - number == offset]
    pages = ", ".join(str(e.page) for e in numbered[:10])
    return ctx.error(
        "page_numbers", "Page numbers",
        f"{len(numbered)} page(s) carry a page number ({pages}{'...' if len(numbered) > 10 else ''}). "
        "The venue adds its own page numbers to the proceedings, and a paginated submission "
        "fails its format inspection.",
        category="format", evidence=numbered[:8], cfp_key="page_numbers",
        remedy="Remove the page numbers: keep \\pagestyle{empty} as the style file sets it, and do "
        "not load a package or option (such as a preprint option) that adds a footer.",
        confidence="high — bare numbers outside the text block that count up with the pages",
    )


@register("title_format", "Title format", module=MODULE, category="format", order=17)
def check_title_format(ctx: CheckContext) -> Finding | None:
    """The title set in capitals, for venues whose style file capitalises it."""
    if not ctx.conf("template.title_all_caps", False):
        return None
    title = title_text(ctx)
    if not title:
        return ctx.skip("title_format", "Title format", "No title could be identified on page 1.",
                        category="format", cfp_key="title_format")
    lower = [w for w in title.split() if any(c.islower() for c in w)]
    if lower:
        return ctx.warn(
            "title_format", "Title format",
            f"The title is not set in capitals ({len(lower)} word(s) have lower-case letters, "
            f"e.g. {lower[0]!r}); the venue asks for the title in ALL CAPITALS.",
            category="format", cfp_key="title_format",
            evidence=[Evidence(page=1, detail="title as printed", quote=title[:120],
                               expected="all capitals")],
            remedy="Build with the official style file, which capitalises \\title{} itself, or "
            "type the title in capitals in the Word template.",
            confidence="medium — the title is taken as the largest type on page 1",
        )
    return ctx.ok("title_format", "Title format", "The title is set in capitals.",
                  category="format", cfp_key="title_format")
