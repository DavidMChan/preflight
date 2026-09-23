"""Was the paper built on this year's style file, in the mode the track needs?

Templates that print a running head say both at once: ICLR's reads "Under
review as a conference paper at ICLR 2027" at submission and "Published as ..."
once ``\\iclrfinalcopy`` is switched on. A stale year means last cycle's style
files; the camera-ready wording on a submission means the switch that prints
the author block was left on.
"""

from __future__ import annotations

import re

from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

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
