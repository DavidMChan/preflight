"""Requirements a PDF checker must not claim to have verified.

Reporting these keeps the tool honest: a green run means "the PDF-checkable
rules pass", not "this submission complies with the CFP".
"""

from __future__ import annotations

from ..context import CheckContext
from ..models import Finding, Severity
from ..registry import register

MODULE = "core.unverifiable"


@register("unverifiable", "Out of scope for a PDF checker", module=MODULE,
          category="unverifiable", order=90)
def check_unverifiable(ctx: CheckContext) -> list[Finding]:
    """Emit one finding per requirement the profile lists as out of scope."""
    entries = ctx.conf("unverifiable", []) or []
    out: list[Finding] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        out.append(
            ctx.finding(
                f"unverifiable.{entry.get('id', 'item')}",
                str(entry.get("title", "Unverifiable requirement")),
                Severity.UNVERIFIABLE,
                str(entry.get("message", "")),
                category="unverifiable",
            )
        )
    return out
