"""Section-heading structure checks: numbering hierarchy and empty sections.

Heading detection in :mod:`preflight.document` is heuristic and occasionally
promotes a bit of bold table-cell or figure-panel text on figure-heavy pages
into a spurious ``Heading``. Numbered headings are the reliable signal --
their numbering has to compose into a well-formed tree regardless of what
detector produced them -- so both checks here key off ``heading.numbering``
and are explicit about ignoring unnumbered headings rather than guessing.
"""

from __future__ import annotations

import re

from ..context import CheckContext
from ..document import Heading
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.headings"

_NUMERIC_RE = re.compile(r"^\d+(?:\.\d+)*$")
_LETTERED_RE = re.compile(r"^[A-Z](?:\.\d+)*$")


def _parts(numbering: str) -> tuple[int | str, ...]:
    """Split "3.1.2" into (3, 1, 2) or "B.2" into ("B", 2)."""
    bits = numbering.split(".")
    out: list[int | str] = []
    for b in bits:
        out.append(int(b) if b.isdigit() else b)
    return tuple(out)


def _effective_numbering(h: Heading) -> str:
    """Repair one specific, observed heading-extraction artifact.

    When a subsection number sits on its own short line ("F.1") with its
    label on a separate run the extractor does not merge, the regex in
    ``Document.headings`` back-tracks to the wrong split: numbering "F",
    label "1" instead of numbering "F.1". That is only possible when the
    detected numbering is a bare top-level token ("F", "3") and the "label"
    is itself nothing but digits -- a real section title never is -- so the
    fix is narrow enough not to touch genuine headings.
    """
    numbering = h.numbering or ""
    text = h.text.strip()
    if len(_parts(numbering)) == 1 and text.isdigit():
        return f"{numbering}.{text}"
    return numbering


@register("heading_hierarchy", "Heading hierarchy", module=MODULE, category="structure", order=37)
def check_heading_hierarchy(ctx: CheckContext) -> Finding:
    """Numbered section headings should form a contiguous, well-formed tree.

    Numeric headings ("3", "3.1") and lettered appendix headings ("A", "B.2")
    are validated as two separate sequences, since ACL/ARR-style appendices
    restart numbering with letters by design. Unnumbered headings -- which
    include spurious detections on table cells and figure panels -- are
    excluded entirely; they carry no numbering to validate.
    """
    numbered = [h for h in ctx.doc.headings if h.numbering]
    pairs = [(_effective_numbering(h), h) for h in numbered]
    numeric = [(n, h) for n, h in pairs if _NUMERIC_RE.match(n)]
    lettered = [(n, h) for n, h in pairs if _LETTERED_RE.match(n)]
    ignored = len(pairs) - len(numeric) - len(lettered)

    problems: list[Evidence] = []
    problems += _check_sequence(numeric, "numeric")
    problems += _check_sequence(lettered, "appendix")

    note = (
        f" {len(numeric)} numeric and {len(lettered)} appendix heading(s) were validated; "
        f"{len(ctx.doc.headings) - len(numbered)} unnumbered heading(s) were ignored"
        f"{f' ({ignored} numbered heading(s) matched neither scheme and were also ignored)' if ignored else ''}."
    )

    if not numeric and not lettered:
        return ctx.skip(
            "heading_hierarchy", "Heading hierarchy",
            "No numbered section headings were detected, so the hierarchy could not be validated.",
            category="structure",
        )

    if problems:
        return ctx.warn(
            "heading_hierarchy", "Heading hierarchy",
            f"{len(problems)} section-numbering issue(s) found.{note}",
            category="structure",
            evidence=problems[:10],
            remedy="Check for a section that was renumbered, deleted, or merged without updating "
            "the numbers around it -- common after a late edit or a \\ref pointing at a moved section.",
            confidence="medium -- heading detection is heuristic and can miss or misfire on a heading",
        )
    return ctx.ok(
        "heading_hierarchy", "Heading hierarchy",
        f"Section numbering is contiguous and well-nested.{note}",
        category="structure",
    )


def _check_sequence(pairs: list[tuple[str, Heading]], label: str) -> list[Evidence]:
    """Validate one numbering scheme (numeric or lettered) in document order.

    ``pairs`` holds (effective numbering, heading) -- the numbering used for
    tree logic, which may be a repaired version of ``heading.numbering``; the
    original ``heading.numbering`` is still what gets quoted in evidence.
    """
    if len(pairs) < 2:
        return []

    problems: list[Evidence] = []
    seen: set[tuple[int | str, ...]] = set()
    scheme_widths: set[int] = set()

    for numbering, h in pairs:
        parts = _parts(numbering)
        seen.add(parts)
        scheme_widths.add(len(parts))

        depth = len(parts)
        if depth == 1:
            continue  # top-level entries are checked for contiguity below

        parent = parts[:-1]
        if parent not in seen:
            problems.append(Evidence(
                page=h.page,
                detail=f"{label} heading {h.numbering!r} ({h.text!r}) has no parent section",
                expected=f"a preceding heading numbered {'.'.join(str(p) for p in parent)}",
            ))

    # Contiguity and start-at-1 within each parent, at every depth.
    by_parent: dict[tuple[int | str, ...], list[tuple[int, Heading, str]]] = {}
    for numbering, h in pairs:
        parts = _parts(numbering)
        if len(parts) < 1:
            continue
        parent = parts[:-1]
        last = parts[-1]
        if isinstance(last, int):
            by_parent.setdefault(parent, []).append((last, h, numbering))

    for parent, siblings in by_parent.items():
        siblings.sort(key=lambda triple: triple[0])
        numbers = [n for n, _, _ in siblings]
        if numbers[0] != 1:
            first_n, first_h, first_num = siblings[0]
            problems.append(Evidence(
                page=first_h.page,
                detail=f"{label} sequence under "
                f"{'.'.join(str(p) for p in parent) if parent else '(top level)'} "
                f"starts at {first_n} ({first_num!r} {first_h.text!r})",
                expected="starting at 1",
            ))
        for prev, nxt in zip(numbers, numbers[1:], strict=False):
            if nxt - prev > 1:
                nxt_h, nxt_num = next((h, num) for n, h, num in siblings if n == nxt)
                problems.append(Evidence(
                    page=nxt_h.page,
                    detail=f"{label} sequence jumps from {prev} to {nxt} "
                    f"({nxt_num!r} {nxt_h.text!r}); a heading is missing in between",
                ))

    # A numbering scheme changing mid-document: numeric headings normally stay
    # at a stable depth range; a jump in the set of widths in use (e.g. flat
    # "1", "2", "3" suddenly switching to "1.1.1" style depth with no "1.1")
    # is caught above via missing parents, so this only flags gross width churn.
    if label == "numeric" and len(scheme_widths) > 3:
        problems.append(Evidence(
            detail=f"section numbering depth varies unusually widely across the document "
            f"(depths seen: {sorted(scheme_widths)})",
        ))

    return problems


@register("empty_headings", "Empty sections", module=MODULE, category="structure", order=38)
def check_empty_headings(ctx: CheckContext) -> Finding:
    """Flag a heading with no body text and no subordinate heading beneath it.

    "4 Results" immediately followed by "4.1 Setup" is correct structure and
    must not be flagged -- a heading only counts as empty when nothing at all,
    neither prose nor a subsection, follows it before the next heading at its
    own level or higher.
    """
    min_words = int(ctx.conf("headings.empty_section_min_words", 3))
    all_headings = ctx.doc.headings
    # Restrict to numbered headings, for the same reason as heading_hierarchy:
    # unnumbered detections routinely include the title, the author block, and
    # bold labels inside figure panels or table cells, none of which are real
    # sections and all of which would otherwise read as "empty". Text from an
    # unnumbered heading in between two numbered ones still counts as content.
    headings = [h for h in all_headings if h.numbering]
    note = (
        f" Only the {len(headings)} numbered heading(s) were checked; "
        f"{len(all_headings) - len(headings)} unnumbered heading(s) (title, panel labels, etc.) were ignored."
    )

    if len(headings) < 2:
        return ctx.skip(
            "empty_headings", "Empty sections",
            "Fewer than two numbered headings were detected; nothing to check between them.",
            category="structure",
        )

    empties: list[Evidence] = []
    for i, h in enumerate(headings):
        nxt = headings[i + 1] if i + 1 < len(headings) else None
        between = ctx.doc.text_between(h, nxt)
        word_count = len(between.split())
        if word_count >= min_words:
            continue
        # A subsection immediately following is legitimate structure, not an
        # empty section -- the parent's "content" is its children.
        if nxt is not None and _is_subordinate(h, nxt):
            continue
        empties.append(Evidence(
            page=h.page,
            detail=f"heading {h.numbering + ' ' if h.numbering else ''}{h.text!r} is followed by "
            f"{'no text' if word_count == 0 else f'only {word_count} word(s)'} before the next heading",
            quote=between.strip()[:80] or None,
            measured=float(word_count),
            expected=f">= {min_words} words, or a subsection",
        ))

    if empties:
        return ctx.warn(
            "empty_headings", "Empty sections",
            f"{len(empties)} numbered heading(s) have little or no content beneath them before the "
            f"next heading.{note}",
            category="structure",
            evidence=empties[:10],
            remedy="Check that no section lost its body text in a template change, and that no "
            "placeholder heading was left in from an outline draft.",
            confidence="medium -- heading detection can miss short intervening text on figure-heavy pages",
        )
    return ctx.ok(
        "empty_headings", "Empty sections",
        f"All {len(headings)} numbered headings are followed by body text or a subsection.{note}",
        category="structure",
    )


def _is_subordinate(parent: Heading, nxt: Heading) -> bool:
    """True if ``nxt`` reads as a numbered child section of ``parent``.

    Without numbering on both sides there is no reliable way to tell a real
    subsection from an unrelated detection, so those cases fall through to
    the word-count test instead of being assumed legitimate.
    """
    if not parent.numbering or not nxt.numbering:
        return False
    p_parts = _parts(parent.numbering)
    n_parts = _parts(nxt.numbering)
    return len(n_parts) > len(p_parts) and n_parts[: len(p_parts)] == p_parts
