"""Cross-reference checks: the relationship between prose callouts and the figure/table
objects they point at.

``figures_structural.py`` already checks caption numbering for gaps and duplicates within
each figure/table sequence; this module never re-derives that. It instead asks four
different questions, all about the link between the *text* and the *objects*:
every object is pointed at from somewhere, every callout in the text points at something
real, panel letters used in prose match panel letters defined in the caption, and
appendix/supplement identifiers ("Appendix C", "Table S1") resolve to something in this PDF.

All four are counting exercises over regexes and ``analysis.captions()`` -- deterministic,
but not desk-rejection conditions, so every finding here is a WARNING, never an ERROR.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..analysis import Caption, captions, clean_text
from ..context import CheckContext
from ..document import Heading, Line
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.crossrefs"

# A single figure/table reference token, in the same shape analysis.captions() produces
# for Caption.label: an optional single uppercase letter (appendix/supplement prefix,
# with or without a following dot: "B.2", "S1"), then digits, then an optional ".sub".
_REF_TOKEN = r"(?:[A-Z]\.?)?\d+(?:\.\d+)?"
_TOKEN_RE = re.compile(r"([A-Z]\.?)?(\d+)(?:\.(\d+))?")
_RANGE_SEP_RE = re.compile(r"^\s*(?:[-–—]|to)\s*$", re.IGNORECASE)

# Lines that open a caption, mirroring analysis.captions()'s own detector -- used to
# exclude captions from the callout search text so a caption's own "Figure 3: ..." label
# never counts as the paper "calling out" figure 3.
_CAPTION_START_RE = re.compile(r"^(Figure|Fig\.|Table|Algorithm|Listing)\s+([A-Z]?\.?\d+(?:\.\d+)?)\s*[:.]")

_PANEL_DEF_RE = re.compile(r"\(([A-Za-z])\)")


# ---------------------------------------------------------------------------
# Shared text extraction: everything except captions, headings, and the bibliography
# ---------------------------------------------------------------------------


def _references_span(ctx: CheckContext) -> tuple[Heading | None, Heading | None]:
    aliases = [str(a) for a in (ctx.conf("structure.references_aliases", ["References", "Bibliography"]) or [])]
    heading = ctx.doc.find_heading(aliases)
    if heading is None:
        return None, None
    following = next((h for h in ctx.doc.headings if h.order > heading.order), None)
    return heading, following


def _is_heading_line(line: Line, headings: list[Heading]) -> bool:
    return any(line.page == h.page and abs(line.bbox[1] - h.bbox[1]) < 0.6 for h in headings)


def _callout_search_pages(ctx: CheckContext) -> list[tuple[int, str]]:
    """(page, text) pairs to scan for callouts: everything except the bibliography.

    Captions are NOT excised here. Skipping forward from a caption start swallows
    the prose that follows it in the same column, which silently lost real
    callouts — a figure discussed immediately under its own caption looked
    uncited. A caption label is instead told apart from a callout by its colon:
    "Figure 3:" introduces the float, "Figure 3" refers to it. Section headings
    are still skipped so "A Appendix" does not refer to itself, and the
    bibliography is skipped so "Table 2 of [12]" inside a citation is not read as
    a cross-reference.
    """
    cached = ctx.shared.get("crossrefs_page_texts")
    if isinstance(cached, list):
        return cached

    ref_heading, ref_following = _references_span(ctx)
    headings = ctx.doc.headings
    lines = ctx.doc.reading_order
    skip_biblio = False
    by_page: dict[int, list[str]] = defaultdict(list)

    for line in lines:
        if ref_heading is not None and not skip_biblio and line.page == ref_heading.page and \
                abs(line.bbox[1] - ref_heading.bbox[1]) < 0.6:
            skip_biblio = True
        if skip_biblio and ref_following is not None and line.page == ref_following.page and \
                abs(line.bbox[1] - ref_following.bbox[1]) < 0.6:
            skip_biblio = False
        if skip_biblio:
            continue

        if _is_heading_line(line, headings):
            continue
        by_page[line.page].append(line.text)

    # Join with newlines, not spaces: clean_text() undoes end-of-line hyphenation
    # only across a newline, and a callout split as "Fig-\nure 3" is common enough
    # that joining with a space silently loses it.
    out = [(p, clean_text(ctx, "\n".join(t))) for p, t in sorted(by_page.items())]
    ctx.shared["crossrefs_page_texts"] = out
    return out


def _context_window(text: str, start: int, end: int, radius: int) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    left_dot = text.rfind(". ", lo, start)
    if left_dot != -1:
        lo = left_dot + 2
    right_dot = text.find(". ", end, hi)
    if right_dot != -1:
        hi = right_dot + 1
    return " ".join(text[lo:hi].strip().split())


# ---------------------------------------------------------------------------
# Expanding "Figures 1-3", "Tables 2 and 3" into individual labels
# ---------------------------------------------------------------------------


def _expand_refs(raw: str, max_span: int) -> list[str]:
    matches = list(_TOKEN_RE.finditer(raw))
    if not matches:
        return []
    tokens = [(m.group(1) or "", int(m.group(2)), m.group(3)) for m in matches]
    seps = [raw[matches[i].end():matches[i + 1].start()] for i in range(len(matches) - 1)]

    def fmt(prefix_raw: str, num: int, sub: str | None) -> str:
        return f"{prefix_raw}{num}" + (f".{sub}" if sub else "")

    labels = [fmt(*tokens[0])]
    for i in range(1, len(tokens)):
        prefix, num, sub = tokens[i]
        prev_prefix, prev_num, prev_sub = tokens[i - 1]
        same_prefix = prefix.rstrip(".").upper() == prev_prefix.rstrip(".").upper()
        is_range = _RANGE_SEP_RE.match(seps[i - 1])
        if is_range and same_prefix and not sub and not prev_sub and 0 < num - prev_num <= max_span:
            labels.extend(fmt(prev_prefix, n, None) for n in range(prev_num + 1, num + 1))
        else:
            labels.append(fmt(prefix, num, sub))
    return labels


def _kind_of(keyword: str) -> str:
    k = keyword.lower().rstrip(".")
    if k.startswith("fig"):
        return "figure"
    if k.startswith("tab"):
        return "table"
    return k


def _figtab_callout_re(ctx: CheckContext) -> re.Pattern[str]:
    fig_words = [str(w) for w in (ctx.conf("crossrefs.figure_keywords", ["Figures", "Figure", "Fig."]) or [])]
    tab_words = [str(w) for w in (ctx.conf("crossrefs.table_keywords", ["Tables", "Table", "Tab."]) or [])]
    keywords = "|".join(re.escape(w) for w in fig_words + tab_words)
    return re.compile(
        rf"\b({keywords})\s+({_REF_TOKEN}(?:\s*(?:[-–—]|to|,|and|&)\s*{_REF_TOKEN})*)"
    )


def _find_callouts(ctx: CheckContext) -> list[tuple[int, str, str, int, int]]:
    """(page, kind, label, match_start, match_end) for every figure/table callout."""
    cached = ctx.shared.get("crossrefs_callouts")
    if isinstance(cached, list):
        return cached
    max_span = int(ctx.conf("crossrefs.range_max_span", 20))
    pattern = _figtab_callout_re(ctx)
    out: list[tuple[int, str, str, int, int]] = []
    for page, text in _callout_search_pages(ctx):
        for m in pattern.finditer(text):
            # "Figure 3:" introduces the float; "Figure 3" refers to it. Without
            # this, every caption counts as a reference to itself and nothing is
            # ever reported as uncited.
            if text[m.end():m.end() + 2].lstrip().startswith(":"):
                continue
            kind = _kind_of(m.group(1))
            for label in _expand_refs(m.group(2), max_span):
                out.append((page, kind, label, m.start(), m.end()))
    ctx.shared["crossrefs_callouts"] = out
    return out


# ---------------------------------------------------------------------------
# 1. figure_callouts
# ---------------------------------------------------------------------------


@register("figure_callouts", "Every figure and table is referred to in the text", module=MODULE,
          category="figures", order=43)
def check_figure_callouts(ctx: CheckContext) -> Finding:
    """Flag figures/tables that no sentence in the paper ever points at."""
    cap = int(ctx.conf("crossrefs.max_reported", 15))
    caps = captions(ctx)
    objects = [c for c in caps if c.kind in {"figure", "table"}]
    if not objects:
        return ctx.skip(
            "figure_callouts", "Every figure and table is referred to in the text",
            "No figure or table captions were extracted from this PDF.", category="figures",
        )

    referenced = {(kind, label) for _p, kind, label, _s, _e in _find_callouts(ctx)}
    missing = [c for c in objects if (c.kind, c.label) not in referenced]

    if missing:
        evidence = [Evidence(page=c.page, detail=f"{c.name} is never mentioned in the prose",
                             quote=c.text[:140]) for c in missing[:cap]]
        return ctx.warn(
            "figure_callouts",
            "Every figure and table is referred to in the text",
            f"{len(missing)} of {len(objects)} figure/table object(s) are never called out "
            "anywhere in the running text. A float the text never points at is either "
            "redundant or was left behind by an edit.",
            category="figures",
            evidence=evidence,
            remedy="Either cite the object explicitly (\"as shown in Table 4, ...\") or remove it.",
            confidence="medium -- relies on a regex over 'Figure N'/'Table N' phrasing; an "
            "unconventional callout style could be missed",
        )
    return ctx.ok(
        "figure_callouts",
        "Every figure and table is referred to in the text",
        f"All {len(objects)} figure/table object(s) are referred to somewhere in the text.",
        category="figures",
    )


# ---------------------------------------------------------------------------
# 2. dangling_callouts
# ---------------------------------------------------------------------------


@register("dangling_callouts", "Cross-references resolve", module=MODULE,
          category="figures", order=44)
def check_dangling_callouts(ctx: CheckContext) -> Finding:
    """Flag callouts in the text ("Figure 7") that name an object the PDF does not have."""
    cap = int(ctx.conf("crossrefs.max_reported", 15))
    radius = int(ctx.conf("crossrefs.context_window_chars", 100))
    caps = captions(ctx)
    existing = {(c.kind, c.label) for c in caps}
    callouts = _find_callouts(ctx)
    if not callouts:
        return ctx.skip(
            "dangling_callouts", "Cross-references resolve",
            "No 'Figure N'/'Table N' style callouts were found in the prose.", category="figures",
        )

    pages = dict(_callout_search_pages(ctx))
    dangling: list[Evidence] = []
    seen: set[tuple[int, str, str]] = set()
    for page, kind, label, start, end in callouts:
        if (kind, label) in existing:
            continue
        key = (page, kind, label)
        if key in seen:
            continue
        seen.add(key)
        quote = _context_window(pages.get(page, ""), start, end, radius)
        dangling.append(Evidence(page=page, detail=f"references {kind} {label}, which does not exist",
                                 quote=quote, expected=f"a {kind} captioned '{label}' in this PDF"))

    if dangling:
        n_objects = len({c.kind for c in caps})  # noqa: F841 - kept for readability at call sites below
        counts = f"{sum(1 for c in caps if c.kind == 'figure')} figure(s), " \
                 f"{sum(1 for c in caps if c.kind == 'table')} table(s)"
        return ctx.warn(
            "dangling_callouts",
            "Cross-references resolve",
            f"{len(dangling)} callout(s) point at a figure or table that is not in this PDF "
            f"(the paper has {counts}). This often means a float was deleted or renumbered "
            "after the text was last edited.",
            category="figures",
            evidence=dangling[:cap],
            remedy="Update the callout to the correct number, or restore the missing float.",
            confidence="medium -- relies on a regex over 'Figure N'/'Table N' phrasing",
        )
    return ctx.ok(
        "dangling_callouts",
        "Cross-references resolve",
        f"All {len(callouts)} figure/table callout(s) in the text resolve to a caption in this PDF.",
        category="figures",
    )


# ---------------------------------------------------------------------------
# 3. panel_completeness
# ---------------------------------------------------------------------------


def _defined_panels(text: str) -> list[str]:
    seen: list[str] = []
    for m in _PANEL_DEF_RE.finditer(text):
        letter = m.group(1).upper()
        if letter not in seen:
            seen.append(letter)
    return seen


def _panel_callout_re() -> re.Pattern[str]:
    return re.compile(r"\b(Figure|Fig\.)\s+(\d+(?:\.\d+)?)\s*\(?([A-Z])\)?\b(?!\.\d)")


@register("panel_completeness", "Figure panels", module=MODULE, category="figures", order=45)
def check_panel_completeness(ctx: CheckContext) -> Finding:
    """Check that panel letters defined in a caption and used in the text are consistent.

    Only runs against figures whose caption actually enumerates panels like "(A) ... (B)
    ..."; most figures have none, and that is normal, not a finding.
    """
    min_panels = int(ctx.conf("crossrefs.panel_min_defined", 2))
    cap = int(ctx.conf("crossrefs.max_reported", 15))
    caps = {c.label: c for c in captions(ctx) if c.kind == "figure"}
    paneled = {label: _defined_panels(c.text) for label, c in caps.items()}
    paneled = {label: panels for label, panels in paneled.items() if len(panels) >= min_panels}

    if not paneled:
        return ctx.skip(
            "panel_completeness", "Figure panels",
            "No figure caption defines lettered panels (e.g. \"(A) ... (B) ...\").",
            category="figures",
        )

    issues: list[Evidence] = []

    # Direction 1: a caption defines (A), (B), (D) -- (C) is missing.
    for label, panels in paneled.items():
        positions = sorted(ord(p) - ord("A") for p in panels)
        missing = [chr(p + ord("A")) for p in range(positions[0], positions[-1] + 1) if p not in positions]
        if missing:
            issues.append(Evidence(
                page=caps[label].page,
                detail=f"Figure {label}'s caption defines panels {', '.join(sorted(panels))} but "
                f"skips {', '.join(missing)}",
                quote=caps[label].text[:140],
            ))

    # Direction 2: the text calls out a panel letter the caption never defined.
    for page, text in _callout_search_pages(ctx):
        for m in _panel_callout_re().finditer(text):
            num, letter = m.group(2), m.group(3).upper()
            panels = paneled.get(num)
            if panels is None:
                continue  # not a paneled figure -- nothing to check
            if letter not in panels:
                issues.append(Evidence(
                    page=page,
                    detail=f"text refers to Figure {num}{letter}, but its caption only defines "
                    f"panel(s) {', '.join(sorted(panels))}",
                    quote=_context_window(text, m.start(), m.end(), 100),
                ))

    if issues:
        return ctx.warn(
            "panel_completeness",
            "Figure panels",
            f"{len(issues)} panel-labeling inconsistenc(y/ies) across {len(paneled)} paneled "
            "figure(s).",
            category="figures",
            evidence=issues[:cap],
            remedy="Make sure every panel letter used in the text is defined in the caption, "
            "and that a caption's panel lettering has no gaps.",
            confidence="medium -- panel detection relies on \"(A)\"-style markers in the caption",
        )
    return ctx.ok(
        "panel_completeness",
        "Figure panels",
        f"Checked {len(paneled)} figure(s) with lettered panels; panel definitions and "
        "in-text references agree.",
        category="figures",
    )


# ---------------------------------------------------------------------------
# 4. supplement_identifiers
# ---------------------------------------------------------------------------


def _appendix_callout_re(ctx: CheckContext) -> re.Pattern[str]:
    words = [str(w) for w in (ctx.conf("crossrefs.appendix_keywords", ["Appendix", "Appendices"]) or [])]
    keywords = "|".join(re.escape(w) for w in words)
    token = r"[A-Z](?:\.\d+(?:\.\d+)?)?"
    return re.compile(rf"\b({keywords})\s+({token}(?:\s*(?:,|and|&)\s*{token})*)\b")


def _section_callout_re(ctx: CheckContext) -> re.Pattern[str]:
    words = [str(w) for w in (ctx.conf("crossrefs.section_keywords", ["Section", "Sec."]) or [])]
    keywords = "|".join(re.escape(w) for w in words)
    return re.compile(rf"\b({keywords})\s+([A-Z]\.\d+(?:\.\d+)?)\b")


def _valid_appendix_identifiers(ctx: CheckContext, caps: list[Caption]) -> set[str]:
    valid: set[str] = set()
    for h in ctx.doc.headings:
        if h.numbering and re.match(r"^[A-Z](\.|$)", h.numbering):
            valid.add(h.numbering.upper().rstrip("."))
    for c in caps:
        if re.match(r"^[A-Z]", c.label):
            valid.add(c.label.upper())
            valid.add(c.label.upper()[0])
    return valid


def _resolves(token: str, valid: set[str]) -> bool:
    token = token.upper()
    if token in valid:
        return True
    return any(v == token or v.startswith(token + ".") for v in valid)


@register("supplement_identifiers", "Supplementary and appendix references", module=MODULE,
          category="figures", order=46)
def check_supplement_identifiers(ctx: CheckContext) -> Finding:
    """Every "Appendix X" / "Section A.3" reference must resolve to a heading in this PDF,
    and no two appendix/supplement objects should share an identifier.

    Figure/table callouts like "Table S1" are already checked for resolution by
    ``dangling_callouts``; this check adds the identifiers that check cannot see
    (section/appendix headings) and duplicate-identifier detection scoped to the
    appendix/supplement labels specifically.
    """
    cap = int(ctx.conf("crossrefs.max_reported", 15))
    radius = int(ctx.conf("crossrefs.context_window_chars", 100))
    caps = captions(ctx)
    valid = _valid_appendix_identifiers(ctx, caps)

    unresolved: list[Evidence] = []
    if valid:
        # Only run the resolution half once the PDF actually has lettered appendix
        # material -- otherwise every "Appendix" mention would be a false positive.
        appendix_re = _appendix_callout_re(ctx)
        section_re = _section_callout_re(ctx)
        for page, text in _callout_search_pages(ctx):
            for m in appendix_re.finditer(text):
                for token in re.findall(r"[A-Z](?:\.\d+(?:\.\d+)?)?", m.group(2)):
                    if not _resolves(token, valid):
                        unresolved.append(Evidence(
                            page=page, detail=f"references Appendix {token}, which does not exist",
                            quote=_context_window(text, m.start(), m.end(), radius),
                            expected="a heading or captioned object under that appendix letter",
                        ))
            for m in section_re.finditer(text):
                token = m.group(2)
                if not _resolves(token, valid):
                    unresolved.append(Evidence(
                        page=page, detail=f"references Section {token}, which does not exist",
                        quote=_context_window(text, m.start(), m.end(), radius),
                        expected="a heading numbered that way in this PDF",
                    ))

    dup_groups: dict[tuple[str, str], list[Caption]] = defaultdict(list)
    for c in caps:
        if re.match(r"^[A-Z]", c.label):
            dup_groups[(c.kind, c.label.upper())].append(c)
    duplicates = [(k, v) for k, v in dup_groups.items() if len(v) > 1]
    dup_evidence = [
        Evidence(detail=f"'{k[0].capitalize()} {k[1]}' appears {len(v)} times, on pages "
                 f"{', '.join(str(c.page) for c in v)}")
        for k, v in duplicates
    ]

    evidence = (unresolved + dup_evidence)[:cap]
    if evidence:
        parts = []
        if unresolved:
            parts.append(f"{len(unresolved)} unresolved appendix/section reference(s)")
        if dup_evidence:
            parts.append(f"{len(dup_evidence)} duplicated appendix identifier(s)")
        return ctx.warn(
            "supplement_identifiers",
            "Supplementary and appendix references",
            f"{' and '.join(parts)} found. An unresolvable \"Appendix C\" usually means "
            "the appendix is missing, was renamed, or lives in a separate supplementary "
            "file that this PDF does not contain.",
            category="figures",
            evidence=evidence,
            remedy="Make sure every appendix/section letter cited in the text has a matching "
            "heading, and that no two objects share an identifier.",
            confidence="medium -- heading and label detection are both heuristic",
        )
    if not valid:
        return ctx.skip(
            "supplement_identifiers", "Supplementary and appendix references",
            "No lettered appendix headings or supplement-labeled objects were found in this PDF.",
            category="figures",
        )
    return ctx.ok(
        "supplement_identifiers",
        "Supplementary and appendix references",
        "Every appendix/section reference in the text resolves, and no appendix/supplement "
        "identifier is duplicated.",
        category="figures",
    )
