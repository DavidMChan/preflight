"""The deterministic checks added for markup, citations, statistics and figures.

The bar for these is that they stay quiet on a clean paper: they run on every
submission by default, so a false positive costs an author real time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from preflight.context import CheckContext, Settings
from preflight.document import Document
from preflight.models import Severity
from preflight.profile import load_profile
from preflight.registry import load_builtin_checks

NEW_MODULES = {
    "core.markup", "core.pdf", "core.citations", "core.statistics",
    "core.abbreviations", "core.crossrefs", "core.headings", "core.figure_quality",
}


def _ctx(path: Path) -> CheckContext:
    profile = load_profile("arr")
    return CheckContext(doc=Document(path), profile=profile, track=profile.track("long"),
                        settings=Settings.offline())


def test_every_new_module_is_registered_and_advisory() -> None:
    """None of these is a documented desk-rejection condition, so none may error."""
    registry = load_builtin_checks()
    found = {c.module for c in registry if c.module in NEW_MODULES}
    assert found == NEW_MODULES
    for check in registry:
        if check.module in NEW_MODULES:
            assert check.requires == ()          # deterministic: no opt-in flag
            assert not check.is_async            # no network, so no need


def test_new_checks_are_quiet_on_a_clean_paper(clean_paper: Path) -> None:
    """A synthetic, well-formed paper must not trip any of them."""
    registry = load_builtin_checks()
    ctx = _ctx(clean_paper)
    try:
        noisy: list[str] = []
        for check in registry:
            if check.module not in NEW_MODULES:
                continue
            for finding in check.run(ctx):
                if finding.severity in (Severity.ERROR, Severity.WARNING):
                    noisy.append(f"{finding.check_id}: {finding.message[:90]}")
        assert not noisy, "false positives on a clean paper:\n" + "\n".join(noisy)
    finally:
        ctx.doc.close()


def test_new_checks_never_crash_and_are_fast(clean_paper: Path) -> None:
    import time

    registry = load_builtin_checks()
    ctx = _ctx(clean_paper)
    try:
        started = time.monotonic()
        for check in registry:
            if check.module in NEW_MODULES:
                for finding in check.run(ctx):
                    # A crash is reported as SKIPPED with the exception in the message.
                    assert "Check failed to run" not in finding.message, finding.message
        assert time.monotonic() - started < 20.0
    finally:
        ctx.doc.close()


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["the effect was significant (p = 0)", "p = .000", "p = 0.000", "p=0.0", "P < .000"],
)
def test_p_value_zero_fires(text: str) -> None:
    from preflight.checks.statistics import _ZERO_PATTERN

    assert _ZERO_PATTERN.search(text), text


@pytest.mark.parametrize(
    "text",
    ["p = 0.001", "p < .001", "p = 0.04", "p = .032", "p = 0.0498", "p < 0.05"],
)
def test_p_value_zero_does_not_fire_on_valid_reports(text: str) -> None:
    """Only a value that rounds to nothing is wrong; small p-values are correct."""
    from preflight.checks.statistics import _ZERO_PATTERN

    assert not _ZERO_PATTERN.search(text), text


# ---------------------------------------------------------------------------
# Cross-references
# ---------------------------------------------------------------------------

def test_a_caption_is_not_a_callout_to_itself(clean_paper: Path) -> None:
    """"Figure 3:" introduces a float; "Figure 3" refers to one.

    Without that distinction every caption counts as its own reference and the
    check can never report anything.
    """
    from preflight.checks.crossrefs import _find_callouts

    ctx = _ctx(clean_paper)
    try:
        ctx.shared["crossrefs_page_texts"] = [(1, "See Figure 2 for detail. Figure 2: A caption.")]
        callouts = _find_callouts(ctx)
        labels = [label for _, _, label, _, _ in callouts]
        assert labels == ["2"], labels        # the caption occurrence is not counted
    finally:
        ctx.doc.close()


def test_hyphenated_callouts_survive_the_line_break(clean_paper: Path) -> None:
    """"Fig-\\nure 3" is a reference to Figure 3, and was silently lost."""
    from preflight.analysis import clean_text

    ctx = _ctx(clean_paper)
    try:
        joined = clean_text(ctx, "as shown in Fig-\nure 3 the effect holds")
        assert "Figure 3" in joined
    finally:
        ctx.doc.close()


# ---------------------------------------------------------------------------
# PDF integrity
# ---------------------------------------------------------------------------

def test_pdf_health_reports_the_facts(clean_paper: Path) -> None:
    registry = load_builtin_checks()
    ctx = _ctx(clean_paper)
    try:
        finding = registry.checks["pdf_health"].run(ctx)[0]
        assert finding.severity is Severity.PASS
        assert "12 pages" in finding.message
        assert "not encrypted" in finding.message
    finally:
        ctx.doc.close()


def test_base14_fonts_are_exempt_from_embedding() -> None:
    """The standard fonts are legitimately never embedded."""
    profile = load_profile("arr")
    base14 = [f.lower() for f in profile.get("pdf.base14_fonts", [])]
    assert "helvetica" in base14
    assert "times-roman" in base14
    assert len(base14) == 14


# ---------------------------------------------------------------------------
# Numeric-vs-author-year citation style
# ---------------------------------------------------------------------------

_AUTHOR_YEAR_BODY = (
    "Scratchpads help spatial reasoning (Menon et al., 2024; Duan et al., 2025), and "
    "interleaved traces are cheap to collect (Li et al., 2025a). Shi et al. (2025) agree.\n"
)
#: A label vector of the kind papers print in an appendix or a quoted prompt.
_DATA_VECTORS = "".join(
    f"row {i}: [0, 0, 2, 0, 3, 0, 0, 0, 0, 0]\n" for i in range(40)
)


def test_bracketed_data_vectors_are_not_numeric_citations() -> None:
    """A page of "[0, 0, 2, ...]" must not make an author-year paper look numeric.

    When it did, the bibliography was read by position and almost every entry
    came back "uncited".
    """
    from preflight.checks.citations import detect_numeric_style

    assert not detect_numeric_style(_AUTHOR_YEAR_BODY + _DATA_VECTORS, 19)


def test_real_numeric_citations_are_still_detected() -> None:
    from preflight.checks.citations import detect_numeric_style

    body = "Prior work [1], [2, 3] and [5-7] agree, as does [4] and [8].\n"
    assert detect_numeric_style(body, 19)


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [
        ([1], True),
        ([2, 3], True),
        ([5, 6, 7], True),
        ([0], False),                       # reference lists start at [1]
        ([0, 0, 2, 0, 3], False),           # zeros and repeats: data, not a citation
        ([2, 2], False),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], False),   # longer than any real citation group
        ([], False),
    ],
)
def test_citation_group_plausibility(numbers: list[int], expected: bool) -> None:
    from preflight.checks.citations import _is_citation_group

    assert _is_citation_group(numbers, None) is expected


def test_citation_group_ceiling_is_optional() -> None:
    """Detection wants the ceiling; the resolution check must not have it.

    A citation past the end of the list is the failure that check reports.
    """
    from preflight.checks.citations import _is_citation_group

    assert not _is_citation_group([25], 19)
    assert _is_citation_group([25], None)
