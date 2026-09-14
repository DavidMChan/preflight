"""Profile loading, inheritance, and the module/severity composition rules."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from preflight.models import Severity
from preflight.profile import ProfileError, available_profiles, load_profile


def test_bundled_profiles_load() -> None:
    assert "arr" in available_profiles()
    assert "neurips" in available_profiles()
    assert "baylearn" in available_profiles()
    # Abstract bases are not offered as choices.
    assert "base" not in available_profiles()
    assert "acl" not in available_profiles()


def test_arr_inherits_acl_geometry() -> None:
    arr = load_profile("arr")
    assert arr.lineage == ("arr", "acl", "base")
    assert arr.get("geometry.page_width_pt") == 595.0        # from base
    assert arr.get("fonts.body_size_pt") == 11.0             # from acl
    assert arr.get("structure.limitations_required") is True  # from acl
    assert arr.track("long").content_page_limit == 8          # from arr
    assert arr.track("short").content_page_limit == 4


def test_neurips_overrides_geometry_and_replaces_tracks() -> None:
    neurips = load_profile("neurips")
    assert neurips.get("geometry.page_width_pt") == 612.0
    assert neurips.get("columns.expected_columns") == 1
    # tracks_replace: true means base's `long` track is gone.
    assert set(neurips.tracks) == {"main"}
    # ...but non-replaced mappings still inherit.
    assert neurips.get("anonymity.forbidden_url_domains")


def test_modules_compose_across_inheritance() -> None:
    arr = load_profile("arr")
    assert "core.geometry" in arr.modules       # base
    assert "neurips.checklist" in arr.modules   # arr adds it
    assert "arr.responsible_nlp" in arr.modules


def test_module_enabled_matches_parent_namespaces() -> None:
    arr = load_profile("arr")
    assert arr.module_enabled("core.geometry")
    assert not arr.module_enabled("nonexistent.module")


def test_severity_override_softens_failures_only() -> None:
    neurips = load_profile("neurips")
    assert neurips.severity_for("limitations_present", Severity.ERROR) is Severity.WARNING
    # A pass is never rewritten into a failure, or vice versa.
    assert neurips.severity_for("limitations_present", Severity.PASS) is Severity.PASS
    assert neurips.severity_for("limitations_present", Severity.SKIPPED) is Severity.SKIPPED


def test_arr_disables_the_checklist_presence_check() -> None:
    arr = load_profile("arr")
    assert not arr.check_enabled("neurips_checklist_present")
    assert arr.check_enabled("neurips_checklist_complete")


def test_baylearn_is_a_two_page_neurips_abstract() -> None:
    baylearn = load_profile("baylearn")
    assert baylearn.lineage == ("baylearn", "neurips", "base")
    assert baylearn.get("geometry.page_width_pt") == 612.0     # NeurIPS US Letter
    assert baylearn.get("columns.expected_columns") == 1
    assert set(baylearn.tracks) == {"abstract"}
    assert baylearn.track(None).content_page_limit == 2
    # A two-page abstract carries no Limitations section and no checklist.
    assert baylearn.get("structure.limitations_required") is False
    assert not any(baylearn.check_enabled(c) for c in (
        "neurips_checklist_present",
        "neurips_checklist_position",
        "neurips_checklist_complete",
        "neurips_checklist_justifications",
        "neurips_checklist_consistency",
    ))
    # Only acknowledgements and references buy space past the limit.
    assert baylearn.get("structure.unlimited_after") == ["references", "acknowledgments"]


def test_unknown_conference_and_track_are_reported() -> None:
    with pytest.raises(ProfileError, match="unknown conference"):
        load_profile("definitely-not-a-venue")
    with pytest.raises(ProfileError, match="unknown track"):
        load_profile("arr").track("keynote")


def test_custom_profile_from_path(tmp_path: Path) -> None:
    path = tmp_path / "myvenue.yaml"
    path.write_text(textwrap.dedent("""
        conference:
          key: myvenue
          name: My Venue
          extends: acl
          default_track: long
        tracks_replace: true
        tracks:
          long:
            content_page_limit: 12
        severity:
          margins: warning
    """))
    profile = load_profile(str(path))
    assert profile.name == "My Venue"
    assert profile.track("long").content_page_limit == 12
    assert profile.get("fonts.body_size_pt") == 11.0     # inherited from acl
    assert profile.severity_for("margins", Severity.ERROR) is Severity.WARNING


def test_circular_inheritance_is_rejected(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"conference: {{key: a, extends: '{b}'}}\ntracks: {{long: {{content_page_limit: 8}}}}\n")
    b.write_text(f"conference: {{key: b, extends: '{a}'}}\ntracks: {{long: {{content_page_limit: 8}}}}\n")
    with pytest.raises(ProfileError, match="circular"):
        load_profile(str(a))


# -- the US Letter two-column venues --------------------------------------

LETTER_TWO_COLUMN = ("cvpr", "aistats", "icml", "aaai", "icra")


@pytest.mark.parametrize("key", LETTER_TWO_COLUMN)
def test_letter_two_column_profiles_load(key: str) -> None:
    profile = load_profile(key)
    assert key in available_profiles()
    assert profile.lineage == (key, "base")
    assert profile.get("geometry.page_width_pt") == 612.0     # US Letter, not base's A4
    assert profile.get("geometry.page_height_pt") == 792.0
    assert profile.get("columns.expected_columns") == 2
    assert profile.get("fonts.body_size_pt") == 10.0          # not base's 11pt


@pytest.mark.parametrize("key", LETTER_TWO_COLUMN)
def test_letter_two_column_profiles_replace_the_base_track(key: str) -> None:
    """base.yaml defines `long`; inheriting it would offer a track that does not exist."""
    profile = load_profile(key)
    assert "long" not in profile.tracks
    assert profile.default_track in profile.tracks


@pytest.mark.parametrize("key", LETTER_TWO_COLUMN)
def test_letter_two_column_severity_keys_name_real_checks(key: str) -> None:
    """A severity override for a misspelled check id would silently do nothing."""
    from preflight.registry import describe, load_builtin_checks

    load_builtin_checks()
    ids = {entry["id"] for entry in describe()}
    profile = load_profile(key)
    assert set(profile.severity_overrides) <= ids
    assert set(profile.disabled_checks) <= ids


def test_content_page_limits_match_the_published_calls() -> None:
    assert load_profile("cvpr").track("main").content_page_limit == 8
    assert load_profile("aistats").track("main").content_page_limit == 8
    assert load_profile("aistats").track("camera_ready").content_page_limit == 9
    assert load_profile("icml").track("main").content_page_limit == 8
    assert load_profile("aaai").track("main").content_page_limit == 7
    assert load_profile("icra").track("main").content_page_limit == 8


def test_icra_counts_the_complete_pdf_and_enables_papercept_rules() -> None:
    icra = load_profile("icra")
    assert icra.get("structure.unlimited_after") == []
    assert icra.get("structure.appendix_must_follow_references") is False
    assert icra.get("pdf.minimum_version") == "1.4"
    assert icra.get("pdf.require_base14_embedding") is True
    assert icra.get("pdf.forbid_type3_fonts") is True
    assert icra.get("pdf.forbid_hyperlinks") is True
    assert icra.get("pdf.forbid_bookmarks") is True


def test_unverified_geometry_cannot_desk_reject() -> None:
    """Profiles without published checker tolerances keep margin findings advisory."""
    for key in LETTER_TWO_COLUMN:
        profile = load_profile(key)
        assert profile.severity_for("margins", Severity.ERROR) is Severity.WARNING, key
