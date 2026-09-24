"""Conference profiles.

A conference is a *composition of check modules* plus the settings those modules
read. Profiles are YAML so they stay editable by people who are not going to
open the source tree, and they support single inheritance via ``extends`` so a
venue can start from a shared base and change only what differs.

Resolution order for a key is: this profile > its parent > its grandparent ...
Mappings merge recursively; lists and scalars are replaced wholesale.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from .models import Severity


class ProfileError(ValueError):
    """A profile file is missing, malformed, or asks for something undefined."""


@dataclass(slots=True)
class TrackSpec:
    """One submission type within a conference (long paper, short paper, demo...)."""

    name: str
    content_page_limit: int
    description: str = ""
    # A cap on the whole PDF, for venues that allow end matter past the content
    # limit but only so much of it (ICASSP's 4 + 1).
    total_page_limit: int | None = None


@dataclass(slots=True)
class Profile:
    key: str
    name: str
    source: str
    tracks: dict[str, TrackSpec]
    default_track: str
    modules: tuple[str, ...]
    severity_overrides: dict[str, Severity]
    disabled_checks: frozenset[str]
    lineage: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict)

    # -- lookups ----------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}

    def get(self, path: str, default: Any = None) -> Any:
        """Fetch a dotted path such as ``geometry.page_width_pt``."""
        node: Any = self.raw
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def track(self, name: str | None) -> TrackSpec:
        key = (name or self.default_track).lower()
        if key not in self.tracks:
            options = ", ".join(sorted(self.tracks))
            raise ProfileError(f"unknown track {name!r} for {self.key}; available: {options}")
        return self.tracks[key]

    def check_enabled(self, check_id: str) -> bool:
        return check_id not in self.disabled_checks

    def module_enabled(self, module: str) -> bool:
        """A module is active if it, or one of its parent namespaces, is listed."""
        if not self.modules:
            return True
        parts = module.split(".")
        candidates = {".".join(parts[: i + 1]) for i in range(len(parts))}
        return bool(candidates & set(self.modules))

    def severity_for(self, check_id: str, reported: Severity) -> Severity:
        """Let a venue soften or harden a check without forking its code.

        ``PASS``/``SKIPPED`` are never rewritten — an override expresses how much
        a *failure* matters, not whether one happened.
        """
        if reported in (Severity.PASS, Severity.SKIPPED):
            return reported
        return self.severity_overrides.get(check_id, reported)

    def cfp_note(self, check_id: str) -> str | None:
        value = self.section("cfp").get(check_id)
        return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(name_or_path: str) -> tuple[dict[str, Any], str, str]:
    """Return (data, key, source) for a bundled profile name or a filesystem path."""
    candidate = Path(name_or_path).expanduser()
    if candidate.suffix in {".yaml", ".yml"} or candidate.exists():
        if not candidate.is_file():
            raise ProfileError(f"no profile file at {candidate}")
        data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        return data, candidate.stem, str(candidate)

    key = name_or_path.lower()
    for suffix in (".yaml", ".yml"):
        try:
            text = resources.files("preflight.conferences").joinpath(key + suffix).read_text("utf-8")
        except (FileNotFoundError, ModuleNotFoundError, IsADirectoryError):
            continue
        return (yaml.safe_load(text) or {}), key, f"<bundled:{key}{suffix}>"

    options = ", ".join(available_profiles()) or "none"
    raise ProfileError(f"unknown conference {name_or_path!r}; bundled profiles: {options}")


def _resolve(name_or_path: str, seen: tuple[str, ...] = ()) -> tuple[dict[str, Any], str, str, tuple[str, ...]]:
    data, key, source = _read_yaml(name_or_path)
    if key in seen:
        raise ProfileError(f"circular profile inheritance: {' -> '.join([*seen, key])}")
    lineage = (*seen, key)

    parent_name = (data.get("conference") or {}).get("extends") if isinstance(data.get("conference"), dict) else None
    if not parent_name:
        return data, key, source, lineage

    parent_data, _, _, parent_lineage = _resolve(str(parent_name), lineage)

    # `<key>_replace: true` drops the inherited value instead of merging into it,
    # so a venue can start a mapping (e.g. `tracks`) from scratch.
    trimmed_parent = {
        k: v for k, v in parent_data.items() if not data.get(f"{k}_replace")
    }
    merged = _deep_merge(trimmed_parent, data)

    # `modules` composes rather than replaces: a child extends its parent's set
    # unless it opts out explicitly with `modules_replace: true`.
    if not data.get("modules_replace"):
        merged["modules"] = _dedupe([*(parent_data.get("modules") or []), *(data.get("modules") or [])])
    return merged, key, source, parent_lineage


def _dedupe(items: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item)
        if text not in out:
            out.append(text)
    return out


def _parse(data: dict[str, Any], key: str, source: str, lineage: tuple[str, ...]) -> Profile:
    conf = data.get("conference")
    if not isinstance(conf, dict):
        raise ProfileError(f"{source}: missing top-level `conference:` mapping")

    raw_tracks = data.get("tracks")
    if not isinstance(raw_tracks, dict) or not raw_tracks:
        raise ProfileError(f"{source}: missing `tracks:` mapping")

    tracks: dict[str, TrackSpec] = {}
    for tname, tdata in raw_tracks.items():
        if not isinstance(tdata, dict) or "content_page_limit" not in tdata:
            raise ProfileError(f"{source}: track {tname!r} needs a `content_page_limit`")
        tracks[str(tname).lower()] = TrackSpec(
            name=str(tname).lower(),
            content_page_limit=int(tdata["content_page_limit"]),
            description=str(tdata.get("description", "")),
            total_page_limit=(
                int(tdata["total_page_limit"]) if tdata.get("total_page_limit") is not None else None
            ),
        )

    default_track = str(conf.get("default_track", next(iter(tracks)))).lower()
    if default_track not in tracks:
        raise ProfileError(f"{source}: default_track {default_track!r} is not a defined track")

    overrides: dict[str, Severity] = {}
    for cid, value in (data.get("severity") or {}).items():
        try:
            overrides[str(cid)] = Severity(str(value).lower())
        except ValueError as exc:
            allowed = ", ".join(s.value for s in Severity)
            raise ProfileError(f"{source}: bad severity {value!r} for {cid!r}; use one of {allowed}") from exc

    return Profile(
        key=str(conf.get("key", key)).lower(),
        name=str(conf.get("name", key)),
        source=source,
        tracks=tracks,
        default_track=default_track,
        modules=tuple(_dedupe(data.get("modules") or [])),
        severity_overrides=overrides,
        disabled_checks=frozenset(str(c) for c in (data.get("disabled_checks") or [])),
        lineage=lineage,
        raw=data,
    )


def load_profile(name_or_path: str) -> Profile:
    data, key, source, lineage = _resolve(name_or_path)
    return _parse(data, key, source, lineage)


def available_profiles(include_abstract: bool = False) -> list[str]:
    """Bundled profile keys. Abstract bases (``abstract: true``) are hidden by default."""
    try:
        root = resources.files("preflight.conferences")
    except ModuleNotFoundError:  # pragma: no cover
        return []
    out: list[str] = []
    for entry in root.iterdir():
        if not entry.name.endswith((".yaml", ".yml")):
            continue
        key = entry.name.rsplit(".", 1)[0]
        if not include_abstract:
            try:
                data = yaml.safe_load(entry.read_text("utf-8")) or {}
            except Exception:  # pragma: no cover
                data = {}
            if (data.get("conference") or {}).get("abstract"):
                continue
        out.append(key)
    return sorted(out)
