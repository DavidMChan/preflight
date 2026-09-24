"""Verifiers for the citations that are not papers.

A bibliography routinely cites a GitHub repository, a model card, a vendor
announcement or an RFC. Asking CrossRef about those and then reporting "not
found in any bibliographic database" is how a checker manufactures an
accusation, so each of these gets a verifier that actually addresses it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote, urlparse

from .core import Candidate, Reference
from .sources import Session

_GITHUB_RE = re.compile(r"github\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_GITLAB_RE = re.compile(r"gitlab\.com/([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_HF_RE = re.compile(r"huggingface\.co/(?:(datasets|models|spaces)/)?([\w.-]+)/([\w.-]+)", re.IGNORECASE)
_PYPI_RE = re.compile(r"pypi\.org/project/([\w.-]+)", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE)


@dataclass(slots=True)
class Resolution:
    """What a web verifier concluded about a non-paper citation."""

    ok: bool
    detail: str
    url: str | None = None
    title: str | None = None
    source: str = "web"
    # The host refused us (403/405/429): the path is served, but nothing on it was read.
    blocked: bool = False


def _strip(name: str) -> str:
    return name.rstrip("/.,;)]}").removesuffix(".git")


async def github(session: Session, url: str) -> Resolution | None:
    match = _GITHUB_RE.search(url)
    if not match:
        return None
    owner, repo = match.group(1), _strip(match.group(2))
    headers = {"Accept": "application/vnd.github+json"}
    if session.github_token:
        headers["Authorization"] = f"Bearer {session.github_token}"
    data = await session.json(f"https://api.github.com/repos/{owner}/{repo}", rate=1.0, headers=headers)
    if isinstance(data, dict) and data.get("full_name"):
        stars = data.get("stargazers_count")
        detail = f"repository exists ({stars} stars)" if stars is not None else "repository exists"
        if data.get("archived"):
            detail += ", archived"
        return Resolution(True, detail, url=data.get("html_url"), title=data.get("full_name"),
                          source="github")
    # The API rate-limits aggressively unauthenticated; fall back to the page.
    page = await session.get(f"https://github.com/{owner}/{repo}", rate=2.0, method="HEAD")
    if page is not None and page.status_code < 400:
        return Resolution(True, "repository page reachable", url=str(page.url),
                          title=f"{owner}/{repo}", source="github")
    if page is not None and page.status_code == 404:
        return Resolution(False, "repository does not exist (404)", url=url, source="github")
    return None


async def gitlab(session: Session, url: str) -> Resolution | None:
    match = _GITLAB_RE.search(url)
    if not match:
        return None
    project = quote(f"{match.group(1)}/{_strip(match.group(2))}", safe="")
    data = await session.json(f"https://gitlab.com/api/v4/projects/{project}", rate=2.0)
    if isinstance(data, dict) and data.get("path_with_namespace"):
        return Resolution(True, "repository exists", url=data.get("web_url"),
                          title=data.get("path_with_namespace"), source="gitlab")
    return None


async def huggingface(session: Session, url: str) -> Resolution | None:
    match = _HF_RE.search(url)
    if not match:
        return None
    kind, owner, name = match.group(1), match.group(2), _strip(match.group(3))
    repo_id = f"{owner}/{name}"
    endpoints = ["models", "datasets"] if not kind else [kind if kind != "models" else "models"]
    for endpoint in endpoints:
        data = await session.json(f"https://huggingface.co/api/{endpoint}/{repo_id}", rate=4.0)
        if isinstance(data, dict) and (data.get("id") or data.get("modelId")):
            downloads = data.get("downloads")
            noun = endpoint.rstrip("s")
            detail = f"{noun} exists on the Hub"
            if downloads:
                detail += f" ({downloads:,} downloads)"
            return Resolution(True, detail, url=f"https://huggingface.co/{repo_id}",
                              title=repo_id, source="huggingface")
    return Resolution(False, "no such model or dataset on the Hub", url=url, source="huggingface")


async def pypi(session: Session, url: str) -> Resolution | None:
    match = _PYPI_RE.search(url)
    if not match:
        return None
    name = _strip(match.group(1))
    data = await session.json(f"https://pypi.org/pypi/{name}/json", rate=4.0)
    if isinstance(data, dict) and (data.get("info") or {}).get("name"):
        version = (data.get("info") or {}).get("version")
        return Resolution(True, f"package exists (latest {version})",
                          url=f"https://pypi.org/project/{name}/", title=name, source="pypi")
    return Resolution(False, "no such package on PyPI", url=url, source="pypi")


async def liveness(session: Session, url: str) -> Resolution | None:
    """Last resort: does the cited URL actually resolve?

    A 403 or 405 still tells us the host is serving that path — plenty of sites
    refuse HEAD or block non-browser agents — so only an outright 404/410 counts
    as absent.
    """
    if not url:
        return None
    target = url if url.startswith("http") else f"https://{url}"
    permitted, delay = await session.allowed(target)
    if not permitted:
        return None             # the site's robots.txt asks tools like this one to stay out
    rate = 1.0 / delay if delay else 4.0
    response = await session.get(target, rate=rate, method="HEAD")
    if response is None:
        response = await session.get(target, rate=rate)
    if response is None:
        return None
    host = urlparse(str(response.url)).netloc
    if response.status_code < 400:
        return Resolution(True, f"URL resolves ({host})", url=str(response.url), source="url")
    if response.status_code in (403, 405, 429):
        return Resolution(True, f"URL exists but blocked automated access ({response.status_code})",
                          url=target, source="url", blocked=True)
    if response.status_code in (404, 410):
        return Resolution(False, f"URL returns {response.status_code}", url=target, source="url")
    return None


_ROUTES = (github, gitlab, huggingface, pypi)


async def resolve_web(session: Session, reference: Reference) -> Resolution | None:
    """Verify a non-paper citation by whatever route actually fits it."""
    url = reference.url or ""
    arxiv = _ARXIV_URL_RE.search(url)
    if arxiv and not reference.arxiv_id:
        reference.arxiv_id = arxiv.group(1)

    for route in _ROUTES:
        result = await route(session, url)
        if result is not None:
            return result
    return await liveness(session, url)


def as_candidate(resolution: Resolution, reference: Reference) -> Candidate:
    """Wrap a web resolution so it can be reported like any other match."""
    return Candidate(
        source=resolution.source,
        title=resolution.title or reference.label,
        year=reference.year,
        url=resolution.url,
        exact_id=True,
    )


def needs_web_route(reference: Reference) -> bool:
    return reference.kind.has_web_route
