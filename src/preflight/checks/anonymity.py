"""Double-blind anonymity checks: title block, identifiers, links, and metadata.

Anonymity leaks are heuristic by nature -- a university name in related work is
perfectly legitimate -- so almost everything here is a WARNING that hands the
author the evidence and asks for a human decision. The two unambiguous cases,
a real email address in the title block and a forbidden tracking domain, are
allowed to be errors. Every list, threshold and toggle comes from the profile.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..context import CheckContext
from ..document import Line
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.anonymity"

CATEGORY = "anonymity"

_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dXx]\b")
# "Jane Q. Doe", "Jane Doe, John Smith" -- two to four capitalised tokens.
_NAME_RE = re.compile(
    r"^[A-Z][a-z'’-]+(?:\s+(?:[A-Z]\.|[A-Z][a-z'’-]+|van|von|de|del|der|di|da))"
    r"{1,3}$"
)

_DEFAULT_AFFILIATION_KEYWORDS = (
    "university", "universit", "institute", "institut", "college", "school of",
    "laboratory", "labs", "academy", "department of", "dept.", "faculty of",
    "research center", "research centre", "inc.", "corp.", "corporation",
    "gmbh", "ltd.", "llc", "s.r.l.", "a.s.", "co., ltd",
)
_DEFAULT_FORBIDDEN_URLS = (
    "dropbox.com", "bit.ly", "tinyurl.com", "goo.gl", "t.co/", "ow.ly",
    "rebrand.ly", "shorturl.at", "is.gd", "cutt.ly",
)
_DEFAULT_IDENTITY_URLS = (
    "github.com", "gitlab.com", "bitbucket.org", "drive.google.com",
    "docs.google.com", "sites.google.com", "colab.research.google.com",
    "huggingface.co", "zenodo.org", "figshare.com", "kaggle.com",
)
_DEFAULT_ANONYMOUS_URLS = ("anonymous.4open.science", "anonymous.science", "osf.io/anonymous")
_DEFAULT_SELF_REFERENCE = (
    r"in our (?:previous|prior|earlier) (?:work|paper|study)",
    r"our (?:previous|prior|earlier) (?:work|paper|study) \(",
    r"we (?:have )?(?:previously|earlier) (?:showed|shown|proposed|introduced|released)",
    r"as (?:we|i) (?:showed|argued|proposed) in \(",
    r"our (?:group|lab|team)'s (?:previous|prior|earlier)",
    r"building on our own",
)
_DEFAULT_METADATA_FIELDS = ("author", "title", "subject", "keywords", "creator", "producer")
_DEFAULT_PRODUCER_ALLOWLIST = (
    "pdftex", "pdflatex", "xetex", "luatex", "latex with hyperref", "tex live",
    "miktex", "ghostscript", "dvips", "dvipdfm", "acrobat distiller", "quartz",
    "libreoffice", "microsoft word", "word for", "skia/pdf", "chromium",
)
_DEFAULT_FILENAME_HINT = r"(?i)\.(?:tex|pdf|docx?|dvi)$|microsoft word -|[\\/]"


def _enabled(ctx: CheckContext) -> bool:
    return bool(ctx.conf("anonymity.enabled", True))


def _disabled_finding(ctx: CheckContext, check_id: str, title: str) -> Finding:
    return ctx.skip(check_id, title, "Anonymity checking is switched off in this profile.",
                    category=CATEGORY)


def _conf_list(ctx: CheckContext, path: str, default: Iterable[str]) -> list[str]:
    value = ctx.conf(path, None)
    if value is None:
        return [str(v) for v in default]
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _reference_pages(ctx: CheckContext) -> set[int]:
    """Pages occupied by the bibliography, whose URLs and emails belong to citations."""
    aliases = _conf_list(ctx, "anonymity.references_heading_aliases", ("references", "bibliography"))
    start = ctx.doc.find_heading(aliases)
    if start is None:
        return set()
    after = [h for h in ctx.doc.headings if h.order > start.order and h.page >= start.page]
    end = next((h for h in after if h.page > start.page or h.bbox[1] > start.bbox[1]), None)
    if end is None:
        return set(range(start.page, ctx.doc.page_count + 1))
    last = end.page
    if last > start.page and not any(
        ln.bbox[1] < end.bbox[1] - 1.0 and ln.text.strip()
        for ln in ctx.doc.pages[last - 1].lines
    ):
        last -= 1  # the following section starts at the very top of its page
    return set(range(start.page, last + 1))


def _title_block_lines(ctx: CheckContext) -> list[Line]:
    """Lines on page 1 above the abstract, where authors and affiliations live."""
    height = float(ctx.conf("anonymity.title_block_height_pt", 200.0))
    page = ctx.doc.pages[0]
    out: list[Line] = []
    for line in sorted(page.lines, key=lambda ln: ln.bbox[1]):
        if line.bbox[1] > height:
            continue
        text = " ".join(line.text.split())
        if not text or text.isdigit() or not any(c.isalpha() for c in text):
            continue
        if line.bbox[2] < page.width * 0.08 or line.bbox[0] > page.width * 0.92:
            continue  # the template's margin line numbers
        out.append(line)
    return out


def _looks_like_person(text: str, ctx: CheckContext) -> bool:
    max_words = int(ctx.conf("anonymity.name_max_words", 5))
    parts = [p.strip() for p in re.split(r",| and |&|·|\|", text) if p.strip()]
    if not parts or len(text.split()) > max_words * max(len(parts), 1):
        return False
    return all(_NAME_RE.match(p) and len(p.split()) <= max_words for p in parts)


@register("anonymity_title_block", "Anonymous title block", module=MODULE,
          category=CATEGORY, order=30)
def check_title_block(ctx: CheckContext) -> Finding:
    """Look for authors, affiliations, emails or ORCIDs in the page-1 title block."""
    if not _enabled(ctx):
        return _disabled_finding(ctx, "anonymity_title_block", "Anonymous title block")

    height = float(ctx.conf("anonymity.title_block_height_pt", 200.0))
    expected = ctx.conf("anonymity.expected_author_line", None)
    keywords = [k.lower() for k in _conf_list(ctx, "anonymity.affiliation_keywords",
                                              _DEFAULT_AFFILIATION_KEYWORDS)]
    lines = _title_block_lines(ctx)
    block_text = " ".join(" ".join(ln.text.split()) for ln in lines)

    anonymous_line: Line | None = None
    if expected:
        wanted = " ".join(str(expected).lower().split())
        anonymous_line = next(
            (ln for ln in lines if wanted in " ".join(ln.text.lower().split())), None
        )

    title_line = max(lines, key=lambda ln: ln.size, default=None)
    emails: list[Evidence] = []
    orcids: list[Evidence] = []
    suspects: list[Evidence] = []

    for line in lines:
        text = " ".join(line.text.split())
        for m in _EMAIL_RE.finditer(text):
            emails.append(Evidence(page=1, detail="email address in the title block",
                                   quote=m.group(0), measured=line.bbox[1],
                                   expected=f"nothing identifying above y={height:.0f} pt",
                                   bbox=line.bbox))
        for m in _ORCID_RE.finditer(text):
            orcids.append(Evidence(page=1, detail="ORCID in the title block", quote=m.group(0),
                                   bbox=line.bbox))
        if line is title_line or line is anonymous_line:
            continue
        low = text.lower()
        hit = next((k for k in keywords if k in low), None)
        if hit:
            suspects.append(Evidence(page=1, detail=f"affiliation-like keyword {hit!r}",
                                     quote=text, measured=line.bbox[1], bbox=line.bbox))
        elif _looks_like_person(text, ctx):
            suspects.append(Evidence(page=1, detail="line reads like an author name",
                                     quote=text, measured=line.bbox[1], bbox=line.bbox))

    if emails:
        return ctx.error(
            "anonymity_title_block", "Anonymous title block",
            f"{len(emails)} email address(es) appear in the title block of page 1. "
            "A submission carrying author emails is not anonymous.",
            category=CATEGORY, evidence=(emails + orcids + suspects)[:10],
            remedy="Rebuild with the anonymous option of the official style "
            "(e.g. \\usepackage[review]{acl}) so the author block is replaced.",
            confidence="high — an email address in the title block is unambiguous",
            cfp_key="anonymity",
        )

    if anonymous_line is not None and not orcids and not suspects:
        return ctx.ok(
            "anonymity_title_block", "Anonymous title block",
            f"The title block carries the anonymous placeholder {str(expected)!r} and no emails, "
            "ORCIDs, names or affiliations.",
            category=CATEGORY,
            evidence=[Evidence(page=1, detail="anonymous author line",
                               quote=" ".join(anonymous_line.text.split()),
                               measured=anonymous_line.bbox[1], bbox=anonymous_line.bbox)],
            cfp_key="anonymity",
        )

    if orcids or suspects:
        anon_note = (
            f" The expected placeholder {str(expected)!r} is present, which argues the template is "
            "in review mode, so these may be false positives."
            if anonymous_line is not None else ""
        )
        return ctx.warn(
            "anonymity_title_block", "Anonymous title block",
            f"{len(orcids) + len(suspects)} line(s) in the top {height:.0f} pt of page 1 look like "
            f"author or affiliation content. Potential leak — review manually.{anon_note}",
            category=CATEGORY, evidence=(orcids + suspects)[:10],
            remedy="Remove author names, affiliations, ORCIDs and acknowledgements from the "
            "submission version, or rebuild with the anonymous style option.",
            confidence=("low — the anonymous placeholder is present" if anonymous_line is not None
                        else "medium — capitalised names and institution keywords are heuristics"),
            cfp_key="anonymity",
        )

    if expected and anonymous_line is None:
        return ctx.warn(
            "anonymity_title_block", "Anonymous title block",
            f"No author or affiliation content was found in the top {height:.0f} pt of page 1, but "
            f"the expected placeholder {str(expected)!r} is missing either. Confirm the paper was "
            "built with the anonymous style option.",
            category=CATEGORY,
            evidence=[Evidence(page=1, detail="title-block text", quote=block_text[:200])],
            confidence="medium — the placeholder wording varies between template versions",
            cfp_key="anonymity",
        )

    return ctx.ok(
        "anonymity_title_block", "Anonymous title block",
        f"No emails, ORCIDs, author names or affiliations in the top {height:.0f} pt of page 1.",
        category=CATEGORY, cfp_key="anonymity",
    )


@register("anonymity_emails", "Email addresses", module=MODULE, category=CATEGORY, order=31)
def check_emails(ctx: CheckContext) -> Finding:
    """Email addresses anywhere outside the bibliography are a likely identity leak."""
    if not _enabled(ctx):
        return _disabled_finding(ctx, "anonymity_emails", "Email addresses")

    ignore = [re.compile(p, re.IGNORECASE) for p in _conf_list(ctx, "anonymity.email_ignore_regexes", ())]
    ref_pages = _reference_pages(ctx)
    seen: set[str] = set()
    hits: list[Evidence] = []
    for page, address in ctx.doc.emails:
        if page in ref_pages or address.lower() in seen:
            continue
        if any(p.search(address) for p in ignore):
            continue
        seen.add(address.lower())
        hits.append(Evidence(page=page, detail="email address in the body text", quote=address))

    if not hits:
        skipped = f" ({len(ref_pages)} reference page(s) excluded)" if ref_pages else ""
        return ctx.ok("anonymity_emails", "Email addresses",
                      f"No email addresses outside the references{skipped}.",
                      category=CATEGORY, cfp_key="anonymity")

    return ctx.warn(
        "anonymity_emails", "Email addresses",
        f"{len(hits)} email address(es) appear outside the references. Potential leak — "
        "review manually; contact addresses and dataset-request addresses both show up this way.",
        category=CATEGORY, evidence=hits[:10],
        remedy="Remove contact addresses from the submission version; add them back for camera-ready.",
        confidence="high for personal addresses, medium for addresses quoted from datasets or prompts",
        cfp_key="anonymity",
    )


@register("anonymity_orcids", "ORCID identifiers", module=MODULE, category=CATEGORY, order=32)
def check_orcids(ctx: CheckContext) -> Finding:
    """An ORCID resolves to a named researcher, so any occurrence is a leak candidate."""
    if not _enabled(ctx):
        return _disabled_finding(ctx, "anonymity_orcids", "ORCID identifiers")

    seen: set[str] = set()
    hits: list[Evidence] = []
    for page, orcid in ctx.doc.orcids:
        if orcid in seen:
            continue
        seen.add(orcid)
        hits.append(Evidence(page=page, detail="ORCID identifier", quote=orcid))

    if not hits:
        return ctx.ok("anonymity_orcids", "ORCID identifiers",
                      "No ORCID identifiers anywhere in the document.",
                      category=CATEGORY, cfp_key="anonymity")

    return ctx.warn(
        "anonymity_orcids", "ORCID identifiers",
        f"{len(hits)} ORCID identifier(s) appear in the document; each resolves to a named "
        "researcher. Potential leak — review manually.",
        category=CATEGORY, evidence=hits[:10],
        remedy="Strip ORCIDs from the submission version.",
        confidence="high — the identifier format is unambiguous, but it may belong to a cited author",
        cfp_key="anonymity",
    )


def _normalise_url(url: str) -> str:
    cleaned = url.strip().rstrip(".,;:)]}’\"'")
    return re.sub(r"(?i)^https?://(?:www\.)?", "", cleaned).lower()


def _url_inventory(ctx: CheckContext) -> list[tuple[int, str, str]]:
    """(page, source, url) for every link annotation and textual URL, deduplicated."""
    out: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for page, uri in ctx.doc.hyperlinks:
        key = _normalise_url(uri)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((page, "link annotation", uri.strip()))
    for page, url in ctx.doc.textual_urls:
        key = _normalise_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((page, "printed URL", url.strip()))
    return out


@register("anonymity_urls", "Repository and hosting links", module=MODULE,
          category=CATEGORY, order=33)
def check_urls(ctx: CheckContext) -> list[Finding]:
    """Classify every link against the profile's forbidden, identity and anonymous lists."""
    if not _enabled(ctx):
        return [_disabled_finding(ctx, "anonymity_urls", "Repository and hosting links")]

    forbidden = [d.lower() for d in _conf_list(ctx, "anonymity.forbidden_url_domains",
                                               _DEFAULT_FORBIDDEN_URLS)]
    identity = [d.lower() for d in _conf_list(ctx, "anonymity.identity_url_domains",
                                              _DEFAULT_IDENTITY_URLS)]
    anonymous = [d.lower() for d in _conf_list(ctx, "anonymity.anonymous_url_domains",
                                               _DEFAULT_ANONYMOUS_URLS)]
    skip_refs = bool(ctx.conf("anonymity.ignore_urls_in_references", True))
    ref_pages = _reference_pages(ctx) if skip_refs else set()

    bad: list[Evidence] = []
    suspect: list[Evidence] = []
    good: list[Evidence] = []
    for page, source, url in _url_inventory(ctx):
        key = _normalise_url(url)
        anon_hit = next((d for d in anonymous if d in key), None)
        if anon_hit:
            good.append(Evidence(page=page, detail=f"anonymised host {anon_hit} ({source})", quote=url))
            continue
        forbidden_hit = next((d for d in forbidden if d in key), None)
        if forbidden_hit:
            bad.append(Evidence(page=page, detail=f"forbidden domain {forbidden_hit} ({source})",
                                quote=url))
            continue
        if page in ref_pages:
            continue  # citations legitimately point at repositories
        identity_hit = next((d for d in identity if d in key), None)
        if identity_hit:
            suspect.append(Evidence(page=page, detail=f"identity-bearing host {identity_hit} ({source})",
                                    quote=url))

    findings: list[Finding] = []
    if bad:
        findings.append(ctx.error(
            "anonymity_urls", "Repository and hosting links",
            f"{len(bad)} link(s) point at hosting or link-shortening services the CFP forbids in "
            "anonymous submissions because they can track who opens them.",
            category=CATEGORY, evidence=bad[:10],
            remedy="Replace with an anonymised mirror (for example anonymous.4open.science) or "
            "drop the link until camera-ready.",
            confidence="high — the domain is on the profile's forbidden list",
            cfp_key="anonymity",
        ))
    if suspect:
        pages = sorted({e.page for e in suspect if e.page})
        findings.append(ctx.warn(
            "anonymity_urls_identity", "Code and data links",
            f"{len(suspect)} link(s) outside the references point at hosts that usually carry an "
            f"account name (page(s) {', '.join(str(p) for p in pages[:8])}). Potential leak — "
            "verify each target is anonymized.",
            category=CATEGORY, evidence=suspect[:10],
            remedy="Serve code and data from an anonymous mirror for the review period.",
            confidence="medium — such links are allowed if the account and content are anonymous",
            cfp_key="anonymity",
        ))
    if good:
        findings.append(ctx.ok(
            "anonymity_urls_anonymous", "Anonymised artifact links",
            f"{len(good)} link(s) use an anonymised host, which is exactly what the CFP asks for.",
            category=CATEGORY, evidence=good[:6], cfp_key="anonymity",
        ))
    if not findings:
        total = len(_url_inventory(ctx))
        findings.append(ctx.ok(
            "anonymity_urls", "Repository and hosting links",
            f"None of the {total} distinct link(s) outside the references point at a forbidden or "
            "identity-bearing host.",
            category=CATEGORY, cfp_key="anonymity",
        ))
    return findings


@register("anonymity_self_reference", "Self-referential phrasing", module=MODULE,
          category=CATEGORY, order=34)
def check_self_reference(ctx: CheckContext) -> Finding:
    """Phrases such as "in our previous work" identify the authors by citation."""
    if not _enabled(ctx):
        return _disabled_finding(ctx, "anonymity_self_reference", "Self-referential phrasing")

    patterns = _conf_list(ctx, "anonymity.self_reference_patterns", _DEFAULT_SELF_REFERENCE)
    window = int(ctx.conf("anonymity.self_reference_context_chars", 120))
    cap = int(ctx.conf("anonymity.max_reported_self_references", 8))

    text = " ".join(ctx.doc.text.split())
    hits: list[Evidence] = []
    spans: list[tuple[int, int]] = []
    for raw in patterns:
        try:
            pattern = re.compile(raw, re.IGNORECASE)
        except re.error:
            continue
        for m in pattern.finditer(text):
            if any(m.start() < b and a < m.end() for a, b in spans):
                continue  # two profile patterns matched the same sentence
            spans.append((m.start(), m.end()))
            start = max(0, m.start() - window // 2)
            hits.append(Evidence(detail=f"matches {raw!r}",
                                 quote=text[start : m.end() + window // 2]))

    if not hits:
        return ctx.ok(
            "anonymity_self_reference", "Self-referential phrasing",
            f"None of the {len(patterns)} self-reference patterns from the profile occur in the text.",
            category=CATEGORY, cfp_key="anonymity",
        )
    return ctx.warn(
        "anonymity_self_reference", "Self-referential phrasing",
        f"{len(hits)} phrase(s) refer to the authors' own earlier work in the first person, which "
        "de-anonymises the submission through the cited paper. Potential leak — review manually.",
        category=CATEGORY, evidence=hits[:cap],
        remedy="Cite your own prior work in the third person, e.g. \"Smith et al. (2024) showed\".",
        confidence="medium — the phrasing sometimes refers to work described earlier in this paper",
        cfp_key="anonymity",
    )


def _metadata_leak(field: str, value: str, ctx: CheckContext, allow: list[str]) -> str | None:
    """Reason this metadata value looks like a human, or None if it is innocuous."""
    text = value.strip()
    if not text:
        return None
    if field in {"creator", "producer"}:
        low = text.lower()
        return None if any(a in low for a in allow) else "names a tool the profile does not recognise"
    if field == "title":
        if re.search(str(ctx.conf("anonymity.metadata_filename_regex", _DEFAULT_FILENAME_HINT)), text):
            return "looks like a source filename or authoring-tool path"
        return "names a person" if _looks_like_person(text, ctx) else None
    if field == "author":
        return "carries an author name" if len(text) > 1 else None
    return "is non-empty and may carry identity"


@register("anonymity_metadata", "PDF metadata and annotations", module=MODULE,
          category=CATEGORY, order=35)
def check_metadata(ctx: CheckContext) -> Finding:
    """Inspect document metadata, comment annotations and embedded files for names."""
    if not _enabled(ctx):
        return _disabled_finding(ctx, "anonymity_metadata", "PDF metadata and annotations")

    fields = [f.lower() for f in _conf_list(ctx, "anonymity.metadata_fields", _DEFAULT_METADATA_FIELDS)]
    allow = [a.lower() for a in _conf_list(ctx, "anonymity.metadata_producer_allowlist",
                                           _DEFAULT_PRODUCER_ALLOWLIST)]
    meta = ctx.doc.metadata

    hits: list[Evidence] = []
    for field in fields:
        value = str(meta.get(field, "") or "")
        reason = _metadata_leak(field, value, ctx, allow)
        if reason:
            hits.append(Evidence(detail=f"/{field.capitalize()} {reason}", quote=value))

    for page, kind, content in ctx.doc.annotations:
        hits.append(Evidence(page=page, detail=f"{kind} annotation left in the PDF", quote=content))
    for name in ctx.doc.embedded_files:
        hits.append(Evidence(detail="embedded file", quote=name))

    if not hits:
        producer = str(meta.get("producer", "") or "").strip() or "unset"
        creator = str(meta.get("creator", "") or "").strip() or "unset"
        return ctx.ok(
            "anonymity_metadata", "PDF metadata and annotations",
            "Document metadata carries no author identity, and there are no annotations or "
            "embedded files.",
            category=CATEGORY,
            evidence=[Evidence(detail=f"/Producer {producer!r}, /Creator {creator!r} — "
                               "recognised as build tooling, not a person")],
            cfp_key="anonymity",
        )
    return ctx.warn(
        "anonymity_metadata", "PDF metadata and annotations",
        f"{len(hits)} metadata field(s), annotation(s) or embedded file(s) may identify the "
        "authors. Potential leak — review manually.",
        category=CATEGORY, evidence=hits[:10],
        remedy="Clear the document properties before submitting (pdftex writes no /Author unless "
        "\\pdfinfo or hyperref's pdfauthor sets one) and flatten or delete PDF comments.",
        confidence="medium — metadata often names software or the paper title rather than a person",
        cfp_key="anonymity",
    )
