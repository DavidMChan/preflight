"""PDF-container health checks: blank pages, font embedding, structural integrity.

Conversion to PDF can silently disrupt content -- a page can render as text with
no glyphs painted, a font can fail to embed and reflow on the reviewer's
machine, or the producer's PDF writer can leave the file in a state a strict
publisher pipeline rejects. None of this is a documented desk-rejection
condition on its own, so every finding here is a warning: it is deterministic
measurement, not a CFP rule, and a venue that wants one of these promoted can
do so through its profile's ``severity:`` map.
"""

from __future__ import annotations

import re

from ..context import CheckContext
from ..document import PageInfo
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.pdf"

# The 14 standard fonts every PDF-consuming application ships, which are
# legitimately never embedded. Names as they appear in /BaseFont, minus any
# subset-tag prefix.
_DEFAULT_BASE14 = [
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Symbol", "ZapfDingbats",
]


def _ignore_pattern(ctx: CheckContext) -> re.Pattern[str]:
    return re.compile(str(ctx.conf("fonts.ignore_small_text_regex", r"^\s*\d{1,4}\s*$")))


@register("blank_pages", "Blank pages", module=MODULE, category="format", order=90)
def check_blank_pages(ctx: CheckContext) -> Finding:
    """Find pages with no text, no images and no vector drawings.

    A page holding only a full-page figure is not blank -- it has an image. A
    page carrying only a running line number or folio is effectively blank to
    a reader, so that residual text is exempted rather than saving the page.
    """
    ignore = _ignore_pattern(ctx)
    blank: list[Evidence] = []

    for page in ctx.doc.pages:
        if page.images or page.drawings_bbox:
            continue
        texts = [ln.text.strip() for ln in page.lines if ln.text.strip()]
        if texts and not all(ignore.match(t) for t in texts):
            continue  # real content on the page
        if texts:
            detail = f"only page/line-number text remains: {', '.join(texts)[:60]}"
        else:
            detail = "no text, images or drawings"
        blank.append(Evidence(page=page.number, detail=detail))

    if blank:
        pages = ", ".join(str(e.page) for e in blank)
        return ctx.warn(
            "blank_pages",
            "Blank pages",
            f"{len(blank)} page(s) appear blank: {pages}. This is often an intentional "
            "page break but can also be a PDF-conversion artifact that dropped content.",
            category="format",
            evidence=blank[:10],
            remedy="Open these pages directly in the PDF to confirm nothing was meant to be there "
            "(a dropped float, a stray \\clearpage, or a figure that failed to render).",
            confidence="medium — a deliberately empty page (e.g. before an appendix) looks identical",
        )
    return ctx.ok(
        "blank_pages",
        "Blank pages",
        f"No blank pages found across {ctx.doc.page_count} page(s).",
        category="format",
    )


@register("font_embedding", "Font embedding", module=MODULE, category="format", order=91)
def check_font_embedding(ctx: CheckContext) -> Finding:
    """Every non-standard font referenced by the PDF should carry its own program.

    An unembedded font renders with a substitute on any machine that lacks it,
    which can shift line breaks, hide glyphs, or trip a publisher's ingestion
    pipeline. Type 3 glyphs live in PDF content streams (CharProcs), so there is
    no separate font file to embed. They are normally exempt, but profiles for
    publishers such as IEEE/PaperCept can prohibit them explicitly.
    """
    base14 = {str(s).lower() for s in (ctx.conf("pdf.base14_fonts", _DEFAULT_BASE14) or [])}
    require_base14 = bool(ctx.conf("pdf.require_base14_embedding", False))
    forbid_type3 = bool(ctx.conf("pdf.forbid_type3_fonts", False))
    raw = ctx.doc.doc

    missing: dict[str, set[int]] = {}
    type3: dict[str, set[int]] = {}
    seen: dict[str, str] = {}  # simple name -> subtype, for the report
    all_fonts: dict[str, set[int]] = {}

    for page in ctx.doc.pages:
        try:
            fonts = raw[page.number - 1].get_fonts(full=True)
        except Exception:  # pragma: no cover - defensive
            continue
        for item in fonts:
            xref, _ext, subtype, basefont, *_rest = item
            if subtype == "Type3":
                if forbid_type3:
                    type3.setdefault(basefont or "unnamed Type 3 font", set()).add(page.number)
                continue  # self-contained; nothing external to embed
            simple = basefont.split("+", 1)[-1] if "+" in basefont else basefont
            if not simple or (simple.lower() in base14 and not require_base14):
                continue  # base-14 standard font, legitimately unembedded
            all_fonts.setdefault(simple, set()).add(page.number)
            seen[simple] = subtype
            try:
                buf = raw.extract_font(xref)
                embedded = bool(buf and len(buf) > 3 and buf[3])
            except Exception:  # pragma: no cover - defensive
                embedded = True  # cannot prove absence; don't false-positive
            if not embedded:
                missing.setdefault(simple, set()).add(page.number)

    if missing or type3:
        evidence = [
            Evidence(
                detail=f"font {name!r} ({seen.get(name, '?')}) is not embedded",
                page=min(pages),
                expected="embedded font program",
            )
            for name, pages in sorted(missing.items())
        ]
        evidence.extend(
            Evidence(
                detail=f"font {name!r} is a forbidden Type 3 bitmap font",
                page=min(pages),
                expected="Type 1, TrueType, or another scalable embedded font",
            )
            for name, pages in sorted(type3.items())
        )
        pages_all = sorted({p for pages in missing.values() for p in pages})
        pages_all = sorted({*pages_all, *(p for pages in type3.values() for p in pages)})
        issues = []
        if missing:
            issues.append(f"{len(missing)} unembedded font(s)")
        if type3:
            issues.append(f"{len(type3)} Type 3 font(s)")
        return ctx.warn(
            "font_embedding",
            "Font embedding",
            f"{' and '.join(issues)} found on "
            f"page(s) {', '.join(str(p) for p in pages_all[:10])}"
            f"{'...' if len(pages_all) > 10 else ''}.",
            category="format",
            evidence=evidence[:10],
            remedy="Regenerate the PDF with scalable fonts and font embedding enabled "
            "(pdflatex: use Type 1/OpenType fonts and embed with pdftex.map/-dEmbedAllFonts, "
            "or use PaperCept's compliant conversion).",
            confidence="high — checked via the PDF's own font descriptor and extract_font()",
            cfp_key="pdf_fonts",
        )
    if not all_fonts:
        return ctx.skip(
            "font_embedding", "Font embedding",
            "No non-standard fonts referenced (Type 3 or base-14 only).",
            category="format",
        )
    return ctx.ok(
        "font_embedding",
        "Font embedding",
        f"All {len(all_fonts)} non-standard font(s) referenced in the document are embedded.",
        category="format",
        cfp_key="pdf_fonts",
    )


@register("pdf_health", "PDF integrity", module=MODULE, category="format", order=92)
def check_pdf_health(ctx: CheckContext) -> Finding:
    """Cheap structural facts about the PDF container itself."""
    doc = ctx.doc.doc
    problems: list[Evidence] = []

    if ctx.doc.page_count == 0:
        problems.append(Evidence(detail="the PDF has zero pages"))

    if ctx.doc.is_encrypted:
        problems.append(Evidence(detail="the PDF is encrypted"))
        required = int(ctx.conf("pdf.required_permission_bits", 60))  # PRINT|MODIFY|COPY|ANNOTATE
        perms = int(doc.permissions)
        if (perms & required) != required:
            problems.append(
                Evidence(detail="restrictive permissions on an encrypted file",
                          measured=perms, expected=f"bitmask including {required}")
            )

    if bool(getattr(doc, "is_repaired", False)):
        problems.append(Evidence(detail="PyMuPDF had to repair the file structure on open; "
                                        "the source PDF is malformed"))

    version = str(ctx.doc.metadata.get("format") or "unknown")
    minimum_version = ctx.conf("pdf.minimum_version", None)
    if minimum_version is not None:
        match = re.search(r"(\d+(?:\.\d+)?)", version)
        if match and tuple(int(p) for p in match.group(1).split(".")) < tuple(
            int(p) for p in str(minimum_version).split(".")
        ):
            problems.append(Evidence(detail=f"PDF version {match.group(1)} is too old",
                                     measured=float(match.group(1)),
                                     expected=f">= PDF {minimum_version}"))

    if bool(ctx.conf("pdf.forbid_hyperlinks", False)) and ctx.doc.hyperlinks:
        for page, uri in ctx.doc.hyperlinks[:8]:
            problems.append(Evidence(page=page, detail="embedded hyperlink annotation", quote=uri,
                                     expected="printed URL without an embedded link"))

    if bool(ctx.conf("pdf.forbid_bookmarks", False)):
        try:
            bookmarks = doc.get_toc(simple=True)
        except Exception:  # pragma: no cover - malformed outline tree
            bookmarks = []
        if bookmarks:
            problems.append(Evidence(detail=f"{len(bookmarks)} PDF bookmark(s) found",
                                     expected="no document bookmarks"))

    zero_area: list[PageInfo] = [p for p in ctx.doc.pages if p.width <= 0 or p.height <= 0]
    for p in zero_area:
        problems.append(Evidence(page=p.number, detail="zero-area MediaBox",
                                 measured=p.width * p.height, expected="> 0"))

    producer = str(ctx.doc.metadata.get("producer") or "unknown")

    if problems:
        return ctx.warn(
            "pdf_health",
            "PDF integrity",
            f"{len(problems)} structural issue(s) found in the PDF container.",
            category="format",
            evidence=problems[:10],
            remedy="Regenerate the PDF from source rather than editing the container directly; "
            "remove encryption, embedded links and bookmarks, and use the PDF version required by "
            "the publisher's ingestion pipeline.",
            cfp_key="pdf_health",
        )
    return ctx.ok(
        "pdf_health",
        "PDF integrity",
        f"{ctx.doc.page_count} pages, {version}, not encrypted, structurally sound "
        f"(producer: {producer}).",
        category="format",
        evidence=[Evidence(detail=f"pages={ctx.doc.page_count}, format={version}, "
                                   f"encrypted={ctx.doc.is_encrypted}, producer={producer!r}")],
        cfp_key="pdf_health",
    )
