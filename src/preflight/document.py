"""Deterministic PDF layout analysis.

Every geometric, typographic and structural check reads from this module rather
than touching PyMuPDF directly, so findings can quote exact measurements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pymupdf

BBox = tuple[float, float, float, float]

# PyMuPDF span flag bits.
_FLAG_SUPERSCRIPT = 1 << 0
_FLAG_ITALIC = 1 << 1
_FLAG_SERIF = 1 << 2
_FLAG_MONO = 1 << 3
_FLAG_BOLD = 1 << 4

# PDF text rendering modes that put no marks on the page.
INVISIBLE_RENDER_MODES = {3, 7}

_URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s<>\"'\)\]}]+|"
    r"\b[a-z0-9][a-z0-9._-]*\.(?:com|org|net|io|edu|ai|dev|co|science|site|me)/[^\s<>\"'\)\]}]*"
)
_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dXx]\b")


@dataclass(slots=True)
class Span:
    """A run of characters sharing one font, size and colour."""

    text: str
    bbox: BBox
    font: str
    size: float
    color: int
    flags: int
    page: int
    render_mode: int = 0
    opacity: float = 1.0

    @property
    def bold(self) -> bool:
        return bool(self.flags & _FLAG_BOLD) or "bold" in self.font.lower()

    @property
    def italic(self) -> bool:
        return bool(self.flags & _FLAG_ITALIC)

    @property
    def rgb(self) -> tuple[float, float, float]:
        c = self.color
        return (((c >> 16) & 0xFF) / 255.0, ((c >> 8) & 0xFF) / 255.0, (c & 0xFF) / 255.0)

    @property
    def luminance(self) -> float:
        r, g, b = self.rgb
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @property
    def invisible(self) -> bool:
        return self.render_mode in INVISIBLE_RENDER_MODES or self.opacity <= 0.01

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass(slots=True)
class Line:
    """A visual line: spans plus the column it was assigned to."""

    spans: list[Span]
    bbox: BBox
    page: int
    column: int = 0

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.spans)

    @property
    def size(self) -> float:
        """Size of the widest span — robust to trailing footnote markers."""
        if not self.spans:
            return 0.0
        return max(self.spans, key=lambda s: s.width).size

    @property
    def heading_size(self) -> float:
        """The size a reader perceives the line at.

        Small-caps headings (the ICLR and NeurIPS templates set every heading
        this way) reach the extractor as a full-size initial followed by a
        smaller run of capitals: "R" at 12pt, "EFERENCES" at 9.6pt. The widest
        span is then the small one, and ``size`` under-reports the heading. When
        every letter on the line is a capital and the line mixes sizes, the
        largest span is the one the eye measures.
        """
        if not self.small_caps:
            return self.size
        return max(s.size for s in self.spans if any(c.isalpha() for c in s.text))

    @property
    def small_caps(self) -> bool:
        """True if the line is set in small capitals: all capitals, two sizes."""
        lettered = [s for s in self.spans if any(c.isalpha() for c in s.text)]
        if len(lettered) < 2:
            return False
        letters = "".join(c for s in lettered for c in s.text if c.isalpha())
        if not letters.isupper():
            return False
        return len({round(s.size, 1) for s in lettered}) >= 2

    @property
    def bold(self) -> bool:
        visible = [s for s in self.spans if s.text.strip()]
        if not visible:
            return False
        return sum(s.width for s in visible if s.bold) > 0.6 * sum(s.width for s in visible)


@dataclass(slots=True)
class Heading:
    """A detected section heading, in reading order across the document."""

    text: str
    page: int
    bbox: BBox
    size: float
    order: int
    numbering: str | None = None
    kind: str = "section"

    @property
    def normalized(self) -> str:
        return re.sub(r"[^a-z ]", "", self.text.lower()).strip()


@dataclass(slots=True)
class PageInfo:
    number: int                      # 1-based
    width: float
    height: float
    mediabox: BBox
    cropbox: BBox
    lines: list[Line] = field(default_factory=list)
    images: list[BBox] = field(default_factory=list)
    drawings_bbox: list[BBox] = field(default_factory=list)
    column_bounds: list[tuple[float, float]] = field(default_factory=list)

    @property
    def spans(self) -> list[Span]:
        return [s for ln in self.lines for s in ln.spans]

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    def inside_graphic(self, bbox: BBox, slack: float = 2.0) -> bool:
        """True if an image or filled drawing encloses ``bbox``.

        Text baked into a figure is exempt from body typography rules: a 2.6pt
        Type 3 label inside a screenshot is a bad figure, not a font violation.
        """
        return any(covers(box, bbox, slack) for box in self.images) or any(
            covers(box, bbox, slack) for box in self.drawings_bbox
        )

    @property
    def image_coverage(self) -> float:
        """Largest single image's share of the page — a scan approaches 1.0."""
        area = self.width * self.height
        if area <= 0 or not self.images:
            return 0.0
        return max(((b[2] - b[0]) * (b[3] - b[1])) / area for b in self.images)


class Document:
    """A parsed submission PDF."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no such PDF: {self.path}")
        self.doc = pymupdf.open(self.path)
        if self.doc.page_count == 0:
            raise ValueError(f"{self.path} contains no pages")
        self.metadata: dict[str, Any] = dict(self.doc.metadata or {})
        self.is_encrypted = bool(self.doc.is_encrypted)
        self.column_bands: list[tuple[float, float]] = []
        self.pages: list[PageInfo] = [self._read_page(i) for i in range(self.doc.page_count)]
        self._build_column_model()

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        try:
            self.doc.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def __enter__(self) -> Document:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def page_count(self) -> int:
        return len(self.pages)

    # -- parsing ----------------------------------------------------------
    def _read_page(self, index: int) -> PageInfo:
        page = self.doc[index]
        rect = page.rect
        info = PageInfo(
            number=index + 1,
            width=float(rect.width),
            height=float(rect.height),
            mediabox=tuple(float(v) for v in page.mediabox),  # type: ignore[arg-type]
            cropbox=tuple(float(v) for v in page.cropbox),  # type: ignore[arg-type]
        )

        trace = self._render_modes(page)
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") == 1:  # image block
                info.images.append(tuple(float(v) for v in block["bbox"]))  # type: ignore[arg-type]
                continue
            for line in block.get("lines", []):
                spans: list[Span] = []
                for sp in line.get("spans", []):
                    text = sp.get("text", "")
                    if not text:
                        continue
                    bbox = tuple(float(v) for v in sp["bbox"])
                    mode, opacity = trace.get(_trace_key(bbox), (0, 1.0))
                    spans.append(
                        Span(
                            text=text,
                            bbox=bbox,  # type: ignore[arg-type]
                            font=str(sp.get("font", "")),
                            size=float(sp.get("size", 0.0)),
                            color=int(sp.get("color", 0)),
                            flags=int(sp.get("flags", 0)),
                            page=info.number,
                            render_mode=mode,
                            opacity=opacity,
                        )
                    )
                if spans:
                    info.lines.append(Line(spans=spans, bbox=tuple(float(v) for v in line["bbox"]), page=info.number))  # type: ignore[arg-type]

        for d in page.get_drawings():
            info.drawings_bbox.append(tuple(float(v) for v in d["rect"]))  # type: ignore[arg-type]

        return info

    @staticmethod
    def _render_modes(page: pymupdf.Page) -> dict[tuple[int, int, int, int], tuple[int, float]]:
        """Map span bboxes to (render mode, opacity) via the low-level text trace.

        ``get_text("dict")`` does not expose the Tr operator or alpha, which are
        exactly what invisible-text tricks rely on.
        """
        out: dict[tuple[int, int, int, int], tuple[int, float]] = {}
        try:
            trace = page.get_texttrace()
        except Exception:  # pragma: no cover - older PyMuPDF
            return out
        for item in trace:
            mode = int(item.get("type", 0))
            opacity = float(item.get("opacity", 1.0))
            if mode == 0 and opacity >= 0.99:
                continue  # only record the interesting ones
            for ch in item.get("chars", []):
                bbox = ch[3] if len(ch) > 3 else None
                if not bbox:
                    continue
                out[_trace_key(bbox)] = (mode, opacity)
        return out

    def _build_column_model(self) -> None:
        """Infer the document's column bands from modal line left-edges.

        ACL papers have two column starts that recur on nearly every page. A
        histogram of left edges finds them far more reliably than looking for
        whitespace gutters, which tables and full-width figures fill in.
        """
        edges: dict[int, int] = {}
        for page in self.pages:
            for ln in page.lines:
                if not _is_body_line(ln, page):
                    continue
                key = int(round(ln.bbox[0] / 2.0))
                edges[key] = edges.get(key, 0) + 1

        peaks: list[tuple[float, int]] = []
        for key, count in sorted(edges.items(), key=lambda kv: -kv[1]):
            x = key * 2.0
            if any(abs(x - px) < 40.0 for px, _ in peaks):
                continue
            peaks.append((x, count))
            if len(peaks) >= 4:
                break

        page_width = self.pages[0].width or 595.0
        peaks = [(x, c) for x, c in peaks if c >= 3]
        if not peaks:
            starts = [0.0]
        else:
            primary, primary_count = peaks[0]          # peaks are count-ordered
            # A real second column carries about as many lines as the first. A
            # handful of lines sharing a left edge -- a stack of centred
            # equations, one table's column -- is not a column, and treating it
            # as one scrambles reading order on single-column papers.
            secondary = next(
                (x for x, c in peaks[1:]
                 if abs(x - primary) > 0.25 * page_width and c >= 0.15 * primary_count),
                None,
            )
            starts = sorted([primary, secondary]) if secondary is not None else [primary]

        # Column width: median extent of lines that begin at each start and do
        # not run into the next column.
        bands: list[tuple[float, float]] = []
        for i, s in enumerate(starts):
            limit = starts[i + 1] if i + 1 < len(starts) else page_width
            widths = [
                ln.bbox[2] - ln.bbox[0]
                for page in self.pages
                for ln in page.lines
                if _is_body_line(ln, page) and abs(ln.bbox[0] - s) < 4.0 and ln.bbox[2] <= limit - 1.0
            ]
            widths.sort()
            width = widths[int(len(widths) * 0.9)] if widths else (limit - s)
            bands.append((s, min(s + width, limit)))
        self.column_bands = bands

        for page in self.pages:
            page.column_bounds = list(bands)
            for ln in page.lines:
                page_centre = (ln.bbox[0] + ln.bbox[2]) / 2
                ln.column = min(
                    range(len(bands)),
                    key=lambda i: abs(page_centre - (bands[i][0] + bands[i][1]) / 2),
                )

    def column_fit(self, page: PageInfo) -> float:
        """Share of a page's body characters that sit inside a single column band.

        1.0 means cleanly columnised; a low value on an appendix page is what
        "the appendix is single-column" actually looks like in the geometry.
        """
        if len(self.column_bands) < 2:
            return 0.0
        inside = 0
        total = 0
        for ln in page.lines:
            if not _is_body_line(ln, page):
                continue
            n = len(ln.text.strip())
            if not n:
                continue
            total += n
            if any(b0 - 3.0 <= ln.bbox[0] and ln.bbox[2] <= b1 + 6.0 for b0, b1 in self.column_bands):
                inside += n
        return inside / total if total else 0.0

    def spans_both_columns(self, line: Line) -> bool:
        """True if a line crosses the gutter between the first two columns."""
        if len(self.column_bands) < 2:
            return False
        gutter_start = self.column_bands[0][1]
        gutter_end = self.column_bands[1][0]
        return line.bbox[0] < gutter_start - 3.0 and line.bbox[2] > gutter_end + 3.0

    # -- derived views ----------------------------------------------------
    @cached_property
    def spans(self) -> list[Span]:
        return [s for p in self.pages for s in p.spans]

    @cached_property
    def visible_spans(self) -> list[Span]:
        return [s for s in self.spans if s.text.strip() and not s.invisible]

    @cached_property
    def reading_order(self) -> list[Line]:
        """Lines in column-major reading order, which is what ACL papers use."""
        out: list[Line] = []
        for page in self.pages:
            out.extend(sorted(page.lines, key=lambda ln: (ln.column, round(ln.bbox[1], 1), ln.bbox[0])))
        return out

    @cached_property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.reading_order)

    @cached_property
    def body_font_size(self) -> float:
        """Modal font size over visible text, rounded to 0.5pt."""
        tally: dict[float, float] = {}
        for s in self.visible_spans:
            if len(s.text.strip()) < 2:
                continue
            key = round(s.size * 2) / 2
            tally[key] = tally.get(key, 0.0) + len(s.text.strip())
        if not tally:
            return 0.0
        return max(tally.items(), key=lambda kv: kv[1])[0]

    @cached_property
    def dominant_font(self) -> tuple[str, float]:
        """(font name, share of characters) for the most-used font."""
        tally: dict[str, int] = {}
        for s in self.visible_spans:
            n = len(s.text.strip())
            if n:
                tally[s.font] = tally.get(s.font, 0) + n
        if not tally:
            return ("", 0.0)
        total = sum(tally.values())
        name, count = max(tally.items(), key=lambda kv: kv[1])
        return (name, count / total)

    # -- headings ---------------------------------------------------------
    @cached_property
    def headings(self) -> list[Heading]:
        body = self.body_font_size or 11.0
        starts = [b[0] for b in self.column_bands] or [0.0]
        out: list[Heading] = []
        order = 0
        for line in self.reading_order:
            text = " ".join(line.text.split())
            if not _looks_like_heading(text):
                continue
            size = line.heading_size
            # Small caps are a heading style in their own right: the ICLR
            # template sets subsection headings as 10pt small caps, no larger
            # than the body and not bold.
            if not (line.bold or size >= body + 0.4 or line.small_caps):
                continue
            # Headings begin at a column edge; table cells and inline bold runs
            # start anywhere, and captions announce themselves.
            spans_gutter = self.spans_both_columns(line)
            # A numbered heading's label is indented by the width of its number,
            # which the extractor often emits as a separate run. When that run
            # is found at the column edge it anchors the label wherever a long
            # number ("B.2.1") pushed it.
            numbering_run = self._numbering_before(line, starts)
            at_column = numbering_run is not None or any(
                -6.0 <= line.bbox[0] - s <= 34.0 for s in starts
            )
            # "Abstract" and the title block are centred rather than set at a
            # column edge, so anchoring only to columns misses them.
            if not (spans_gutter or at_column or self._centred_over_block(line)):
                continue
            m = re.match(r"^((?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)*))[.\s]+(.+)$", text)
            numbering, label = (m.group(1), m.group(2)) if m else (None, text)
            if numbering is None:
                numbering = numbering_run
            out.append(
                Heading(
                    text=label.strip(),
                    page=line.page,
                    bbox=line.bbox,
                    size=size,
                    order=order,
                    numbering=numbering,
                )
            )
            order += 1
        return out

    def _centred_over_block(self, line: Line) -> bool:
        """True if a line is centred over the text block that follows it.

        ACL centres "Abstract" over the abstract block and the title over the
        page, neither of which starts at a column edge.
        """
        page = self.pages[line.page - 1]
        following = [
            other for other in page.lines
            # Directly beneath, and horizontally overlapping: without the overlap
            # test a neighbouring column's lines widen the block and the centre
            # lands in the gutter.
            if 0.0 <= other.bbox[1] - line.bbox[3] < 40.0
            and other.bbox[0] < line.bbox[2] and other.bbox[2] > line.bbox[0]
            and other.text.strip() and not other.text.strip().isdigit()
        ]
        following.sort(key=lambda o: o.bbox[1])
        block = following[:4]
        if len(block) < 2:
            # Nothing below it: fall back to the page, which catches the title.
            return abs((line.bbox[0] + line.bbox[2]) / 2 - page.width / 2) < 30.0
        left = min(o.bbox[0] for o in block)
        right = max(o.bbox[2] for o in block)
        if right - left < 40.0:
            return False
        centre = (line.bbox[0] + line.bbox[2]) / 2
        return abs(centre - (left + right) / 2) < 12.0

    def _numbering_before(self, line: Line, starts: list[float]) -> str | None:
        """Recover a section number emitted as its own run left of the label."""
        for other in self.pages[line.page - 1].lines:
            if other is line or abs(other.bbox[1] - line.bbox[1]) > 1.5:
                continue
            if other.bbox[2] > line.bbox[0] + 1.0:
                continue
            token = other.text.strip()
            if re.fullmatch(r"(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)\.?", token) and any(
                -6.0 <= other.bbox[0] - s <= 6.0 for s in starts
            ):
                return token.rstrip(".")
        return None

    def find_heading(self, aliases: list[str]) -> Heading | None:
        """First heading whose label matches one of ``aliases`` (case/punct-insensitive)."""
        wanted = {re.sub(r"[^a-z ]", "", a.lower()).strip() for a in aliases}
        for h in self.headings:
            if h.normalized in wanted:
                return h
        return None

    # -- links and identifiers -------------------------------------------
    @cached_property
    def hyperlinks(self) -> list[tuple[int, str]]:
        """(page, uri) for every link annotation."""
        out: list[tuple[int, str]] = []
        for i in range(self.doc.page_count):
            try:
                for link in self.doc[i].get_links():
                    uri = link.get("uri")
                    if uri:
                        out.append((i + 1, str(uri)))
            except Exception:  # pragma: no cover
                continue
        return out

    @cached_property
    def textual_urls(self) -> list[tuple[int, str]]:
        out: list[tuple[int, str]] = []
        for page in self.pages:
            for match in _URL_RE.finditer(page.text.replace("\n", " ")):
                out.append((page.number, match.group(0).rstrip(".,;)")))
        return out

    @cached_property
    def emails(self) -> list[tuple[int, str]]:
        return [
            (p.number, m.group(0))
            for p in self.pages
            for m in _EMAIL_RE.finditer(p.text.replace("\n", " "))
        ]

    @cached_property
    def orcids(self) -> list[tuple[int, str]]:
        return [
            (p.number, m.group(0))
            for p in self.pages
            for m in _ORCID_RE.finditer(p.text.replace("\n", " "))
        ]

    @cached_property
    def annotations(self) -> list[tuple[int, str, str]]:
        """(page, annot type, content) for comment-style annotations."""
        out: list[tuple[int, str, str]] = []
        for i in range(self.doc.page_count):
            try:
                annot = self.doc[i].first_annot
            except Exception:  # pragma: no cover
                continue
            while annot is not None:
                content = (annot.info.get("content") or "").strip()
                title = (annot.info.get("title") or "").strip()
                if content or title:
                    out.append((i + 1, annot.type[1], f"{title}: {content}".strip(": ")))
                annot = annot.next
        return out

    @cached_property
    def embedded_files(self) -> list[str]:
        try:
            return [self.doc.embfile_info(i)["filename"] for i in range(self.doc.embfile_count())]
        except Exception:  # pragma: no cover
            return []

    def page_text(self, number: int) -> str:
        return self.pages[number - 1].text if 1 <= number <= self.page_count else ""

    def text_between(self, start: Heading, end: Heading | None) -> str:
        """Text in reading order from one heading up to (not including) the next."""
        lines = self.reading_order
        try:
            start_idx = next(i for i, ln in enumerate(lines) if _same_line(ln, start))
        except StopIteration:  # pragma: no cover
            return ""
        end_idx = len(lines)
        if end is not None:
            for i in range(start_idx + 1, len(lines)):
                if _same_line(lines[i], end):
                    end_idx = i
                    break
        return "\n".join(ln.text for ln in lines[start_idx + 1 : end_idx])


def covers(box: BBox, inner: BBox, slack: float = 2.0) -> bool:
    """True if ``box`` encloses ``inner``, with a little slack for rounding."""
    return (
        box[0] <= inner[0] + slack
        and box[1] <= inner[1] + slack
        and box[2] >= inner[2] - slack
        and box[3] >= inner[3] - slack
    )


def _same_line(line: Line, heading: Heading) -> bool:
    return line.page == heading.page and abs(line.bbox[1] - heading.bbox[1]) < 0.6


def _trace_key(bbox: Any) -> tuple[int, int, int, int]:
    return (round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3]))


_CAPTION_RE = re.compile(r"(?i)^(figure|fig\.|table|algorithm|listing|panel|equation|appendix table)\b\s*[a-z]?\.?\d")


def _is_body_line(line: Line, page: PageInfo) -> bool:
    """Exclude margin line numbers, invisible text, and non-alphabetic runs."""
    text = line.text.strip()
    if not text or text.isdigit():
        return False
    if not any(c.isalpha() for c in text):
        return False
    if all(s.invisible for s in line.spans):
        return False
    # ACL's anonymous template prints line numbers outside the text block.
    if line.bbox[2] < page.width * 0.08 or line.bbox[0] > page.width * 0.92:
        return False
    return True


def _looks_like_heading(text: str) -> bool:
    if not text or len(text) > 90:
        return False
    if _CAPTION_RE.match(text.strip()):
        return False
    stripped = text.strip()
    if len(stripped) < 3:
        return False
    if stripped.endswith((".", ",", ";", ":")) and not re.match(r"^\d+(\.\d+)*\.?$", stripped):
        return False
    # Must start with a number/letter label or a capital letter.
    if not re.match(r"^(?:\d+(?:\.\d+)*|[A-Z](?:\.\d+)*)?[.\s]*[A-Z]", stripped):
        return False
    words = stripped.split()
    return 1 <= len(words) <= 14
