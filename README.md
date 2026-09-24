# preflight

**Identify issues that may result in desk rejection before the submission is reviewed by a program chair.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Checks: 117](https://img.shields.io/badge/checks-117-blue)](#checks)
[![Venues: 10](https://img.shields.io/badge/venues-10-blueviolet)](#conferences)
[![Tests: 366](https://img.shields.io/badge/tests-366%20passing-brightgreen)](#development)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-d7ff64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-de5fe9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

`preflight` analyses a submission PDF against a conference's requirements: page limits, required
sections, margins, fonts, anonymity, appendix placement, concealed text directed at automated
reviewers, reference validity, and the venue's responsible-research checklist. Each finding reports
the specific location and measurement that produced it — `page 5, x=34.0pt, expected >= 69.0pt`.

It includes profiles for ACL Rolling Review, NeurIPS, ICLR, ICML, CVPR, AAAI, AISTATS, ICRA, ICASSP
and BayLearn. Conference profiles are YAML files; additional venues can be configured without
modifying the source code.

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ ARR pre-flight (long): 0 errors, 3 warnings, 21 passed                       │
╰────────────────────────── /Users/you/papers/submission.pdf ──────────────────╯

Errors — likely desk rejection

✗ ERROR Content page limit  [page_limit]
      Main content appears to run through page 9; long papers allow 8. Content runs up
      to the 'Limitations' heading on page 10. Exceeding the page limit is explicitly a
      desk-rejection condition.
      · page 10 — first unlimited section: 'Limitations' — measured 9.0, expected <= 8
      · 23 pages total in the PDF (14 of them unlimited material)
      confidence: high — derived from the position of the first unlimited section, not raw page count
      fix: Move material into the appendix (which is unlimited and sits after the references),
           or cut content. Do not shrink fonts or margins to fit.
```

---

## Classification of findings

The tool maintains a clear distinction among four categories:

| Category | Basis | Reported as |
|---|---|---|
| **Deterministic** | Measurements of the PDF's geometry, fonts and text boxes: page size, margins, page limits, section order, column layout; and each reference compared against its record in a key source. | `ERROR` — a documented desk-rejection condition, or a citation that no key source confirms or whose record contradicts it, accompanied by the supporting measurement. |
| **Heuristic** | Pattern matching and model judgement: anonymity leaks, prompt injection, checklist coverage, undersized table text. | `WARNING` — an institution named in related work may be legitimate, so adjudication requires a human reader. |
| **Advisory** | Reviewer perspectives: triage, experimental design, statistics, figures, prose, ethics. | `WARNING` — assessments of quality and presentation rather than venue requirements. |
| **Out of scope** | Originality, dual submission, citation coverage, OpenReview metadata, reviewer registration. | `N/A` — enumerated explicitly on every run. |

A successful result indicates that the PDF-checkable rules pass. It should not be interpreted as
confirmation that the submission is fully compliant, and the report states this.

---

## Installation

```bash
git clone https://github.com/DavidMChan/preflight.git && cd preflight
uv sync
```

Python 3.12 is required (the project pins `>=3.12,<3.13`). Credentials are supplied through a
`.env` file:

```bash
cp .env.example .env
# OPENAI_API_KEY=sk-...
# PREFLIGHT_LLM_MODEL=gpt-5.6-luna
```

The API key is optional. In its absence the model-backed checks report themselves as skipped, and
the deterministic report is unaffected.

---

## Usage

```bash
uv run preflight paper.pdf                          # select a conference interactively
uv run preflight paper.pdf -c arr -t short          # ARR short-paper limits
uv run preflight paper.pdf -c neurips               # a different venue
uv run preflight paper.pdf -c icra                  # ICRA complete-paper checks
uv run preflight paper.pdf -c icassp                # ICASSP 4 + 1 pages, author list required
uv run preflight paper.pdf -c iclr -t camera_ready  # ICLR, 10-page camera-ready limit
uv run preflight paper.pdf --offline                # deterministic checks only, no network
uv run preflight tui paper.pdf                      # interactive interface
```

A PDF path supplied without a subcommand is equivalent to `preflight check <path>`.

Model-backed checks and reference verification are enabled by default. They can be disabled
individually with `--no-llm`, `--no-refcheck` and `--no-scores`, or collectively with `--offline`.

Results are cached to avoid repeated processing costs:

```bash
uv run preflight show                           # re-read the last run in full
uv run preflight show min_font_size             # a single finding with all its evidence
uv run preflight show --list                    # enumerate cached runs
```

```bash
uv run preflight conferences                    # list bundled profiles
uv run preflight checks -c arr                  # list the checks a profile runs
uv run preflight paper.pdf --json report.json --markdown report.md
uv run preflight *.pdf --quiet --strict         # batch processing
```

Additional options: `-m/--model` selects the model, `--concurrency` limits simultaneous model calls
(default 8), `--verbose` reports all evidence and CFP references, `--quiet` restricts output to
errors and warnings, and `--strict` treats warnings as a non-zero exit condition.

Exit codes: `0` clean · `1` errors · `2` warnings under `--strict`, or an invalid file or profile.

### Terminal interface

`preflight tui paper.pdf` presents findings in the left pane and full evidence in the right pane.
`r` re-runs · `l` LLM · `s` scores · `v` verbose · `n` next PDF · `e` export Markdown · `q` quit.

---

## Conferences

Ten bundled profiles, each selected with `-c`:

| Key | Venue | Tracks (content page limit) | Format |
|---|---|---|---|
| `arr` | ACL Rolling Review | `long` 8p · `short` 4p · `demo` 6p | A4, two-column, 11pt; Limitations mandatory; ARR checklist |
| `neurips` | NeurIPS (main track) | `main` 9p | US Letter, single-column, 10pt; NeurIPS checklist mandatory |
| `iclr` | ICLR (main conference) | `main` 9p · `camera_ready` 10p | US Letter, single-column, 10pt; AI use statement mandatory |
| `icml` | ICML | `main` 8p · `position` 8p · `camera_ready` 9p | US Letter, two-column, 10pt; Impact Statement mandatory |
| `cvpr` | CVPR (IEEE/CVF) | `main` 8p · `rebuttal` 1p | US Letter, two-column, 10pt; no appendix in the submission PDF |
| `aaai` | AAAI (main technical track) | `main` 7p, and six further tracks | US Letter, two-column, 10pt; reproducibility checklist |
| `aistats` | AISTATS | `main` 8p · `camera_ready` 9p | US Letter, two-column, 10pt |
| `icra` | ICRA 2027 (IEEE RAS) | `main` 8p complete paper | US Letter, two-column, 10pt; references included; double-anonymous |
| `icassp` | ICASSP 2026 (IEEE SPS) | `main` 4p, plus a 5th page of end matter only | US Letter or A4, two-column, 10pt or 9pt; not blind; ethics and conflict-of-interest statements mandatory |
| `baylearn` | BayLearn Symposium (abstracts) | `abstract` 2p | NeurIPS format, no checklist |

Each profile encodes the most recent author kit published at the time it was written, named in a
note at the top of the file. `cvpr`, `aistats` and `icml` carry 2026 numbers and `aaai` carries
AAAI-27 numbers, because no later kit exists yet. Where a value could not be confirmed against an
official source it is documented in the profile and its check is held at warning, so a superseded
number cannot report a desk rejection.

---

## Checks

117 checks across 27 modules. Each profile enables a subset; ARR enables 116.

### Deterministic — `core.geometry`, `core.fonts`, `core.structure`

- **Page dimensions** compared against the style specification (A4 for ACL, US Letter for NeurIPS
  and ICLR). A venue that accepts more than one size (ICASSP: US Letter or A4) gives each its own
  text block.
- **Margins**, measured from text and image bounding boxes using the style's tolerances. Line
  numbers in anonymous templates are excluded.
- **Column layout** — gutter width for two-column venues, and for single-column venues (NeurIPS,
  ICLR) confirmation that the body was not set in columns.
- **Body font size**, taken as the modal size across the document, against every size the style
  offers (ICASSP's spconf sets 10pt, or 9pt with `\ninept`), and **minimum font size**, reported
  separately for figure labels, table cells and running prose.
- **Content page limit.** The limit applies to content preceding the first venue-approved unlimited
  section. For venues such as ICRA that count references and appendices, every PDF page counts.
  ICASSP allows only named end matter past the limit, so there every other section, figure and table
  counts wherever it sits, and a track's `total_page_limit` caps the whole PDF.
- **Limitations section** — presence, title, position before the references, and absence of new
  methods or results.
- **Appendix** — venue-specific placement and column layout.

### Venue statements and template — `core.statements`, `core.template`

A profile's `statements:` map declares the sections a venue requires or recommends beside the
paper — for ICLR, the AI use, ethics and reproducibility statements. Each is located as a heading or
a bold run-in head, and reported as missing (an error when required, a warning when recommended),
still holding the template's placeholder text (treated as missing when required), placed after the
references, longer than the venue's cap, or silent on a topic it must name (`must_mention`: for
ICASSP, conflicts of interest, even when there are none). Statements a venue excludes from the page limit are
listed in `structure.unlimited_after`. With `--llm`, a model also confirms that the statement
settles every item the venue lists; for ICLR, the twelve tasks with required AI disclosure. A
venue can say when its items apply (`must_address_when`): ICASSP's ethics items bind only work
with human or animal subjects, sensitive personal data or a sensitive application, so the model
first reads the paper to decide, and a paper outside that condition passes.

`core.template` reads the style file's running head, which records the template year and whether
the camera-ready switch is on: an earlier year's style files, or author names enabled on a
submission, are errors. For venues that paginate their own proceedings it also reports page numbers
on the submission, and it checks that the title is set in capitals where the style file does that.

### Anonymity — `core.anonymity`

Title-block scanning for names, affiliations, emails and ORCIDs; URL classification into forbidden
(tracking services and link shorteners), identity-bearing, and anonymised hosts; self-reference
phrasing; and PDF metadata, annotations and embedded filenames. URLs within the bibliography are
excluded.

Findings in this module are warnings with a stated confidence, with the exception of an email
address in the title block. Venues that do not review blind (ICASSP) switch the module off and
instead require an author block: a blank one, or the template's placeholder names, is an error.

### Concealed content — `core.hidden`

Invisible text (render mode 3/7, zero opacity), microscopic text, text set in the page colour, text
outside the CropBox, and prompt-injection patterns.

Injection patterns match text that addresses an automated reviewer rather than text that discusses
one. Visible matches are reported as warnings; concealed matches are reported as errors.

### Responsible NLP checklist — `arr.responsible_nlp`

All eighteen ARR checklist items (A1–A2, B1–B6, C1–C4, D1–D5, E1), each evaluating whether the
paper contains the information the item requires.

The checklist is completed in OpenReview rather than attached to the PDF, so checklist *answers*
are not inspected. No individual item produces an error; the CFP permits a negative answer
accompanied by a justification. The summary check reports systematic omission, which is the
desk-rejectable condition.

Items are defined in [`src/preflight/conferences/rnlp/`](src/preflight/conferences/rnlp/), one YAML
file each.

### NeurIPS checklist — `neurips.checklist`

Presence, position (after the references and appendices), completeness of the sixteen answers,
residual template placeholders, missing justifications, and consistency against the paper body.
Mandatory for NeurIPS. ARR includes the module in optional mode: a missing checklist is not
reported, and a checklist that is present is validated.

### Reference verification — `integrations.refcheck`

A reference passes only when a key source confirms it: a record whose title and authors match the
citation. The record is then compared field by field against the citation. Four findings report the
result:

| Check | Severity | Reports |
|---|---|---|
| `reference_verification` | `ERROR` | References that no key source confirms and no web search traced to a page. |
| `reference_verification.web_only` | `WARNING` | References no key source has, but that a web search located at a page that answers; their details are unchecked. |
| `reference_details` | `ERROR` | Confirmed references whose record contradicts the citation: a printed DOI or arXiv identifier that is unregistered or belongs to a different work; page ranges; years; venues, including Findings cited as the main conference and a cited venue that no record confirms; authors absent from the record, altered given names, reordered authors, and author lists shortened without "et al.". |
| `reference_versions` | `WARNING` | A preprint, or a work cited without a venue, whose published version exists. |

The key sources, and the terms under which each is queried:

| Source | Route | Etiquette |
|---|---|---|
| DOI registry | CrossRef, then doi.org content negotiation (DataCite and other registrars), then the handle API for existence | CrossRef's published limits: one request per second and one at a time, or three with `mailto` |
| ACL Anthology | The complete bibliography (`anthology.bib.gz`, 13 MB), indexed locally | Downloaded at most weekly |
| arXiv | `export.arxiv.org`, titles and identifiers batched into single queries | One request every three seconds over one connection |
| DBLP | `sparql.dblp.org`, all titles in one query, and first author with year for retitled works | The ten-second crawl delay in its robots.txt; `dblp.org` itself disallows automated access and is not queried |
| OpenReview | The search API | Its published policy of twenty requests per minute |
| OpenAlex | The works API | Requires `OPENALEX_API_KEY` beyond a small budget shared per network |
| Open Library | The search API, for books | One request every two seconds |
| Publisher pages | `citation_*` metadata on the landing page | Only where the site's robots.txt permits, at its crawl delay |

Every request identifies the tool in its User-Agent, with a contact address when `refcheck.mailto`
or `PREFLIGHT_MAILTO` is set. Rate and concurrency limits that a host publishes in its response
headers are adopted for the rest of the run. A throttled request (429 or 503) is retried up to four
times, waiting as long as the host's `Retry-After` asks or else backing off exponentially with jitter;
the host's rate is halved on each refusal and recovers as answers come back. A host is dropped only
when it asks for a wait of more than a minute, or refuses 25 times in a row. A repository, model card, package or blog post is
verified against the thing it is: the GitHub API, the Hugging Face Hub, PyPI, or its own URL.

The model's web search is used only to locate a record. The DOI, arXiv identifier or URL it returns
is fetched from the key source and matched like any other record. A work that no key source confirms
is found only by web search (a warning) when the search points to a page that answers and no record
there contradicts the citation. It stays unconfirmed (an error) when the search gives no page or
identifier, the page refuses or is gone, or the record it points to is a different work.

Measured on a 91-entry bibliography from a submission:

| Measurement | Result |
|---|---|
| Wall clock, cold | 80s, plus approximately 25s to parse the bibliography |
| Wall clock, warm | under 1s, all 91 from the cache |
| Confirmation | 90 by a key source · 1 resolved to a live source · 1 web search |
| Details | 24 references contradicted by their records, among them two DOIs belonging to other papers, five incorrect page ranges, invented and altered author names, and fifteen preprints or venue-less citations with published versions |

The cache is keyed on each entry's text, so an edited entry is checked again. An unconfirmed
reference is reported as unconfirmed rather than fabricated, since indexes are incomplete; it is
still an error, because a reader must be able to trace every citation. The severities are set by
`refcheck.not_found_is_error`, `refcheck.details_mismatch_is_error` and
`refcheck.published_version_is_error`.

### File and manuscript integrity — `core.markup`, `core.pdf`, `core.citations`, `core.statistics`, `core.abbreviations`, `core.crossrefs`, `core.headings`, `core.figure_quality`

The system performs an additional 23 deterministic checks, requiring approximately one second in
total and no model-based processing.

| Module | Coverage |
|---|---|
| `core.markup` | PDF comment annotations, which also disclose the commenter's name; highlight annotations and `\colorbox`-style highlights; and residual revision markup. |
| `core.pdf` | Blank pages, non-embedded or unsubset fonts, Type 3 bitmap fonts where a venue discourages them, encryption, file size, and general file health. |
| `core.citations` | Resolution of every in-text citation to a bibliography entry, and of every entry to a citation. |
| `core.statistics` | p-values reported as zero (`p = .000`), mixed p-value conventions, and recomputation of "N of M (X%)" statements. |
| `core.abbreviations` | Expansion at first use, and a single expansion per abbreviation. |
| `core.crossrefs` | Reference to every float from the text and resolution of every callout; agreement between figure panels and captions; resolution and uniqueness of appendix and supplement identifiers. |
| `core.headings` | Contiguous and correctly nested section numbering; absence of empty sections. |
| `core.figure_quality` | Effective DPI of raster figures, and retention of figure colour distinctions under greyscale printing and a deuteranopia transform. |

All are reported as warnings, as none corresponds to a documented desk-rejection condition.

Certain requirements cannot be evaluated from a PDF and are reported as such rather than
approximated: tracked changes, structured variable declarations, and the existence of a
supplementary *file* require the submission package.

### Reviewer perspectives — `llm.review`, `llm.prose`, `llm.audit`

Thirteen audits, each applying a reviewer's perspective to the full paper.

| Audit | Perspective |
|---|---|
| `busy_reviewer` | Simulates rapid initial review by an area chair, focusing on the abstract, figures and conclusion. Returns Accept / Borderline / Reject and the least costly change that would alter it. |
| `surface_triage` | Title, abstract and contribution claims — the material most likely to receive attention during an initial review. |
| `ai_prose_signals` | Prose exhibiting generic or machine-generated characteristics, assessed from measured statistics rather than impression. |
| `typos_grammar` | Spelling and grammar errors, systematically filtered for PDF-extraction artifacts. |
| `experiment_claims` | Whether the experiments can answer the claims: baselines, ablations, external validity. |
| `experiment_statistics` | Seeds, variance, error bars, paired tests, multiple comparisons. |
| `experiment_leakage` | Splits, contamination, benchmark leakage, tuning on the test set. |
| `claims_vs_numbers` | Agreement between the abstract's figures and the tables, and between the text and the numbers. |
| `figure_table` | Whether each caption is self-contained: metric, data, units, direction of improvement. |
| `formula_audit` | Dimensions, indices, bounds, sign conventions, undefined symbols. |
| `complexity_claims` | Asymptotic claims: the governing variable and the applicable regime. |
| `security_ethics` | Data rights, personally identifying information, dual use, and claim terms such as "secure" and "anonymous". |
| `systems_performance` | Latency, throughput, speedup and cost claims, where present. |

Each audit is a YAML file in
[`src/preflight/conferences/audits/`](src/preflight/conferences/audits/).

### Prose diagnostics — `core.prose`, `core.figures`, `core.surface`

Counted rather than judged, with no model involved: a sentence-length histogram against a 46-word
threshold, connective frequencies, runs of three or more sentences with identical openings, bare
demonstratives, hollow concluding sentences, caption completeness and numbering, whether each
substantive title term appears in the body, and, where the template limits it, the number of
paragraphs in the abstract.

### Reviewer-style scores — `llm.semantic`

A rehearsal review of the full paper: soundness, excitement, clarity and reproducibility scored
1–5, together with strengths, weaknesses and desk-rejection risks.

Each dimension is scored against written anchors describing the criteria corresponding to a score
of 2 compared with those corresponding to a score of 4, and the prompt states that 3 is not a
neutral default. Each justification must identify the section,
table or value supporting the score and, below 5, the change that would raise it by one point. The
`overall` value is a predicted outcome rather than an average.

The result is labelled an estimate and never blocks a run.

---

## Conference configuration

A conference profile is a composition of check modules together with the settings those modules
read. Profiles are YAML with single inheritance.

```
base.yaml          venue-agnostic defaults, every module
└── acl.yaml       A4, two-column, 11pt Times, ACL tolerances
    └── arr.yaml   ARR page limits, mandatory Limitations, ARR checklist
neurips.yaml       US Letter, single-column, 10pt, NeurIPS checklist mandatory
└── baylearn.yaml  BayLearn abstracts: NeurIPS format, 2 pages, no checklist
iclr.yaml          US Letter, single-column, 10pt, 9 pages (10 for camera ready)
icra.yaml          US Letter, two-column, 10pt, 8 complete pages (references included)
icassp.yaml        US Letter or A4, two-column, 10pt, 4 pages + 1 of end matter, not blind
```

```yaml
conference:
  key: myvenue
  name: My Venue
  extends: acl              # inherit ACL geometry and typography

tracks_replace: true        # <key>_replace discards the inherited value instead of merging
tracks:
  long: { content_page_limit: 12 }

modules:                    # appended to the parent's set
  - neurips.checklist

disabled_checks:
  - neurips_checklist_present

severity:                   # adjust a shared check without forking it
  margins: warning
  limitations_present: error
```

```bash
uv run preflight paper.pdf -c ./myvenue.yaml
```

Three mechanisms make modules reusable across venues:

- **`modules`** determines which check modules run, composed along the inheritance chain.
- **`severity`** remaps a check's failure severity per venue. `PASS` and `SKIPPED` are never
  rewritten, as an override expresses the significance of a failure rather than its occurrence.
- **`disabled_checks`** removes an individual check while retaining the remainder of its module.

Numeric tolerances are defined in the profile rather than in Python.

---

## Adding a check

```python
from ..context import CheckContext
from ..models import Evidence, Finding
from ..registry import register

MODULE = "core.geometry"

@register("paper_size", "Paper size", module=MODULE, category="format", order=10)
def check_paper_size(ctx: CheckContext) -> Finding:
    """Compare every page's MediaBox against the style's required dimensions."""
    want = float(ctx.conf("geometry.page_width_pt", 595.0))
    ...
    return ctx.error("paper_size", "Paper size", "...", evidence=[
        Evidence(page=3, detail="page is 612.0 x 792.0 pt", expected="595 x 842 pt (A4)"),
    ])
```

Conventions observed by the existing checks:

- Read every threshold from the profile through `ctx.conf("dotted.path", default)`. Venue-specific
  values are not hard-coded.
- Construct findings with `ctx.ok`, `ctx.warn`, `ctx.error` or `ctx.skip`.
- Attach `Evidence` containing the measurement.
- Reserve `ERROR` for conditions demonstrable from the PDF. Where the discriminator is unreliable,
  report a warning.
- Declare `requires=("enable_llm",)` for any check requiring a model, and define it as `async def`.
- A failing check is caught and reported as `SKIPPED`; it does not cause the overall process to
  fail.

---

## Development

```bash
uv run pytest                                  # 366 tests, no network required
uv run ruff check src tests
uv run python scripts/try_question.py A1 paper.pdf    # a single checklist item
```

Tests construct their own synthetic PDFs and depend on no external files.

---

## Limitations and Out-of-Scope Assessments

The following are not evaluated: originality and prior publication, concurrent submission, adequacy
of citation coverage, correctness of the author list, resubmission metadata, the truth of checklist
answers, the accuracy of AI-use disclosure, and co-author reviewer registrations.

---

## License

MIT. See [LICENSE](LICENSE).
