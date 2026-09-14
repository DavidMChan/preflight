# preflight

**Identify issues that may result in desk rejection before the submission is reviewed by a program chair.**

[![Python 3.12](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Checks: 104](https://img.shields.io/badge/checks-104-blue)](#checks)
[![Venues: 9](https://img.shields.io/badge/venues-9-blueviolet)](#conferences)
[![Tests: 229](https://img.shields.io/badge/tests-229%20passing-brightgreen)](#development)
[![Lint: ruff](https://img.shields.io/badge/lint-ruff-d7ff64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-de5fe9?logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

`preflight` analyses a submission PDF against a conference's requirements: page limits, required
sections, margins, fonts, anonymity, appendix placement, concealed text directed at automated
reviewers, reference validity, and the venue's responsible-research checklist. Each finding reports
the specific location and measurement that produced it — `page 5, x=34.0pt, expected >= 69.0pt`.

It includes profiles for ACL Rolling Review, NeurIPS, ICLR, ICML, CVPR, AAAI, AISTATS, ICRA and
BayLearn. Conference profiles are YAML files; additional venues can be configured without
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
| **Deterministic** | Measurements of the PDF's geometry, fonts and text boxes: page size, margins, page limits, section order, column layout. | `ERROR` — a documented desk-rejection condition, accompanied by the supporting measurement. |
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

Nine bundled profiles, each selected with `-c`:

| Key | Venue | Tracks (content page limit) | Format |
|---|---|---|---|
| `arr` | ACL Rolling Review | `long` 8p · `short` 4p · `demo` 6p | A4, two-column, 11pt; Limitations mandatory; ARR checklist |
| `neurips` | NeurIPS (main track) | `main` 9p | US Letter, single-column, 10pt; NeurIPS checklist mandatory |
| `iclr` | ICLR (main conference) | `main` 9p · `camera_ready` 10p | US Letter, single-column, 10pt |
| `icml` | ICML | `main` 8p · `position` 8p · `camera_ready` 9p | US Letter, two-column, 10pt; Impact Statement mandatory |
| `cvpr` | CVPR (IEEE/CVF) | `main` 8p · `rebuttal` 1p | US Letter, two-column, 10pt; no appendix in the submission PDF |
| `aaai` | AAAI (main technical track) | `main` 7p, and six further tracks | US Letter, two-column, 10pt; reproducibility checklist |
| `aistats` | AISTATS | `main` 8p · `camera_ready` 9p | US Letter, two-column, 10pt |
| `icra` | ICRA 2027 (IEEE RAS) | `main` 8p complete paper | US Letter, two-column, 10pt; references included; double-anonymous |
| `baylearn` | BayLearn Symposium (abstracts) | `abstract` 2p | NeurIPS format, no checklist |

Each profile encodes the most recent author kit published at the time it was written, named in a
note at the top of the file. `cvpr`, `aistats` and `icml` carry 2026 numbers and `aaai` carries
AAAI-27 numbers, because no later kit exists yet. Where a value could not be confirmed against an
official source it is documented in the profile and its check is held at warning, so a superseded
number cannot report a desk rejection.

---

## Checks

104 checks across 25 modules. Each profile enables a subset; ARR enables 103.

### Deterministic — `core.geometry`, `core.fonts`, `core.structure`

- **Page dimensions** compared against the style specification (A4 for ACL, US Letter for NeurIPS
  and ICLR).
- **Margins**, measured from text and image bounding boxes using the style's tolerances. Line
  numbers in anonymous templates are excluded.
- **Column layout** — gutter width for two-column venues, and for single-column venues (NeurIPS,
  ICLR) confirmation that the body was not set in columns.
- **Body font size**, taken as the modal size across the document, and **minimum font size**,
  reported separately for figure labels, table cells and running prose.
- **Content page limit.** The limit applies to content preceding the first venue-approved unlimited
  section. For venues such as ICRA that count references and appendices, every PDF page counts.
- **Limitations section** — presence, title, position before the references, and absence of new
  methods or results.
- **Appendix** — venue-specific placement and column layout.

### Anonymity — `core.anonymity`

Title-block scanning for names, affiliations, emails and ORCIDs; URL classification into forbidden
(tracking services and link shorteners), identity-bearing, and anonymised hosts; self-reference
phrasing; and PDF metadata, annotations and embedded filenames. URLs within the bibliography are
excluded.

Findings in this module are warnings with a stated confidence, with the exception of an email
address in the title block.

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

Every bibliography entry is verified through four tiers, and a reference exits at the first tier
that resolves it:

1. **Cache.** References recur across drafts.
2. **Bulk databases, concurrently.** CrossRef, OpenAlex and DOI resolution are issued for all
   references simultaneously under per-host rate limits.
3. **Routing by cited artifact type.** A repository is verified against the GitHub API, a model
   card against the Hugging Face Hub, a package against PyPI, and a blog post against its own URL.
4. **Live web search.** Remaining references are submitted to the model's `web_search` tool under
   scholarly domain filters and return a citation URL.

Measured on a 62-entry bibliography from a submission:

| Measurement | Result |
|---|---|
| Wall clock, cold | 62 references in 39s, plus approximately 16s to parse the bibliography |
| Wall clock, warm | 1.7s, identical verdicts |
| Resolution | 47 verified in a database · 12 resolved to a live source · 3 author-order discrepancies |
| False negatives | none |

`refcheck.mailto` should be set to a contact address.

An unresolved reference is reported as unverified rather than fabricated, since bibliographic
indexes are incomplete. It is a warning by default and never an error.

### File and manuscript integrity — `core.markup`, `core.pdf`, `core.citations`, `core.statistics`, `core.abbreviations`, `core.crossrefs`, `core.headings`, `core.figure_quality`

The system performs an additional 22 deterministic checks, requiring approximately one second in
total and no model-based processing.

| Module | Coverage |
|---|---|
| `core.markup` | PDF comment annotations, which also disclose the commenter's name; highlight annotations and `\colorbox`-style highlights; and residual revision markup. |
| `core.pdf` | Blank pages, non-embedded fonts, encryption, and general file health. |
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
demonstratives, hollow concluding sentences, caption completeness and numbering, and whether each
substantive title term appears in the body.

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
uv run pytest                                  # 229 tests, no network required
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
