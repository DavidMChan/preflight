# preflight

**Catch the things that get papers desk-rejected, before a chair does.**

`preflight` reads a submission PDF and checks it against a conference's rules: page limits,
required sections, margins, fonts, anonymity, appendix placement, hidden text aimed at automated
reviewers, reference validity, and the venue's responsible-research checklist. It tells you exactly
what is wrong and exactly where — `page 5, x=34.0pt, expected >= 69.0pt` — rather than
"formatting may be wrong".

It ships with a profile for **ACL Rolling Review (ARR)** and one for **NeurIPS**, and conferences
are YAML files, so adding your own venue does not mean touching the code.

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

## Why it is built this way

Three tiers, and the tool never confuses them:

| Tier | What it is | How it reports |
|---|---|---|
| **Deterministic** | Measured off the PDF's geometry, fonts and text boxes. Page size, margins, page limits, section order, column layout. | `ERROR` — a documented desk-rejection condition, with the measurement that proves it. |
| **Heuristic** | Pattern and model judgement. Anonymity leaks, prompt injection, checklist coverage, undersized table text. | `WARNING` — always. A university name in related work is legitimate; only a human can rule. |
| **Advisory** | Reviewer lenses: triage, experimental design, statistics, figures, prose, ethics. | `WARNING`, never more. Judgements about quality, not venue rules. |
| **Out of scope** | Originality, dual submission, citation coverage, OpenReview metadata, reviewer registration. | `N/A` — listed explicitly, so a green run never reads as "this submission complies". |

A clean run means *the PDF-checkable rules pass*. It does not mean your paper is compliant, and the
report says so.

Everything that can be measured is measured. A model is used only where the question is genuinely
semantic — does this Limitations section introduce new results? is this URL identifying? — and every
such finding names the model that produced it.

The model-backed checks also report their own coverage. If a paper ever had to be trimmed to fit the
model input, the run says exactly how much went unread, because a confident verdict on a document
the model only half saw is worse than no verdict.

---

## Install

```bash
git clone <this repo> && cd preflight
uv sync --extra hallucinator     # drop the extra to skip reference validation
```

Python 3.12 (the `hallucinator` wheels are 3.12-only). Then create a `.env`:

```bash
cp .env.example .env
# OPENAI_API_KEY=sk-...
# PREFLIGHT_LLM_MODEL=gpt-5.6-luna
```

---

## Use

```bash
uv run preflight paper.pdf                      # pick a conference interactively
uv run preflight paper.pdf -c arr -t short      # short paper limits
uv run preflight paper.pdf -c neurips           # a different venue
uv run preflight paper.pdf -c iclr -t camera_ready  # ICLR, 10-page camera-ready limit
uv run preflight paper.pdf --offline            # deterministic checks only, no network
uv run preflight tui paper.pdf                  # interactive
```

There is no default venue. Without `-c`, preflight opens a picker (arrow keys, or a number)
listing the bundled profiles and starting on the one you chose last; in a script or a pipe it
exits with a usage error instead of guessing.

Model-backed checks and reference verification are **on by default**. Turn them off individually
with `--no-llm` / `--no-refcheck` / `--no-scores`, or all at once with `--offline`. Without an API
key the model checks skip themselves and say so; the deterministic report is unaffected.
`--hallucinator` additionally runs the deprecated third-party backend.

Every run is cached, so you never pay twice for detail:

```bash
uv run preflight show                           # re-read the last run, in full
uv run preflight show min_font_size             # one finding, all its evidence
uv run preflight show --list                    # what is cached
```

```bash
uv run preflight conferences                    # list bundled profiles
uv run preflight checks -c arr                  # list what will run
uv run preflight paper.pdf --json report.json --markdown report.md
uv run preflight *.pdf --quiet --strict         # batch; non-zero exit on warnings too
```

Exit codes: `0` clean · `1` errors · `2` warnings under `--strict`, or a bad file/profile.

### The TUI

`preflight tui paper.pdf` gives findings on the left, full evidence on the right.
`r` re-runs · `l` LLM · `s` scores · `b` reference validation · `v` verbose · `n` next PDF ·
`e` export Markdown · `q` quit.

---

## What it checks

### Deterministic — `core.geometry`, `core.fonts`, `core.structure`

- **Paper size** against the style's dimensions (A4 for ACL, US Letter for NeurIPS and ICLR).
- **Margins**, measured from text and image bounding boxes, with the style's own tolerances.
  Line numbers in anonymous templates are exempt, because they belong in the margin.
- **Column layout** — gutter width for two-column venues, and for single-column ones (NeurIPS,
  ICLR) that the body was not set in columns — from a document-wide column model.
- **Body font size** — the modal size across the document — and **minimum font size**, which
  separates figure labels, table cells and running prose and reports each differently.
- **Content page limit.** This is the check that cannot be a page count: the CFP grants unlimited
  space after the conclusion, so the tool finds the first unlimited section (Limitations,
  references, acknowledgements, ethics, appendix) and counts the content pages *before* it.
- **Limitations section** — present, titled correctly, positioned before the references, and not
  quietly carrying new methods or results.
- **Appendix** — after the references, and still double-column.

### Anonymity — `core.anonymity`

Title-block scanning for names, affiliations, emails and ORCIDs; URL classification into forbidden
(tracking services and link shorteners), identity-bearing, and properly anonymised hosts; self-
reference phrasing; and PDF metadata, annotations and embedded filenames. URLs inside the
bibliography are ignored — they are other people's links.

Every finding here is a warning with its confidence stated, except an email in the title block.

### Hidden content — `core.hidden`

Invisible text (render mode 3/7, zero opacity), microscopic text, text in the page colour, text
outside the CropBox, and prompt-injection patterns.

The injection patterns match text that *addresses* an automated reviewer, not text that *discusses*
one — a paper about prompt injection legitimately quotes these strings, and a checker that cries
wolf on its own subject matter is worse than no checker. Visible matches warn; only concealed
matches escalate to an error. Pages dominated by one image are treated as scans, so an OCR text
layer is not mistaken for concealment.

### Responsible NLP checklist — `arr.responsible_nlp`

All eighteen ARR checklist items (A1–A2, B1–B6, C1–C4, D1–D5, E1), each asking whether the paper
contains the information the item tells you to point at. Sections B, C and D are gated on what the
paper actually did, decided by one applicability call.

The checklist is filled in on OpenReview, not attached to the PDF, so nothing here inspects
checklist *answers*. No single item is ever an error — the CFP is explicit that answering "no" with
a justification is fine. The summary check reports systematic failure, which is the desk-rejectable
condition.

Questions live in [`src/preflight/conferences/rnlp/`](src/preflight/conferences/rnlp/), one YAML
file each. Adding or amending one is a data change.

### NeurIPS checklist — `neurips.checklist`

Presence, position (after references and appendices), completeness of the sixteen answers, leftover
template placeholders, missing justifications, and consistency against the paper body. Mandatory for
NeurIPS. ARR includes the module in optional mode: it never complains that a checklist is missing,
but validates one that is present.

### Reference verification — `integrations.refcheck`

Verifies every entry in the bibliography. Four tiers, and a reference leaves as soon as one of
them answers:

1. **Cache.** The same references recur across drafts.
2. **Bulk databases, fully concurrent.** CrossRef, OpenAlex and DOI resolution are issued for
   every reference at once under per-host rate limits. DBLP, arXiv-by-id and Semantic Scholar are
   opt-in via `refcheck.sources`: arXiv asks for a three-second gap between requests, so firing it
   at a whole bibliography serialises the run behind it, and OpenAlex indexes arXiv anyway.

   Rate limits are adaptive. A fixed rate is a guess about what a service tolerates; when the guess
   is wrong, a refusal halves that host's rate and clean responses earn it back. Dropping a host
   instead would push every reference it could have answered into a far more expensive tier.
3. **Route by what is actually cited.** A GitHub repository is checked against the GitHub API, a
   model card against the Hugging Face Hub, a package against PyPI, a blog post against its own
   URL. Asking CrossRef about a repository and then reporting "not found in any bibliographic
   database" is how a checker manufactures an accusation.
4. **Live web search.** Whatever is left goes to the model's `web_search` tool under scholarly
   domain filters, and comes back with a citation URL. This is the tier a database-only checker
   has no answer for, and it is exactly where such checkers spend their time.

Entries are segmented from the bibliography's hanging indent and parsed by a model in concurrent
batches with a strict schema, because a wrong title becomes a false accusation two tiers later.

Three things guard the entry count. The region ends at the **first heading after the references**,
whatever it is called — papers letter their appendices ("A Main Experiments"), and looking only for
a heading called "Appendix" runs the bibliography to the end of the file and turns every table row
into a citation. Each segment is then shape-checked: a citation carries a year or a locator and is
mostly words, where a table row is mostly numbers. And entries are **deduplicated** by DOI, arXiv
id or normalised title before any lookup, so a work cited twice is verified once and the repeat is
reported.

Extracted URLs are repaired before use. PDF extraction breaks long URLs across lines, producing
`https: //host/path` and `https://develo pers.openai.com/x`; these are rejoined at the line break
and validated, and anything that is still not an absolute URL with a real host is discarded rather
than requested.

**Matching needs more than a title.** "Attention Is All You Need" and "Is Attention All You Need?"
are different real papers with identical bags of words, so an accept also requires the authors to
agree; a close-but-not-certain match goes to the web-search tier rather than being reported.

On the same 62-entry bibliography from a real submission:

| | hallucinator | `refcheck` |
|---|---|---|
| Wall clock, cold | 24 refs in 301s — 62 never finishes inside a sane budget | **62 refs in 39s** |
| Wall clock, warm | — | **1.7s**, identical verdicts |
| False "not found" | blog posts, model cards, vendor announcements | **none** |
| Flagged | 6 of 12, mostly wrongly | 3 of 62, all genuine author-order questions |

Entries resolved as: 47 verified in a database, 12 resolved to a live source (repositories, model
cards, announcements), 3 with an author mismatch worth a look. Parsing the bibliography with a
model adds about 16s on top of the cold number and is what makes the type routing possible.

**Set `refcheck.mailto` to your email address.** CrossRef and OpenAlex both serve identified
clients from a faster pool; without it they throttle, lookups fall through to the expensive tiers,
and the run is both slower and less accurate. The check tells you when this has happened.

A miss is still reported as **suspicion, not proof** — indexes are incomplete, and "not found"
never means "fabricated". It is a warning by default and never an error.

### Reference validation — `integrations.hallucinator` (deprecated)

The original third-party [hallucinator](https://github.com/gianlucasb/hallucinator) backend, kept
for a second opinion and opt-in behind `--hallucinator`. It is superseded by `refcheck` above: it
queries databases in a fallthrough with global rate limits, so unmatched references cost ~45s
each, and it has no notion of what is being cited, so repositories and blog posts get reported as
missing papers.

Its framing is worth preserving and is carried through into the replacement: what a reference
checker produces is a lead to check, not a verdict.

### File and manuscript integrity — `core.markup`, `core.pdf`, `core.citations`, `core.statistics`, `core.abbreviations`, `core.crossrefs`, `core.headings`, `core.figure_quality`

Twenty-two more counted checks, no model involved, about a second for all of them:

| Module | What it checks |
|---|---|
| `core.markup` | PDF comment annotations (which also leak the commenter's name), highlight annotations and `\colorbox`-style highlights, and revision markup left in the file. |
| `core.pdf` | Blank pages, non-embedded fonts, encryption, and basic file health. |
| `core.citations` | Every in-text citation resolves to a bibliography entry, and every entry is cited somewhere. |
| `core.statistics` | p-values reported as zero (`p = .000`), mixed p-value conventions, and "N of M (X%)" recomputed. |
| `core.abbreviations` | Abbreviations spelled out at first use, and one abbreviation to one expansion. |
| `core.crossrefs` | Every float is referred to in the text and every callout resolves; figure panels agree with their captions; appendix and supplement identifiers resolve and are unique. |
| `core.headings` | Numbered section hierarchy is contiguous and well-nested; no empty sections. |
| `core.figure_quality` | Effective DPI of raster figures, and whether figure colours survive greyscale printing or a deuteranopia transform. |

Every one is a **warning**, never an error — none is a documented desk-rejection condition. A venue
that wants one harder promotes it in its own `severity:` map.

Some rules cannot be checked from a PDF at all, and the tool says so rather than approximating:
tracked changes, structured variable declarations, and whether a supplementary *file* exists all
need the submission package. Where a partial substitute exists it is scoped honestly — the revision
check reports "no markup visible in the PDF", not "the source is clean" — and where the PDF's lost
table structure would make a check unreliable (recomputing arbitrary totals, mutually-exclusive
percentage groups) it is deliberately not implemented rather than shipped noisy.

### Reviewer lenses — `llm.review`, `llm.prose`, `llm.audit`

Thirteen audits, each a reviewer's lens applied to the whole paper. They are **advisory by
construction** — none of them can produce an error, because they are judgements about quality and
presentation, not measurements of a documented rule.

| Audit | Lens |
|---|---|
| `busy_reviewer` | An area chair with 80 papers who reads only the abstract, figures and conclusion. Returns Accept / Borderline / Reject and the cheapest change that would move it. |
| `surface_triage` | Title, abstract and contribution claims — the surface a tired reviewer reads first. |
| `ai_prose_signals` | Prose that reads generic or machine-written, reasoned from measured statistics rather than impression. |
| `typos_grammar` | Genuine spelling and grammar errors, ruthlessly filtered for PDF-extraction artifacts. |
| `experiment_claims` | Can the experiments answer the claims? Baselines, ablations, external validity. |
| `experiment_statistics` | Seeds, variance, error bars, paired tests, multiple comparisons. |
| `experiment_leakage` | Splits, contamination, benchmark leakage, tuning on the test set. |
| `claims_vs_numbers` | Do the abstract's numbers match the tables, and do the words match the numbers? |
| `figure_table` | Whether each caption stands alone: metric, data, units, direction of better. |
| `formula_audit` | Dimensions, indices, bounds, sign conventions, undefined symbols. |
| `complexity_claims` | Asymptotic claims — which variable actually governs, and in what regime. |
| `security_ethics` | Data rights, PII, dual use, and claim words like "secure" and "anonymous". |
| `systems_performance` | Latency, throughput, speedup and cost claims, when the paper makes any. |



Each is a YAML file in [`src/preflight/conferences/audits/`](src/preflight/conferences/audits/):
a prompt, a choice of structured context (captions, equations, measured prose statistics, or the
triage surface), and one shared response shape. Adding a lens is a data change.

The hard part of each spec is ruling out what a good paper legitimately does. `formula_audit` is
told that extracted maths is mangled and to report nothing it cannot establish from the prose;
`systems_performance` is told that most papers make no performance claims and returning nothing is
the correct outcome. On a real submission both of those correctly return "does not apply".

### Prose diagnostics — `core.prose`, `core.figures`, `core.surface`

Counted, not judged, and no model involved: a sentence-length histogram against a hard 46-word
break rule, connective-tic frequencies, runs of three or more sentences opening the same way, vague
demonstratives, hollow wrap-up sentences, caption completeness and numbering, and whether every
load-bearing title word actually appears in the body.

These exist because the style advice they encode is measurable, and measuring beats eyeballing.

### Reviewer-style scores — `llm.semantic`

A rehearsal review: soundness, excitement, clarity and reproducibility scored out of 5 with
justifications, plus strengths, weaknesses and concrete desk-reject risks. Built from the title,
abstract, introduction and Limitations, and labelled as an estimate from a partial read.

---

## Conferences are configuration

A conference is a **composition of check modules** plus the settings those modules read. Profiles
are YAML with single inheritance, so a venue starts from a shared base and changes only what
differs.

```
base.yaml          venue-agnostic defaults, every module
└── acl.yaml       A4, two-column, 11pt Times, ACL tolerances
    └── arr.yaml   ARR page limits, mandatory Limitations, ARR checklist
neurips.yaml       US Letter, single-column, 10pt, NeurIPS checklist mandatory
└── baylearn.yaml  BayLearn abstracts: NeurIPS format, 2 pages, no checklist
iclr.yaml          US Letter, single-column, 10pt, 9 pages (10 for camera ready)
```

```yaml
conference:
  key: myvenue
  name: My Venue
  extends: acl              # inherit ACL geometry and typography

tracks_replace: true        # <key>_replace drops the inherited value instead of merging
tracks:
  long: { content_page_limit: 12 }

modules:                    # appended to the parent's set
  - neurips.checklist

disabled_checks:
  - neurips_checklist_present

severity:                   # soften or harden a shared check without forking it
  margins: warning
  limitations_present: error
```

```bash
uv run preflight paper.pdf -c ./myvenue.yaml
```

Three mechanisms make modules reusable across venues:

- **`modules`** decides which check modules run. They compose down the inheritance chain.
- **`severity`** remaps a check's failure severity per venue. `PASS` and `SKIPPED` are never
  rewritten — an override expresses how much a *failure* matters, not whether one happened.
- **`disabled_checks`** removes an individual check while keeping the rest of its module. This is
  how ARR makes the NeurIPS checklist optional rather than forbidden.

Numeric tolerances belong in the profile, never in Python. The ACL numbers come from the official
formatting specification as encoded in
[aclpubcheck](https://github.com/acl-org/aclpubcheck) — A4 is 595×842pt, text starts no higher than
y=57, no further left than x=71, ends at least 71pt from the right edge, and the bottom 62pt are
reserved. The NeurIPS numbers are derived from the style file's text block and are marked in the
profile as needing verification against the current `neurips_*.sty`.

---

## How a run is scheduled

Checks run in three phases:

1. **Deterministic** — pure computation over the parsed PDF. Milliseconds, in order.
2. **Concurrent** — everything that waits on a network. The model-backed checks run on the async
   OpenAI client; the blocking bibliographic lookups go to a worker thread. All of it overlaps.
3. **Aggregate** — checks that summarise what the earlier phases found.

On a 23-page submission this takes the model-backed work — 22 concurrent checks including all
eighteen checklist items and the scoring pass — from **82 seconds to 20**. Tune with
`--concurrency` (default 8). Reference validation runs alongside it and is usually what the run
ends up waiting on.

A full run of 104 checks — every checklist item, all thirteen audits, the counted prose
diagnostics, the semantic checks and scoring — takes about 40 seconds once the reference lookup
is excluded or cached.

Because checks overlap, progress reports what is **still running** rather than what last finished,
and a long check can publish its own sub-status:

```
[52/54] references 34/62: Attention Is All You Need
[52/54] references 34/62: rate limited by arxiv, waiting 3s
```

The Responsible NLP prompts are built so the paper text is byte-identical across all eighteen calls,
with only a short item-specific block appended, keeping a long shared prefix the provider can serve
from cache. There is a test asserting that prefix stays identical.

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

Rules the existing checks follow:

- Read every threshold from the profile via `ctx.conf("dotted.path", default)`. Nothing venue-
  specific gets hard-coded.
- Build findings with `ctx.ok` / `ctx.warn` / `ctx.error` / `ctx.skip`.
- Attach `Evidence` with measurements. It is the part an author acts on.
- `ERROR` only for what you can prove from the PDF. If the discriminator is fragile, warn.
- Declare `requires=("enable_llm",)` for anything that needs a model, and make it `async def`.
- A broken check is caught and reported as `SKIPPED`; it never sinks the run.

---

## Development

```bash
uv run pytest                                  # 152 tests, no network needed
uv run ruff check src tests
uv run python scripts/try_question.py A1 paper.pdf    # one checklist item
```

Tests build their own synthetic PDFs, so nothing depends on a file outside the repo. `uv run preflight checks -c arr` lists all 104 checks and which module each belongs to.

---

## What this tool will not tell you

Originality and prior publication · concurrent submission · whether you cited the right work ·
whether your author list is correct · resubmission metadata · whether your checklist answers are
*true* · whether AI use was disclosed accurately · your co-authors' reviewer registrations.

These are reported as `N/A` on every run, by design. A checker that silently omitted them would let
a green report imply more than it earned.

---

## Credits

Formatting tolerances from [aclpubcheck](https://github.com/acl-org/aclpubcheck) (ACL).
Reference validation by [hallucinator](https://github.com/gianlucasb/hallucinator), whose
"suspicion, not proof" framing is carried through into this tool's output.
