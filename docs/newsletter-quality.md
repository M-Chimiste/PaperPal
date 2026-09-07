# Newsletter quality with local models

The newsletter uses configured local models for editorial selection, evidence extraction, writing, and verification. Configure local providers for the judge, content extraction, newsletter sections, and newsletter intro roles in `config/orchestration.json`. The evaluation CLI accepts LM Studio and Ollama and loads their host configuration from `.env` or explicit model settings. Test-only temperature overrides do not change saved application settings; the real-paper test below used 0.3.

The local deployment command remains `./start.sh`. Migration 018 runs at startup and creates a small metadata table; it does not rebuild the vector index. Draft-only evaluation does not start the app or run migrations.

## What an issue contains

Two featured papers receive up to 280 words of editorial text each; subsequent briefs receive up to 160. Each has **What Changed**, **Why it matters**, **Evidence**, **Caveat**, and a paper link, in that order. A single source-notes appendix at the end groups short original excerpts and evidence IDs by paper. Citations link to paper-scoped HTML anchors, and each notes group links back to its summary. Internal navigation requires a renderer or email client that preserves HTML anchors; external paper links remain ordinary links. Those notes are additional to the editorial word budget. Full source text, extracted claims, model/configuration provenance, and accepted briefs are retained in the edition artifact and per-paper checkpoints.

Paper headings are numbered directly: **1. Paper title**, **2. Paper title**, and so on. The source appendix repeats the same number and title for each group. There is no separate “Paper P1” label above a title. The layout is:

```text
Issue summary covering all included papers

1. First paper title
   Read paper
   What Changed
   Why it matters
   Evidence
   Caveat

2. Second paper title
   ...

Source notes
   1. First paper title
      P1·E1 (S1): source excerpt
   2. Second paper title
      P2·E1 (S2): source excerpt

~Theseus Insight
```

`P2·E1` means evidence item E1 from paper 2; evidence numbering restarts per paper. `S2` identifies an extracted text chunk, not a PDF page. Visible excerpts contain at most 25 words plus an ellipsis; the saved evidence artifact contains the full quotation and source text. Horizontal rules separate the summary, papers, appendix, and signoff.

The preamble summarizes all included papers in connected prose, covering contributions, findings, and boundaries with paper references (P1, P2, etc.), and undergoes a support review. Coverage validation requires every paper. If synthesis fails validation, it uses an extractive prose overview of all contributions rather than bullets covering only the first three. It never falls back to arbitrary raw model output. The signature appears at the end of the issue.

Citation targets have both `name` and `id` attributes: Gmail webmail supports named anchors more broadly than ID-only targets. Some mobile email clients still ignore local jumps ([compatibility tests](https://www.caniemail.com/features/html-anchor-links/)). Visible references include paper-specific labels such as P2·E4 so readers can locate the matching appendix entry even without link support. Markdown escaping precedes HTML escaping, keeping apostrophes and entities readable while escaping untrusted HTML.

## Selection and checks

- The first local judge supplies a shortlist of up to four times the requested paper count. A second local editorial assessment scores relevance, evidence strength, novelty, and usefulness. A strong match to one interest is sufficient; matching every interest is not required.
- The profile database fallback now considers the entire requested date window, rather than only the latest 100 rows. Large windows can consequently take longer to score.
- Editorial selection penalizes repeated topics, overlapping titles, and papers covered in the past 30 days for the same profile(s). Repeat coverage is a penalty, not an absolute ban. History starts with issues produced by this version; historical issues have no reliable paper-ID mapping.
- PDF downloads overlap, but briefs are generated in editorial order. A faster download cannot displace a higher-ranked usable paper.
- Source chunks retain exact text and character offsets. Quotes must occur in their claimed chunk. Non-verbatim quotations are retried, then discarded individually and recorded in `discarded_quotes`; independently verified claims remain usable. A paper still needs a supported problem and contribution. Chunk IDs are **not page numbers**. Oversize or insufficient text fails explicitly instead of being silently truncated.
- Strict schemas are supplied to local providers. Invalid extraction/brief output retries once. A separate local pass checks every field against its evidence and quotes; failed support gets one revision before the paper is rejected.
- Request-local output budgets cap extraction at 4,096 tokens, briefs at 1,800, reviews at 1,200, and editorial scoring at 500 (or the configured limit when smaller). Intro generation/review also have separate caps. Shared clients and saved settings are not modified. These are output limits, not guarantees of total runtime.
- Papers that fail parsing or quality checks are replaced with the next candidate. If every candidate fails, the run fails before generating an issue. A shorter validated issue explicitly reports its coverage.
- Quality events and rejection reasons appear in task diagnostics. Evidence and briefs are checkpointed independently. An interrupted run reuses accepted work; edition insertion is idempotent.

A model review and a matching quote do not prove factual correctness. Human assessment remains necessary to measure actual quality improvements. Selection judgments made from abstracts are provisional; source review is a later step.

## Draft-only evaluation (never sends email)

This tool does not instantiate the newsletter pipeline or an email client, send email, or write to the application database. It reads `.env` for the configured local model host. It writes only new local files and refuses to overwrite results.

A small synthetic fixture is included to exercise the pipeline; it is not a research benchmark:

```sh
venv/bin/python -m scripts.newsletter_eval generate \
  --fixture tests/evaluation/newsletter_smoke.json \
  --temperature 0.3 \
  --output data/evaluation/newsletter-smoke.json
```

On the configured local Gemma model, this synthetic input passed the full extraction/writing/review/intro path at evaluation temperature 0.3, taking about ten minutes. An earlier temperature-1.0 attempt was rejected after retries. This is one compatibility check, not a general temperature recommendation or a production latency benchmark; saved application settings were left unchanged. Expect additional local inference time from the new editorial and review passes.

Freeze a newly generated issue and an explicit interests file:

```sh
venv/bin/python -m scripts.newsletter_eval snapshot \
  --newsletter-id 53 --interests config/research_interests.txt \
  --output data/evaluation/frozen-issue.json
```

Use the actual issue ID. Issues generated before evidence artifacts were introduced cannot be exported this way. The snapshot query uses a read-only database connection.

Compare settings on identical papers and source text:

```sh
venv/bin/python -m scripts.newsletter_eval generate \
  --fixture data/evaluation/frozen-issue.json --temperature 1.0 \
  --output data/evaluation/variant-1.json
venv/bin/python -m scripts.newsletter_eval generate \
  --fixture data/evaluation/frozen-issue.json --temperature 0.3 \
  --output data/evaluation/variant-2.json
venv/bin/python -m scripts.newsletter_eval blind \
  data/evaluation/variant-1.json data/evaluation/variant-2.json \
  --output data/evaluation/blind-review
```

`--reuse-evidence` holds extraction fixed to compare only writing/synthesis. Without it, extraction is rerun on the frozen source. Temperature overrides affect only the evaluation clients, not saved application settings. The output records configuration/prompt/code hashes, model names, temperature, and elapsed time. Comparisons require identical fixture hashes. The blind directory contains shuffled Markdown and styled HTML drafts, a separate answer key with latency, and a ratings template. Rate factual support, specificity, relevance, readability, and non-repetition from 1–5 after checking source evidence.

```sh
venv/bin/python -m scripts.newsletter_eval report \
  --ratings data/evaluation/blind-review/ratings.json \
  --output data/evaluation/human-results.json
```

Unrated dimensions remain unknown and coverage is reported. A synthetic smoke test or local model approval is not a measured improvement on real newsletters. Evaluation drafts are never emailed automatically.

Generation and blind comparisons also write styled HTML previews alongside Markdown. Previews use the same pure renderer as email, including blue heading rules, typography, and horizontal section dividers; rendering does not initialize an email client or send anything.

## Refresh a saved draft after a layout change

Run from the repository root. This example uses the existing local five-paper artifact and writes a new preview without inference, database access, or email. It requires the saved `preamble` and validated `papers` fields; the `data/evaluation` artifacts are local outputs and may not exist in another checkout.

```sh
venv/bin/python - <<'PY'
import json
from pathlib import Path
from scripts.newsletter_eval import save_preview
from theseus_insight.pipeline.newsletter_quality import assemble_newsletter

directory = Path('data/evaluation/five-paper-local')
issue = json.loads((directory / 'newsletter.json').read_text())
content = assemble_newsletter(issue['preamble'], issue['papers'])
save_preview(directory / 'layout-preview.json', content)
PY
```

This writes `layout-preview.md` and `layout-preview.html`; `save_preview` does not write JSON. Choose a new basename for subsequent previews because existing files are not overwritten. Add an issue title/date to the preamble before assembly if desired. Historical artifacts without a saved preamble need their introduction supplied explicitly.

## Real-paper test and email checks

The September 7, 2026 test selected five papers with stored scores of 10/10 from the active profile. Eight candidates were examined: two were filtered for relevance and one failed brief support review. The five included PDFs yielded 51 text chunks and 407 claims with source-matching quotations; 32 non-verbatim quotations were excluded after retries. Quotations matching text does not establish that every paraphrased claim follows from them.

The test exposed scope and baseline overstatements that passed the local reviewer. Three briefs were regenerated locally with editorial feedback, including focused evidence selection for two. An initial synthesized introduction overclaimed long-context retrieval benefits and was replaced with contribution excerpts. The current draft later received a newly generated, reviewed prose summary covering all five papers. This is a locally generated draft with editorial correction, not evidence that unattended review is reliable.

Local artifacts in `data/evaluation/five-paper-local/` include:

| Artifact | Purpose |
| --- | --- |
| `newsletter.html`, `newsletter.md`, `newsletter.json` | Current layout and reviewed content |
| `newsletter-automatic.*` | Original automatic draft, before editorial corrections |
| `test-report.md`, `test-manifest.json` | Run details, hashes, model settings, and review findings |
| Per-paper directories | Downloaded PDFs, parsed Markdown, evidence, rejected quotations, and saved drafts |
| `personal-email-test*-receipt.json` | Recipient, message ID, content hash, and mail-server acceptance for explicitly requested personal tests |

The test used the existing local Gemma model at temperature 0.3. Cached requests and debugging reruns make its elapsed time unsuitable as a clean throughput benchmark. Application settings and the production database were not modified by the test.

Two personal emails were explicitly requested and accepted by SMTP. The first exposed escaped apostrophes and nonworking ID-only reference jumps. The second contained corrected escaping, named anchors, and a new summary. The final matching numbered headings were added afterward and have not been emailed in those tests. Mail-server acceptance does not verify inbox rendering or link behavior; the receipts identify which HTML version was submitted.

### Sending is a separate operation

The evaluation CLI never sends mail. Standard `GmailCommunication` distribution delivery includes the sender as well as configured recipients; it is not a strict single-recipient test path. The personal tests used an explicit one-address SMTP envelope with no CC, BCC, profile-recipient lookup, or extra sender copy.

`scripts/send_newsletter.py` is a legacy database-newsletter sender. Its title-based reference reconstruction does not account for the current numbered headings and source appendix, and its normal delivery path includes the sender. It needs updating before it is used for this layout or a strict one-recipient test. The retained ad hoc test runners are local run artifacts, not a supported sending CLI.

## Implementation and verification

| File | Responsibility |
| --- | --- |
| `theseus_insight/prompt/newsletter_quality.py` | Extraction, writing, editorial, summary, and review instructions |
| `theseus_insight/pipeline/newsletter_quality.py` | Schemas, source checks, editorial feedback, numbered rendering, source appendix, and assembly |
| `theseus_insight/pipeline/stages/newsletter_sections.py` | Candidate processing and per-paper checkpoints |
| `theseus_insight/pipeline/stages/newsletter_content.py` | Reviewed summary, shared assembly, and edition persistence |
| `theseus_insight/communication/rendering.py` | Shared pure Markdown-to-HTML renderer |
| `scripts/newsletter_eval.py` | Draft generation and human-comparison CLI |
| `tests/unit/test_newsletter_quality.py` | Evidence, summary coverage, escaping, layout, and citation-target regressions |

Run the focused checks with `venv/bin/python -m pytest tests/unit/test_newsletter_quality.py -q`. Structural tests verify matching numbered headings, unique `name`/`id` targets, reference resolution, and readable escaped text. These checks do not replace an actual email-client rendering test. Remaining quality work includes stronger evidence-claim entailment, baseline/scope handling, reviewer calibration, and less redundant citation selection.
