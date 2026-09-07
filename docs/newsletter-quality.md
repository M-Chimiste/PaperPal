# Newsletter quality with local models

The newsletter now uses the configured local models for editorial selection, evidence extraction, writing, and verification. Keep the current LM Studio/Ollama models; no model replacement or temperature change is required. The local deployment command remains `./start.sh`. Migration 018 runs at startup and creates a small metadata table; it does not rebuild the vector index.

## What an issue contains

Two featured papers receive up to 280 words of editorial text each; subsequent briefs receive up to 160. Each has **What Changed**, **Why it matters**, **Evidence**, **Caveat**, and a paper link, in that order. A single source-notes appendix at the end groups short original excerpts and evidence IDs by paper. Citations link to paper-scoped HTML anchors, and each notes group links back to its summary. Internal navigation requires a renderer or email client that preserves HTML anchors; external paper links remain ordinary links. Those notes are additional to the editorial word budget. Full source text, extracted claims, model/configuration provenance, and accepted briefs are retained in the edition artifact and per-paper checkpoints.

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

`--reuse-evidence` holds extraction fixed to compare only writing/synthesis. Without it, extraction is rerun on the frozen source. Temperature overrides affect only the evaluation clients, not saved application settings. The output records configuration/prompt/code hashes, model names, temperature, and elapsed time. Comparisons require identical fixture hashes. The blind directory contains shuffled Markdown drafts, a separate answer key with latency, and a ratings template. Rate factual support, specificity, relevance, readability, and non-repetition from 1–5 after checking source evidence.

```sh
venv/bin/python -m scripts.newsletter_eval report \
  --ratings data/evaluation/blind-review/ratings.json \
  --output data/evaluation/human-results.json
```

Unrated dimensions remain unknown and coverage is reported. A synthetic smoke test or local model approval is not a measured improvement on real newsletters. Evaluation drafts are never emailed automatically.

Generation and blind comparisons also write styled HTML previews alongside Markdown. Previews use the same pure renderer as email, including blue heading rules, typography, and horizontal section dividers; rendering does not initialize an email client or send anything.
