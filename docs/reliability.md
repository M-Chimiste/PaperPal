# Reliability, search, and local verification

This change adds local quality gates. It does not add GitHub Actions or depend on a CI provider.

## Install and verify

Python 3.11 on macOS or Linux is the supported locked environment. The lock keeps the verified local package versions and resolves platform-specific extras. LLMFactory is pinned to commit `1eb5d0d6b3107e2267552f40ba32810b63859d1b`.

```sh
uv sync --locked
cd theseus-ui
npm ci
cd ..
make test-db
make check PYTHON=.venv/bin/python
make test-down
```

An existing `venv/` works with `make check` once its dependencies match the lock and `requirements-dev.txt` is installed. `make test-unit` needs no database. Integration tests recreate only an expendable database with a name ending in `_test`; the default is `theseus_test` on port 5434. Never point them at the application database.

`make check` runs backend tests, the frontend tests/build, API-contract drift checks, the evaluation smoke test, and the lint ratchet. The 291 pre-existing lint diagnostics are fingerprinted by file, rule, message, and source text. New diagnostics fail even if other warnings were fixed. `npm run lint` still reports the full debt. Missing golden files fail unless explicitly generated with `UPDATE_GOLDENS=1` and reviewed.

Update API artifacts with `make generate-api`. Update dependencies deliberately with `uv lock`, reviewing both the constraints in `pyproject.toml` and the resulting `uv.lock`. Regenerate the production export:

```sh
uv export --locked --no-dev --no-hashes --emit-index-url --format requirements-txt --no-emit-project -o requirements.lock
make docker-check
```

`make docker-check` builds a validation image, starts an isolated internal network with a disposable PostgreSQL container, checks startup/authentication/encrypted credential persistence across restart, and cleans up the containers/network. It publishes no host ports and never mounts user data.

The export omits hashes because pip cannot hash-check the pinned Git dependency; package versions and the Git commit remain fixed. Docker includes a compiler for llama-cpp, uses `npm ci`, and limits native compilation to two processes. Linux uses the official CPU PyTorch wheel index by default; macOS retains its MPS-compatible wheels. NVIDIA deployments should maintain an explicit GPU lock/source configuration instead of installing CUDA packages into the default CPU image. The media/document stack still makes the image substantial.

## Access and credentials

Direct unauthenticated access is limited to loopback peers and loopback Host headers. Cross-origin requests require the same origin or an explicit `CORS_ORIGINS` entry. Wildcard origins are rejected.

Set a strong `APP_AUTH_TOKEN` for Docker or remote access. The browser prompts for Basic authentication; the username can be `theseus` and the password is that token. A signed, HttpOnly, SameSite session cookie authenticates WebSockets for 12 hours. Use HTTPS for remote deployments. Rotating the token invalidates sessions. Tokens are never put in URLs or local storage. Container port publishing defaults to `127.0.0.1:8000`. Readiness/liveness probes expose only operational status and do not require authentication.

Credential GET responses expose `configured` and an empty value for secrets; only `OLLAMA_URL` remains readable. Blank updates preserve existing values. Secret replacements use randomized, authenticated Fernet encryption, with a key derived from the required `APP_SECRET_KEY`; the old default key is rejected. Never lose this key when backing up the database. The DELETE endpoint removes a saved value and its current process environment value; values supplied by `.env` or the deployment environment return on restart unless removed there too.

**Legacy migration is explicit because plaintext and XOR ciphertext were both stored without format tags.** Keep a database backup and the original `APP_SECRET_KEY`. Select the correct format; use `--key` repeatedly if the database mixes formats. These commands validate without writing until `--apply` is passed:

```sh
python -m scripts.migrate_credentials --format plaintext --key OPENAI_API_KEY
python -m scripts.migrate_credentials --format plaintext --key OPENAI_API_KEY --apply
python -m scripts.migrate_credentials --format xor --key ANTHROPIC_API_KEY --apply
```

For Docker, run the same command through `docker compose run --rm --no-deps theseus-insight-app python -m scripts.migrate_credentials ...` after building the image and starting the database. This also works when legacy credentials prevent the API from starting.

Already-migrated values are authenticated and left unchanged. Migration is transactional and prints only key names. Existing environment credentials remain usable while legacy values await migration. Untagged stored credentials without an environment fallback cause startup to fail with a migration instruction; the system does not guess the format or silently use corrupted secrets.

## Jobs, recovery, and delivery

Migration 017 adds dispatch, delivery receipts, stage diagnostics, and a paper-vector HNSW index. It can take time on a large library; migrations serialize across API instances and readiness stays false until startup completes.

Supported task insertion creates its dispatch record in the same transaction. Workers poll persistent dispatch, claim with an instance ID and a 60-second lease, and renew every ten seconds. A PostgreSQL session lock prevents an expired lease from being stolen while its worker is still running. Shutdown drains handlers; forced cancellation retains the lock until process exit because cancelling an async wrapper cannot stop an inference thread.

Pending jobs survive restart. Interrupted newsletters, bulk embedding, profile ingestion, and star-map tasks can be recovered automatically. Interrupted operations whose external effects are not safe to repeat (including podcasts and database imports) fail with a review instruction and can be explicitly retried from Run History. Legacy jobs created before durable dispatch are left for review; startup no longer resets every active row or kills processes by name. Judge workers watch their launching PID **and creation time**, and stop claiming work if that owner disappears. Standalone CLI workers remain independent.

Schedulers elect one leader using a database lock. Standby instances can take over after the leader exits. Long runtime alone no longer causes a job to be marked failed.

Newsletter checkpoints live under `data/checkpoints/<task_id>`, use atomic replacement, and validate configuration and prompt hashes before reuse. Database checkpoint job IDs are reused on recovery. Changed configuration/prompts or unversioned checkpoints require a new run rather than silently mixing results.

Email receipts are keyed by task ID. Completed deliveries are skipped on retry. A crash or exception after submission starts leaves an uncertain receipt: automatic replay and SMTP-to-Gmail fallback are suppressed when delivery may already have occurred. This is not a claim of exactly-once SMTP delivery. In **Run History → Diagnostics**, inspect sent mail, then explicitly confirm delivery or confirm it was not sent before retrying. A new task ID represents a new requested delivery.

## Diagnostics and search

`/health/live` checks process liveness. `/health/ready` checks completed startup, active dispatch workers, scheduler leadership/standby, and database connectivity. Essential startup failures stop the app instead of announcing success.

Run History diagnostics show stage completion/failure, elapsed time, and exception type. The API exposes the same data at `/api/tasks/{task_id}/diagnostics`. Run provenance records configuration, prompt, and source-code hashes; `/api/runtime/provenance` identifies the running deployment. Secrets and full model prompts are not included in those records.

Hybrid retrieval combines independently bounded vector and full-text candidate sets using weighted reciprocal rank fusion. Keyword matches can qualify without an embedding. Vector retrieval filters model identity. The API reports `count_scope: retrieved_candidates`; totals are counts of the retrieved candidate window, not exhaustive counts across the corpus. `SEARCH_CANDIDATES` defaults to 500 per branch (100–5000), and `SEARCH_CONCURRENCY` defaults to two. Excess concurrent searches return HTTP 429 with `Retry-After`. Embedding/model initialization and SQL run outside the API event loop. Approximate retrieval must be evaluated for recall under profile/date filters; a fast index does not guarantee sufficient recall.

Run the repeatable synthetic benchmark separately from the regression database:

```sh
python -m scripts.benchmark_search --rows 10000
```

It creates/migrates only `theseus_benchmark_test` by default, seeds synthetic rows, records timings and `EXPLAIN (ANALYZE, BUFFERS)` output, then deletes its rows. Increase `--rows` for scale testing. It refuses database names without `_test`. Synthetic timing is not a production latency guarantee.

## Research quality evaluation

`tests/evaluation/research_v1.json` contains 30 developer-curated foundational ML retrieval questions, ten source-paper summaries/links, a development/holdout designation, and three example research profiles. These are starting judgments, not validated judgments about your personal interests. Confirm corpus coverage and replace/extend them with papers from your library before using scores to tune models.

```sh
make evaluation-smoke
python -m scripts.record_retrieval --base-url http://localhost:8000
python -m scripts.evaluate_research --predictions data/evaluation/predictions.json --output data/evaluation/current.json
python -m scripts.evaluate_research --predictions data/evaluation/predictions.json --baseline data/evaluation/previous.json --min-recall 0.8
```

The recorder makes read-only search requests to the running API, which may initialize its configured embedding model. Supply `APP_AUTH_TOKEN` in the environment when authentication is enabled. Evaluation reports recall, precision, nDCG, mean/p95 latency, cost, and explicit metric coverage. Missing cost or citation judgments are `null`, never zero or an invented quality score. Annotate `citation_reviews` with reviewer-checked `supported: true/false` labels to measure claim support; merely finding a citation is not proof it supports a claim. Preserve model, prompt/config hashes, and code revision in prediction provenance. Baseline comparisons reject different datasets or cutoffs.

The lexical smoke baseline verifies evaluation wiring only. It does not test an LLM, establish citation correctness, or demonstrate a production quality improvement.
