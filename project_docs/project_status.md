# Theseus Insight Project Status

## Last Updated: September 7, 2026

---

## Recent Changes

### Repository reliability, retrieval, and local verification (2026-09-07)

**Implemented:**
- Write-only credential status responses, authenticated versioned encryption, explicit transactional migration for legacy plaintext/XOR values, and local/network HTTP + WebSocket access controls.
- Durable task dispatch created atomically with supported tasks, expiring instance leases plus PostgreSQL execution locks, conservative restart recovery, and explicit retry/delivery-resolution controls in Run History.
- Removed import-time task failure marking and machine-wide startup process killing. Judge subprocesses track the specific launching process; scheduled jobs elect one database-backed leader with standby takeover.
- Newsletter checkpoint directories are scoped by task, writes are atomic, configuration/prompt hashes gate reuse, and database checkpoint job IDs survive retry. Persistent email receipts prevent automatic resend after completion and require explicit review after uncertain submission.
- Fixed concrete handler failures discovered by workflow tests: callback shadowing, `config` versus `config_json`, incompatible newsletter constructor kwargs, and serialization of date/model values in task configuration.
- Bounded, off-event-loop retrieval with cached model initialization; independent semantic/full-text candidates, reciprocal rank fusion, model-identity filtering and an HNSW migration. Candidate totals are explicitly distinguished from exhaustive corpus totals.
- Fail-fast startup, liveness/readiness endpoints, stage timing/error diagnostics, deployment/run provenance, and database-backed WebSocket snapshots for reconnects across instances.
- Python 3.11 macOS/Linux lockfiles, pinned LLMFactory commit, portable local Make targets, independent unit/integration suites, API/schema drift checks, an exact lint-debt baseline, and Docker build fixes. No GitHub workflows were added.
- Added 30 starter research-evaluation queries, example profile relevance judgments, recorded API evaluation, human-reviewed citation metrics, baseline comparison, and an isolated synthetic search benchmark.
- Added `docs/reliability.md` covering installation, verification, access, migration, recovery semantics, and evaluation limitations.

**Verification/debug log:**
- Final `make check` passed: 9 independent Python unit tests and 66 backend integration tests. Lifecycle and scheduler ownership tests used the disposable PostgreSQL cluster at port 5434, never the application database; that cluster was stopped after verification.
- Frontend: 13 tests pass and TypeScript/Vite production build passes; existing large-chunk warning remains.
- Local lint ratchet: no new diagnostics; all 291 pre-existing diagnostics remain visible to the full lint command.
- OpenAPI schema and generated TypeScript checks pass.
- 10,000 synthetic papers in a separate `theseus_benchmark_test` database: ten retrieval runs, median ~34.8 ms, maximum ~37.4 ms, 1,000 fused candidates. This is not a live-library performance claim.
- An initial benchmark overlapped test-database resets and returned zero candidates; that result was discarded. The benchmark now owns a separate test database and rejects unexpectedly empty results.
- The prior orchestration golden test depended on the user's local model configuration. It now uses a deterministic fixture; the user's `config/orchestration.json` edits were preserved.
- Dependency resolution initially included unsupported Windows/Python combinations; the supported runtime is now explicit (Python 3.11, macOS/Linux), with the existing verified package versions retained as constraints.
- Docker validation found the legacy Node image incompatible with test dependencies and a missing C/C++ compiler for llama-cpp. The image now uses Node 22.14, `npm ci`, the pinned Python export, and the required native build tools.
- Container runtime validation then detected an unsupported CUDA wheel on Linux ARM. Linux lock resolution now selects official CPU PyTorch wheels, removing unused CUDA packages while preserving macOS MPS versions.
- Final Docker image built successfully. Isolated container validation passed dependency consistency, PyTorch/Docling/application imports, fresh database migrations, readiness, unauthenticated rejection/authenticated access, and encrypted credential persistence across restart. Temporary containers and their internal network were removed; no host ports or user data mounts were used. The credential migration CLI is included in the image, and build context excludes runtime data and dependency caches.

**Deployment/next:**
- Apply migration 017 on the next application startup; allow time for HNSW index creation on large libraries.
- Set `APP_AUTH_TOKEN` for Docker/network access and migrate legacy credentials using the correct explicit format and original `APP_SECRET_KEY` (see the reliability guide).
- Confirm starter relevance judgments/corpus coverage against personal research profiles before using evaluation scores to tune production models. Real model cost and citation judgments remain unknown until recorded/reviewed, rather than being fabricated.


### Restore Docling layout inference on Apple MPS (2026-07-31)

**Implemented:**
- Pinned `transformers==5.8.1` in `requirements.txt`. Transformers 5.9 and
  newer changed the RT-DETR-v2 positional-embedding implementation to allocate
  `float64` tensors on the target device, which crashes on Apple MPS. Version
  5.8.1 uses the requested input dtype and remains compatible with the current
  Docling, sentence-transformers, and PyTorch requirements.
- Downgraded the recovered local `venv/` from Transformers 5.14.1 to 5.8.1.
  This restores the MPS path; no CPU fallback is enabled.

**Verification and debug log:**
- `uv pip check --python ./venv/bin/python`: all 237 installed packages are
  compatible.
- Direct RT-DETR-v2 MPS smoke test produced a finite float32 positional
  embedding on `mps:0` with shape `(1, 80, 256)`.
- End-to-end Docling conversion of a disposable one-page PDF, with
  `AcceleratorDevice.MPS` explicitly selected, completed with
  `ConversionStatus.SUCCESS`, one page, and zero errors.
- The backend test suite could not start in the restricted test shell because
  its PostgreSQL fixture at `localhost:5434` was not reachable there. This
  occurred during test setup before any tests ran and is unrelated to the
  Transformers change.

**Next:**
- Restart the running backend so it imports Transformers 5.8.1, then retry the
  original newsletter/PDF job.
- Revisit the pin after the upstream RT-DETR-v2 MPS fix is merged and included
  in a Transformers release.

### Local launch helper (2026-07-25)

- Added root-level `start.sh` to launch Uvicorn from the repository directory
  with the recovered `venv/` and `.env`, bound to `127.0.0.1:8000`.
- The script uses `exec` so signals such as Ctrl+C reach Uvicorn directly and
  trigger its normal worker, scheduler, and database-pool cleanup.

### Fresh macOS recovery from verified PostgreSQL snapshot (2026-07-25)

- Verified every file in `TheseusInsight-db-snapshot-20260725T121753Z/` against
  `SHA256SUMS`; all checks passed.
- Verified the separately restored `.env` file and `APP_SECRET_KEY` against the
  snapshot fingerprints without exposing their values; both matched exactly.
- Recreated the archived `theseus` PostgreSQL role and restored `theseusdb` into
  the fresh local PostgreSQL 18.4 cluster with `pg_restore --create
  --exit-on-error --jobs=4`.
- Confirmed all 41 base tables restored, including 277,493 papers. PostgreSQL
  loaded `pg_trgm` 1.6 and upgraded the restored pgvector extension from 0.8.0
  to the locally installed compatible version, 0.8.5.
- Used `uv` to install managed CPython 3.11.15 and create the repository-local
  `venv/`; installed all 237 backend dependencies from `requirements.txt`.
- Installed 624 frontend packages with `npm ci` and completed a clean
  TypeScript/Vite production build.
- Recreated the ignored runtime directories under `data/`.
- Started the backend with `.env` loaded before module import. Startup skipped
  all 17 already-applied migrations, verified all critical tables and the
  `paper_profile_scores` unique constraint, and started the task workers and
  scheduler successfully.
- End-to-end checks passed: `/`, `/openapi.json`, `/api/profiles`, and a
  paginated `/api/papers` request all returned HTTP 200.

### Codegen-drift fix: `auto_tune_batch_size` wired end-to-end (2026-06-10, session 8)

The F2 drift bug is resolved by **adding the field to the backend and honoring it** (chosen over removing the UI toggle because the auto-tune capability genuinely exists in `services/embedding_service.py` and the toggle's tooltip describes it accurately — only the plumbing was missing):

- `api/models.py` — `PerformanceConfig.auto_tune_batch_size: bool = True` (next to `embedding_batch_size`, whose description now notes it applies when auto-tuning is off). POST/GET `/api/trends/performance-config` now persist and return it; old persisted configs default to `True`.
- `data_access/settings.py` — new `SettingsRepository.get_performance_config()` helper (JSON-safe, `{}` on missing/corrupt); the GET endpoint and harvest service both use it. Side effect: a corrupt stored config now falls back to recommended defaults instead of 500ing.
- `services/harvest_service.py` — embedding preflight reads the persisted performance config instead of hardcoding `auto_tune_batch_size=True`; when auto-tune is off, the user's `embedding_batch_size` is passed as `gpu_batch_size`.
- `services/embedding_service.py` — `_load_embedding_model` branch order fixed so auto-tune=False honors the configured batch size on **all** devices (previously MPS forced 256 unconditionally, which would have kept the toggle dead on Apple Silicon). Auto-tune-on behavior unchanged (MPS still pins 256; CUDA/CPU tune on first run).
- Frontend: `openapi.json` + `schema.d.ts` regenerated; `PerformanceConfig` is now a generated-type alias in api.ts (hand-written interface deleted, TODO(codegen) drift mention removed). The generated shape matches exactly — all fields required, including the new boolean.
- New characterization test: `test_performance_config_roundtrip` (POST→GET preserves `auto_tune_batch_size`/`embedding_batch_size`).

Suite: 52 backend tests green; frontend build clean, vitest 11/11.

**Note:** the rest of `PerformanceConfig` (max_cores, hdbscan_n_jobs, clustering_batch_size, etc.) is still persisted-but-unconsumed — no pipeline code reads those fields. `auto_tune_batch_size`/`embedding_batch_size` are the first honored fields.

### Ship of Theseus Refactor — F4 + F5 + B10: ROADMAP COMPLETE (2026-06-10, session 7)

**F4 — task-channel unification:**
- `services/taskChannel.ts` — framework-free WS channel (URL construction, connect/retry-to-max, JSON parse, last-message persistence under the frozen `ws_last_message_${taskId}` keys with the 10-minute restore window). 7 vitest cases incl. retry cancellation and the storage opt-out.
- `useWebSocket` rebuilt as a thin React adapter over TaskChannel — public API unchanged, zero call-site churn for the Newsletter/Podcast/Visualizer pages (which consume it via useTaskState).
- `useMindMap` keeps its domain logic (merge mode, generation colors, multi-channel expand) but both of its hand-rolled WS sites now ride TaskChannel (persistence disabled, as before).
- **Scope note:** useTaskState and useDatabaseTaskState remain separate hooks on purpose — one models a single WS-driven task with stuck-task detection, the other a poll-driven export/import pair. The duplication F4 targeted was connection plumbing, now unified in TaskChannel. Optional follow-ups: ProfileStarMap.tsx and ResearchAgent.tsx still call api.ts's raw createWebSocket (same mechanical swap as useMindMap if desired).

**F5 — Papers react-query:** `hooks/usePapersQuery.ts` (useInfiniteQuery; hybrid-vs-regular branch verbatim; FilterState/SortCriterion live here now). Papers.tsx 1,213 → 1,114 lines, 23 → 15 useState; server state fully in react-query (pagination resets via query key, edits patch the cache, scroll observer calls fetchNextPage). **Deferred F5 stragglers:** ResearchTimeline/RunHistory/Visualizer manual-fetch → useQuery conversions and the FilterPanel extraction (small mechanical items, same pattern).

**B10 — config:** `theseus_insight/config.py` with load_orchestration_config() / get_orchestration_config() (DB → file → {}); task handlers route through it. Deliberately uncached (Settings UI updates the stored config at runtime; caching needs invalidation plumbing = behavior change). Sites with their own missing-config semantics (papers 500s, harvest scripts' raise-on-missing-file) keep them — converting those two-liners is optional polish.

**THE SHIP OF THESEUS REFACTOR ROADMAP IS COMPLETE.** All ten phases landed: Phase 0 (52 backend tests + goldens), B1–B10, F0–F5. Headline numbers: TheseusInsight 3,827 → 708-line facade; bulk_operations router 2,734 → 1,387; api/tasks.py 1,933 → 441; Settings.tsx 2,974 → 213; trends.py 2,015 → 4-module package; ~900 lines of dead code deleted (incl. 3 latent bugs found and fixed: always-False validate_database_connection, crash-on-call search_by_keywords, 500-instead-of-400 profile_ids). Optional polish backlog: ranking-path unification (now module-local in pipeline/ranking.py), remaining codegen TODO types in api.ts, F5 stragglers, inline config-parse two-liners, raw createWebSocket in two pages.

### Ship of Theseus Refactor — B8–B9 COMPLETE: TheseusInsight is a facade (2026-06-10, session 6)

**TheseusInsight: 3,827 → 708 lines.** The class now holds: the frozen 40-param `__init__`, the ~120-line run_async orchestrator calling the seven stage modules, signature-exact delegates, and small helpers (_log_error, _clear_judge_model_cache, _cleanup_temp_data). All four public entry points unchanged.

Session 6 lifts (all verbatim, all with delegates):
- `pipeline/profiles_pipeline.py` (372) + `pipeline/embedding_pipeline.py` (404) — **deliberately NOT unified with pipeline/stages/**: their download/embed blocks differ for real (Kaggle forcing, arxiv category config, existing-papers short-circuit returning a result dict, batched embedding with skip_existing).
- `pipeline/ranking.py` (~700) — all five ranking entry points (historical-scores / single-server / multi-server paths). Kept as separate implementations; the "80% overlap" unification claim needs module-local verification before any merge (optional future work, low priority now that they're isolated).
- `pipeline/profile_scoring.py` (~530) — get_profile_papers, get_and_score_profile_papers, store_papers_without_scoring, handle_no_papers_found (notification email template preserved byte-exact).

The `pipeline/` package now: checkpoints (frozen-contract adapter + compat tests), model_loading, ranking, profile_scoring, profiles_pipeline, embedding_pipeline, stages/{download,embed,rank,newsletter_sections,newsletter_content,email,podcast}.

Suite: 51 backend tests green after every one of the 6 commits.

**Refactor roadmap remaining:** F4 (taskChannel + useTask hook unification — preserve sessionStorage/localStorage keys; useMindMap keeps domain logic), F5 (Papers.tsx useInfiniteQuery + FilterPanel + ResearchTimeline/RunHistory/Visualizer conversions), B10 (config consolidation: single cached orchestration parse, canonical ModelConfig).

### Ship of Theseus Refactor — B8 + B9 run_async decomposition (2026-06-10, session 5)

**God class: 3,827 → 2,662 lines; run_async (1,007 lines) fully decomposed.**

**B8 (peripherals):**
- `pipeline/checkpoints.py` — CheckpointAdapter, byte-compatible with the original methods ({stage}_checkpoint.pkl pickle wrapper, newsletter_generation job_type, {'dataframe': records} encoding, always-dual-write). God class keeps same-name delegates (~48 call sites unchanged). Compat tests: legacy checkpoints load through the adapter and vice versa.
- `pdf/markdown_extraction.py` — streaming download, spawn-subprocess parse runner with timeout/terminate/kill ladder, Docling→MarkItDown fallback, both workers (module-level for spawn pickling).
- `pipeline/model_loading.py` — all five providers; env read at call time.

**B9 (stage extraction):** all seven run_async stages now live in `pipeline/stages/` as `async def run(ti, ...)` taking the TheseusInsight instance as context: download (with exit_early flag for the no-papers path), embed (3 early exits converted to the flag; bulk dedupe + threshold filtering verbatim), rank (incl. the empty-checkpoint poisoning guard and embeddings memory release), newsletter_sections (concurrent PDF pool with rolling submission), newsletter_content (intro retry/backoff + json_repair fallback), email, podcast (visualizer re-run + YouTube publish). run_async is a ~90-line orchestrator. Stage tests cover download resume/early-exit/skip paths without network.

**B8/B9 REMAINING (next session):**
- Ranking unification (rank_papers_with_historical_scores 152 lines vs _rank_papers_single_server 382 — claimed 80% overlap needs verification before merging; both still in the god class along with _rank_papers_multi_server).
- run_profiles_pipeline (372 lines) and run_embedding_only_pipeline (404) decomposition — their Stage 1/2/store blocks are near-copies of the extracted stages; reuse pipeline/stages/ where the code genuinely matches.
- Final facade pass: __init__ is still 279 lines; get_and_score_profile_papers (239) and _handle_no_papers_found (128) still inline.

Suite: 51 backend tests green (incl. new checkpoint-compat + stage tests).

### Ship of Theseus Refactor — F3 Settings decomposition (2026-06-10, session 4)

**Settings.tsx: 2,974 → 213 lines.** The page is now a shell holding only the three shared queries (model catalog, providers, inference-server hosts) and the theme toggle. Seven section components under `theseus-ui/src/components/settings/`, each owning its queries/mutations/state and using the shared snackbar context:
- `ModelConfigurationSettings` (613) — orchestration model tabs; ONE generic renderModelConfigFields drives all nine MODEL_TABS keys; inference-servers tab embeds OllamaServersSettings; owns orchestrationConfig query+mutation.
- `ResearchAgentSettings` (695) — single/multi mode config; the two byte-identical 90-line renderModelFields closures (verified by diff) deduplicated into makeModelFieldsRenderer(configType).
- `DatabaseTransferSettings` (898) — export/import incl. WebSocket progress; sole mount point of useDatabaseTaskState (F4 replaces this hook). Export/import error alerts render above this card now, not page-top.
- `PerformanceSettings` (386), `CredentialsSettings` (98), `ModelNameAutocomplete` (126), `TabPanel` (23).
- Dead state removed: `appPasswordFailed` was never set true — its warning Alert could never render.
- Props threading: modelCatalogData/modelProviders/getHostsByProvider flow from the page into ModelConfiguration + ResearchAgent sections (shared queries stay deduped via react-query cache).

Verified: tsc strict 0 errors, vite build, vitest 4/4, eslint at baseline (291 pre-existing).

**Next per roadmap:** B8–B9 — the god-class decomposition (pipeline/ package, checkpoint formats frozen, one stage per commit, kill-and-resume tests). Then F4–F5, B10.

### Ship of Theseus Refactor — B7 complete + F2 (2026-06-10, session 3)

**B7 part 3 — newsletter service:** `services/newsletter_run_service.py` owns the custom newsletter run (profile/recipient resolution — tag ids UNION explicit ids, unlike papers/trends intersect; cross-thread progress callback; orchestration-config load; multi-server job creation; TheseusInsight run). Router endpoint keeps validation + enqueue (581 → 379 lines).

**B7 part 4 — bulk-operations services:** router 2,734 → 1,387 lines. 24 functions moved: `services/harvest_service.py` (arXiv download-for-range, embedding preflight, profile-filter merging, backfill tasks), `services/bulk_judge_service.py` (submission core incl. the nested download+embed+judge pipeline, worker subprocess launch/monitor/signals, conflict checks), `services/scheduling.py` (APScheduler suspend/restore). Six Pydantic request/response models moved to api/models.py. **Preserved contracts:** papers.py imports `_start_bulk_judge_operation` + `BulkJudgeRequest` from the router; websockets.py imports `get_job_metrics` — all re-exported and verified. Function names kept verbatim (underscores included) for a mechanical diff.

**B7 part 5 — 500→400 fix:** the pinned bugs are fixed — get_papers and get_trending_topics now have `except HTTPException: raise` guards so the intended 400 for malformed profile_ids reaches clients; characterization tests updated to assert 400.

**F2 — OpenAPI→TypeScript codegen:** `scripts/dump_openapi.py` (offline spec dump; needs reachable DB), checked-in `theseus-ui/openapi.json` + `src/services/generated/schema.d.ts` (`npm run generate:api`), `generated/types.ts` re-export shim. **28 of 94 hand-written api.ts types are now generated-type aliases**; 23 name-matching types conflicted under strict tsc and are kept hand-written under a `TODO(codegen)` block in api.ts — re-alias as backend models tighten (Research*/MindMap* families need real Pydantic models instead of Dict[str,Any]; several request models mark defaulted fields required in the schema). **Drift bug found:** Settings.tsx `auto_tune_batch_size` doesn't exist in backend PerformanceConfig — UI toggle silently dropped (spawned as separate task).

Suite: 44 backend tests green; frontend tsc-strict 0 errors, vitest 4/4, eslint at baseline.

**Next per roadmap:** F3 (Settings.tsx 2,974 → ~300 via components/settings/*), then B8–B9 (god-class decomposition — checkpoint formats frozen, kill-and-resume tests gate it), F4–F5, B10.

### Ship of Theseus Refactor — B4–B6 + B7 parts 1–2 (2026-06-10, session 2)

**B4 — async bridge:** `db/async_bridge.py` (`run_repo` = asyncio.to_thread wrapper) adopted on the heaviest read paths: papers DB-level pagination, trends dashboard/timeline fetches. `db/README.md` documents the pool policy (sync psycopg canonical; asyncpg confined to the checkpoint/jobs island — no new users).

**B5 — trends split:** `data_access/trends.py` (2,015 lines, 10 classes) → `data_access/trends/` package (topics, trend_metrics, research_interests, profile_interests) with a re-exporting `__init__` — no caller changes. `_profile_filters.profile_filter_clause` collapsed the 7 triplicated three-way profile-filter SQL branches; the unfiltered dashboard variants' JOIN-omission shape is preserved. **Deleted `TopicsRepository.search_by_keywords`**: zero callers and its SQL raised UndefinedFunction on every call (dead since inception — found by new SQL-validity tests that execute every composed query in all 3 filter modes).

**B6 — TaskManager split:** `api/tasks.py` 1,933 → 441 lines (queue/worker infra, status tracking, TaskStatus, singleton). Nine `run_*_task` bodies → `api/task_handlers/` as free `async (task_manager, task_id)` functions + `HANDLERS` registry keyed by task_type. TaskManager keeps lazy-importing thin delegates so the bound-method references routers pass to `enqueue_task` are unchanged (eager imports would be circular — handlers import TaskStatus from tasks.py). **Fixed during extraction:** config-file fallback in `_common.get_orchestration_config` needed `parents[3]` from the new module depth; both fallback branches now pinned by tests.

**B7 part 1:** the 165-line recursive research-agent serializer moved to `api/helpers/serialization.py` as `serialize_research_object` (router keeps an alias import; smoke-tested incl. cycle detection).

**B7 part 2:** `response_model` annotations on 21 previously-untyped endpoints across actions, embedding_service, database, runs_and_tasks, settings, papers + new generic models (TaskQueuedResponse, StatusMessageResponse, etc.) in api/models.py. **This unblocks F2 codegen.** Route-set golden unchanged. Genuinely-dynamic payloads typed `Dict[str, Any]` honestly; FileResponse endpoints left unannotated.

**B7 REMAINING (next session):** service-layer extraction for `bulk_operations.py` (2,734 lines → services/bulk_jobs_service + harvest_service + scheduling) and `newsletters_and_podcasts.py` (→ services/newsletter_run_service owning TheseusInsight instantiation + event-loop juggling). One router per commit; OpenAPI golden must stay identical. Also still pinned-not-fixed: the 500-instead-of-400 invalid profile_ids responses (papers + trends).

Suite: 44 backend tests green.

### Ship of Theseus Refactor — Phase 0 + B1–B3 + F0–F1 (2026-06-10)

**What changed:** Started the incremental full-stack refactor (plan at
`~/.claude/plans/atomic-dazzling-pearl.md`). Behavior-preserving; every
commit gated on a green characterization suite.

**Phase 0 — test safety net (new, zero prod-code changes):**
- `docker-compose.test.yml`: ephemeral pgvector pg14 test DB on port 5434 (tmpfs). `make test` brings it up and runs pytest; `make test-down` removes it. Docker runtime is colima + standalone docker-compose on this machine.
- `tests/` — 33 characterization tests: migration chain 000–016 on a fresh DB (idempotency, staging tables, Default profile), OpenAPI surface golden (181 routes — fails on any route add/remove/rename), papers list/filter/sort semantics, profile CRUD, orchestration defaults golden, secret-setting XOR ciphertext golden, trends envelopes, pgvector similarity ordering with a fake embedder, pure-function goldens, json_repair judge-response contract.
- Critical mechanics documented in `tests/conftest.py`: DATABASE_URL must be set before any theseus_insight import; the test DB must be up before importing main (import-time DB calls in api/tasks.py and research_agent.py); TestClient used WITHOUT `with` so the lifespan (scheduler + judge_worker kill in startup_cleanup.py) never runs in tests.

**B1 — dead code removed:**
- `inference/llm_legacy.py` (786 lines, zero importers), stray real `.py` sources under `inference/__pycache__/mindmap/`, empty vestigial dirs (`storage/`, `api/background_tasks/`, `api/job_management/`, `api/routers/bulk_operations_core/`).
- `api/dependencies.py` trimmed to CREDENTIAL_KEYS; its `validate_database_connection` imported a nonexistent module (always returned False) and had no callers.

**B2 — cross-cutting helpers:**
- `api/helpers/profile_filtering.py`: parse_id_csv / parse_tag_csv / resolve_tag_profile_ids / merge_id_filters replace the copies in papers, trends (×2), profiles routers. Per-site 400-detail strings and blank-segment strictness preserved via flags. Note: get_by_tags filters is_active in SQL, so papers.py's re-check was redundant.
- `api/helpers/serialization.py`: isoformat_fields / decode_json_fields; the seven per-router `_convert_*_timestamps` are now one-line delegates.
- `data_access/base.py`: build_set_clause(updates, extra=...) replaced 8 hand-rolled dynamic-UPDATE builders (scheduled_tasks ×3, tasks, model_catalog, research ×2, trends topics).
- KNOWN PINNED BUGS (intentionally unfixed, tests document them): GET /api/papers and /api/trends return 500 instead of 400 for invalid profile_ids (blanket `except Exception` swallows the HTTPException) — fix deliberately in B7; a tag matching no profiles resolves to an empty id list which repos treat as "no filter" (all rows returned).

**B3 — harvest dedup:**
- `utils/harvest_common.py`: 7 helpers shared by harvest_and_judge.py and paperswithcode_harvest_and_judge.py (verified identical by AST diff modulo docstrings). The two existence-checkers stay per-script (bulk-set vs per-row implementations). `{stage}.pkl` checkpoint format pinned by tests, including legacy-checkpoint loading.

**F0–F1 — frontend foundation:**
- vitest + @testing-library/react + jsdom (`npm test`), scoped to hooks/services/utils.
- SnackbarProvider + useSnackbar (mounted in App.tsx), ConfirmDialog, useDialogState<T>. Pilot: ModelCatalog.tsx converted (dropped 2 Snackbars, 3 dialog boolean/payload pairs, hand-rolled delete dialog).

**Next steps (per plan):** B4 async bridge → B5 trends.py package split → B6 TaskManager handler split → B7 router service extraction + response_model annotations → F2 openapi-typescript codegen → F3 Settings.tsx decomposition → B8–B9 god-class decomposition (checkpoint formats frozen) → F4–F5 task-hook unification + Papers react-query → B10 config consolidation.

**Verification:** `make test` (backend, 33 tests), `cd theseus-ui && npm run build && npm test` (frontend). OpenAPI golden must stay byte-identical through router refactors; update deliberately only when B7 adds response_model annotations.

### SMTP DNS Fallback For Gmail Delivery (2026-03-14)

**What changed:** Added a fallback DNS resolver for Gmail SMTP so newsletter delivery can recover when macOS/VPN DNS resolution intermittently fails even though Gmail itself remains reachable.

**Backend fixes:**
- `theseus_insight/communication/communication.py`
  - Added a socket-level fallback DNS resolver that queries configurable nameservers directly when `socket.getaddrinfo()` fails for the SMTP host.
  - The fallback nameserver list now supports `tcp://` and `udp://` entries so the app can mirror VPN DNS setups like Hiddify's `tcp://8.8.8.8`.
  - Wrapped Gmail SMTP connection creation in `create_gmail_smtp_session()` so both normal newsletter sends and error-notification sends use the same fallback behavior.
  - Added configurable env vars: `SMTP_DNS_FALLBACK_ENABLED`, `SMTP_DNS_NAMESERVERS`, `SMTP_DNS_TIMEOUT_SEC`, and `SMTP_CONNECT_TIMEOUT_SEC`.
  - Added Gmail API fallback delivery using the saved OAuth token (`gmail_token.json` by default): the mailer now tries SMTP first, then Gmail API over HTTPS, then raises if both fail.
  - Shared the same transport fallback across normal newsletter sends and error-notification sends.
- `scripts/send_test_email.py`
  - Updated the SMTP test script to use the same fallback-aware Gmail session helper as the production mailer.
- `README.md`
  - Documented the new SMTP DNS fallback behavior, Gmail API fallback, and related configuration variables.

### Newsletter PDF Parser Fallback Order (2026-03-14)

**What changed:** Reintroduced Docling for newsletter PDF extraction, but kept the timeout isolation and skip behavior from the MarkItDown-only path.

**Backend fixes:**
- `theseus_insight/theseus_insight.py`
  - Added a Docling PDF-to-markdown subprocess worker for newsletter section generation.
  - Refactored newsletter PDF parsing into a shared subprocess runner with hard timeouts.
  - Newsletter extraction now follows: `Docling -> MarkItDown -> skip paper`.
  - Updated verbose logging so the terminal shows the new fallback order explicitly.
  - Fixed the subprocess wait logic so the parent accepts parser output as soon as it is written to the queue, instead of waiting for the child process to fully exit first. This avoids stalls where Docling logs `Finished converting document ...` but the parent never advances.
- `README.md`
  - Updated the newsletter PDF extraction docs to describe the new `Docling -> MarkItDown` fallback chain.

### Newsletter Intro LM Studio Timeout (2026-03-14)

**What changed:** Added a dedicated, longer LM Studio request timeout for the newsletter intro generation step so large local prompts do not restart after the shared 300-second client timeout.

**Backend fixes:**
- `theseus_insight/utils/lmstudio_client.py`
  - Added support for a per-client `request_timeout_sec`.
  - Included the timeout in the LM Studio client cache key so different timeout profiles do not accidentally reuse the same client.
- `theseus_insight/theseus_insight.py`
  - Added `NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC` with a default of `1200`.
  - The intro timeout parser now accepts `0`, `none`, or `off` to disable the timeout entirely for LM Studio intro generation.
  - Applied that timeout only when loading the `newsletter_intro_model` with provider `lmstudio`.
- `README.md`
  - Documented the new `NEWSLETTER_INTRO_REQUEST_TIMEOUT_SEC` environment variable, including how to disable the timeout entirely.

### Newsletter LM Studio Routing + Scoring Diagnostics (2026-03-13)

**What changed:** Investigated the newsletter ranking path after reports that the queue claimed inference was running while LM Studio saw no requests. Fixed two backend issues and added clearer runtime diagnostics.

**Backend fixes:**
- `theseus_insight/theseus_insight.py`
  - Single-server model loading now respects `ModelConfig.host` for LM Studio and Ollama instead of silently falling back to environment defaults.
  - Added an explicit verbose log when newsletter ranking reuses historical paper scores and therefore does not send fresh judge requests.
- `theseus_insight/workers/judge_worker.py`
  - Newsletter judge workers now preserve the queued `profile_id` instead of dropping it to `None`.
  - Added clearer newsletter worker logs showing `paper_id`, `profile_id`, and `server_url`.
- `theseus_insight/utils/lmstudio_client.py`
  - Replaced LM Studio client construction with a local wrapper that disables ambient proxy/env handling during startup checks and OpenAI-compatible client creation.
  - Verified patched client initialization inside the real `theseus` conda environment against `localhost:1234`.
- `theseus_insight/theseus_insight.py`
  - Updated single-server newsletter rank progress metadata to emit `papers_to_score`, `papers_scored`, `papers_failed`, `papers_pending`, and `papers_in_progress` in the same shape the multi-server UI expects.
- `theseus-ui/src/hooks/useTaskState.ts`
  - Updated the polling fallback path to merge `task.metadata` from `/api/tasks/{id}/status`, so the newsletter stats tiles continue updating even if the WebSocket path lags or reconnects.
- `theseus-ui/src/components/newsletter/StatsGrid.tsx`
  - Fixed stats selection logic so missing multi-server summary fields no longer collapse to `0` before fallback values are considered.
  - Preserved multi-server priority order: `scoring_summary` still wins when present, then single-server metadata, then server aggregates.

**Key findings from diagnosis:**
- Live DB orchestration is using `judge_model = ibm/granite-4-h-tiny` with `model_type = lmstudio`.
- Local LM Studio was reachable on `http://localhost:1234` and exposed `ibm/granite-4-h-tiny` via `/v1/models`.
- Recent multi-server newsletter jobs on March 6, 2026 had already scored thousands of tasks before later being marked failed due to server restart cleanup.
- The newsletter pipeline can legitimately skip fresh LM Studio calls when papers already have historical scores in the database.

### Version 1.0 Release (2025-12-31)

**What changed:** Bumped repository versions to 1.0.0 to mark the first major release.

**Files modified:**
- `setup.py` (backend version)
- `theseus_insight/__init__.py` (backend package version)
- `theseus-ui/package.json` (frontend version)
- `theseus-ui/package-lock.json` (frontend lockfile version)

### Dashboard Overhaul + Profile Star Map (2025-12-31)

**What changed:** Replaced the landing dashboard from a redundant nav card grid into a real command-center, and introduced a new “Profile Star Map” visualization (dashboard preview + dedicated page) to explore ~10k profile papers as a constellation.

**Frontend (theseus-ui):**
- New widget-based dashboard layout with:
  - **Pinned shortcuts** (user-customizable + reorderable via drag-and-drop; stored in localStorage)
  - **Recent outputs** (Research Agent history, Podcast history, and recent completed tasks)
  - **Insights strip** (compact stacked-area chart from profile interest timeline data)
  - **Star Map preview** (sampled starfield + CTA to full view)
- Added dedicated Star Map page at `'/star-map'` with Canvas rendering, zoom/pan, hover tooltips, and click-through to Papers search.
- Added sidebar entry **Star Map** for easy access.

**Backend (theseus_insight):**
- Added cached star map point storage + recompute pipeline:
  - SQL migration `scripts/015_profile_star_map.sql`
  - Automatic migrations list updated to include `014_interest_short_labels.sql` and `015_profile_star_map.sql`
  - New profile endpoints:
    - `GET /api/profiles/{profile_id}/star-map`
    - `GET /api/profiles/{profile_id}/star-map/status`
    - `POST /api/profiles/{profile_id}/star-map/recompute`
  - New websocket stream:
    - `ws://localhost:8000/ws/star-map/{task_id}`

**Files added/modified (high-signal):**
- Frontend:
  - `theseus-ui/src/pages/Dashboard.tsx`
  - `theseus-ui/src/pages/ProfileStarMap.tsx`
  - `theseus-ui/src/components/dashboard/*`
  - `theseus-ui/src/components/starMap/StarMapCanvas.tsx`
  - `theseus-ui/src/services/api.ts` (added `starMapApi`, extended websocket types)
  - `theseus-ui/src/components/Layout.tsx`, `theseus-ui/src/App.tsx`
- Backend:
  - `theseus_insight/data_access/star_map.py`
  - `theseus_insight/star_map/task.py`
  - `theseus_insight/api/routers/profiles.py`
  - `theseus_insight/api/routers/websockets.py`
  - `theseus_insight/db/migrations.py`
  - `scripts/015_profile_star_map.sql`

### Database Import Profile Merging Fix (2025-11-25)

**Problem:** Database imports involving the Default profile weren't properly migrating research interests. When importing a database backup where the Default profile matched the existing one, the comparison logic only checked `arxiv_filters`, `tags`, and `email_recipients` - ignoring the most important data: research interests.

**Root Cause:**
1. Profile comparison (`_compare_profiles`) did not include research interests
2. When profiles "matched", interests were simply skipped rather than merged
3. No mechanism to detect and merge new interests from source into existing profiles

**Solution Implemented:**

#### 1. Enhanced Profile Comparison (`ProfileMapper._compare_profiles`)
- Now compares research interests in addition to arxiv_filters, tags, and email_recipients
- Returns detailed comparison results including:
  - `new_interests`: Interests in source but not in target
  - `existing_interests`: Interests common to both
  - `missing_interests`: Interests in target but not in source

#### 2. Smart Interest Merging
- Added `merge_interests` parameter (default: True) throughout the import pipeline
- When profiles match on core config (arxiv_filters, tags, email_recipients):
  - Profile ID is mapped to existing profile
  - New interests from source are queued for merging
  - Interests are merged using case-insensitive duplicate detection

#### 3. New `ProfileMapper` Capabilities
- `interests_to_merge`: Queue for interests to be merged after profile mapping
- `apply_queued_interest_merges()`: Method to apply all queued interest merges
- `profile_merge_log`: Tracks what happened during merge for reporting
- Support for `smart_merge` strategy that updates profile config and merges interests

#### 4. Updated API
- `/api/settings/database/import` endpoint now accepts `merge_interests` parameter
- Default behavior merges interests when profiles match

**Files Modified:**
- `theseus_insight/utils/db_migration/db_import.py` - Core import logic with interest merging
- `theseus_insight/api/routers/database.py` - API endpoint with new parameter
- `theseus_insight/api/tasks.py` - Task manager passes merge_interests to importer

---

## What Needs to Be Implemented Next

### Short Term
1. **Local media prerequisite**: Install `ffmpeg` before exercising podcast/audio/video generation; the recovered core app works, but pydub currently warns that no `ffmpeg` or `avconv` executable is on `PATH`.
2. **Frontend dependency audit**: Review the 24 advisories reported by `npm ci` (1 low, 6 moderate, 17 high) and update dependencies deliberately rather than running an uncontrolled lockfile rewrite during recovery.
3. **Star Map quality**: Replace deterministic random projection with a higher-quality dimensionality reduction (e.g. UMAP) with caching + versioning.
4. **Star Map semantics**: Add cluster labeling + selection actions (open Papers with filters, export selection, seed Mind-Map).
5. **Dashboard iteration**: Add “Continue where I left off” cards (active jobs/tasks) and/or user-configurable widget visibility.
6. **Newsletter observability**: Surface a clear UI message when ranking is reusing historical scores so “no LM Studio traffic” is expected and visible.
7. **Newsletter task tracing**: Add a backend log/event for worker launch success/failure tied to each newsletter task ID for easier debugging.
8. **UI Update**: Add toggle in Settings/Database import UI to control `merge_interests` behavior
9. **Import Preview**: Show user what interests will be merged before confirming import
10. **Logging**: Add more detailed logging about profile/interest merge decisions

### Medium Term
1. **Interest Similarity Detection**: Use embeddings to detect semantically similar interests (not just exact text match)
2. **Merge Conflict Resolution UI**: When profiles differ significantly, show user options
3. **Profile Version History**: Track changes to profiles over time

---

## Debug Log

### 2026-07-25: Local launch script
- Gracefully stopped the recovery-validation Uvicorn process and confirmed its
  task workers, scheduler, WebSocket manager, and database pools shut down.
- Added and shell-syntax-checked `start.sh`; it resolves its own directory
  before launching, so it also works when invoked from outside the repository.

### 2026-07-25: Fresh OS database and application recovery
- Found a fresh Homebrew PostgreSQL 18.4 cluster running on the local Unix socket
  and TCP port 5432. The `theseus` role and `theseusdb` database were absent, so
  there was no pre-existing target data to overwrite.
- Confirmed Homebrew pgvector 0.8.5 supports PostgreSQL 18 and was available to
  the server. The snapshot originated on PostgreSQL 14.18 with pgvector 0.8.0.
- Verified the complete directory-format archive before restore; every entry in
  `SHA256SUMS` returned `OK`.
- Recreated the role from `theseus-role.sql`, preserving its archived password
  hash, then restored the database with four parallel jobs. `pg_restore`
  completed with exit code 0 and no active restore sessions remained.
- Post-restore database checks found 41 public base tables, 277,493 rows in
  `papers`, and extensions `pg_trgm:1.6`, `plpgsql:1.0`, and `vector:0.8.5`.
- The restored `.env` and `APP_SECRET_KEY` fingerprints matched the values
  recorded before the OS reinstall, preserving access to encrypted settings in
  the database.
- Apple Command Line Tools supplied only Python 3.9.6, below the repository's
  declared Python 3.10+ requirement. Used `uv 0.11.32` to download CPython
  3.11.15 and build `venv/`, matching the Dockerfile's Python 3.11 runtime.
- `uv pip install -r requirements.txt` resolved and installed 237 packages,
  including the Git-based LLMFactory dependency and a locally built
  `llama-cpp-python`.
- `npm ci` installed 624 packages. npm reported 24 existing dependency
  advisories and noted deferred install-script approvals for `esbuild` and
  `fsevents`; the subsequent production build nevertheless completed
  successfully (13,752 modules transformed).
- Backend startup initially spent about one minute importing the ML stack, then
  completed normally. Migration verification reported 0 applied, 17 skipped,
  all critical tables present, and the required profile-score constraint intact.
- pydub emitted the only runtime warning: `ffmpeg`/`avconv` is not installed.
  Core API and frontend serving are unaffected; media generation remains to be
  validated after adding `ffmpeg`.
- HTTP smoke tests returned 200 for the built React frontend, OpenAPI document,
  restored profile data, and restored paper data.

### 2026-03-13: Newsletter LM Studio investigation
- Confirmed live DB orchestration still points the newsletter judge at LM Studio (`ibm/granite-4-h-tiny`).
- Confirmed local LM Studio responded successfully to `/v1/models` and advertised the configured judge model.
- Confirmed a discrepancy in the Python runtime: `http://localhost:1234/v1/models` succeeds, while `http://127.0.0.1:1234/v1/models` does not on this machine.
- Identified that `TheseusInsight._load_inference_model()` ignored configured `host` values for single-server LM Studio/Ollama newsletter runs.
- Identified that newsletter judge workers were discarding queued `profile_id`, which could break downstream aggregation and make queue activity harder to interpret.
- Added a verbose log line when ranking reuses historical scores so skipped inference is explicit.
- Hardened the local LM Studio client wrapper to avoid environment/proxy interference and keep the client pinned to `localhost:1234`.
- Identified that single-server newsletter ranking emitted incomplete metadata for the stats tiles and that the UI polling fallback was not refreshing metadata at all.
- Aligned single-server rank metadata with the multi-server stats shape and patched the polling fallback to merge fresh metadata from task status responses.
- Identified a frontend bug in `StatsGrid`: `safeNumber(undefined) -> 0` was causing the component to always choose `0` from missing `scoring_summary` fields before reaching valid single-server fallback values like `papers_scored` and `papers_pending`.
- Fixed `StatsGrid` to choose the first defined numeric source instead of the first coerced zero, which should preserve multi-server behavior while restoring single-server counters.
- Confirmed from live browser payloads that single-server rank updates were already carrying correct top-level metadata (`papers_scored`, `papers_pending`, etc.) while the cards still showed zero.
- Updated the single-server rank callback to emit the same `scoring_summary` object shape as the multi-server monitor (`completed`, `failed`, `pending`, `in_progress`, `total`, `pending_plus_in_progress`).
- Hardened `StatsGrid` to ignore all-zero/stale summaries and synthesize a normalized summary from top-level metadata, server stats, and parsed rank messages when needed.
- Verified the frontend production build after the stats normalization change.
- Added LM Studio/Qwen no-thinking support in the local LM Studio wrapper.
- The wrapper now defaults `LMSTUDIO_DISABLE_THINKING` to enabled, forces `use_thinking=False` for Qwen models, and injects the `/no_think` directive into the latest user message as a request-level fallback.
- This is scoped to LM Studio Qwen models and is intended to keep newsletter generation from getting derailed by chain-of-thought output while normal API providers are unavailable.
- Documented the new LM Studio/Qwen thinking toggle in `README.md`, including the `LMSTUDIO_DISABLE_THINKING` environment variable and LM Studio configuration guidance.
- Added a second safeguard for LM Studio Qwen responses: if the model still returns `<think>...</think>` blocks in chat-completions mode, the local wrapper now strips those blocks before returning the text to newsletter/podcast parsing code.
- Investigated newsletter PDF-stage locking on macOS and found the backend was eagerly importing the podcast visualizer stack (`pygame`) even for newsletter-only runs.
- This interacted badly with Docling/OpenCV (`cv2`) and produced duplicate SDL Objective-C class warnings immediately after PDF processing began.
- Fixed the import graph so `PodcastGenerator` is loaded lazily only when podcast generation is actually requested, including lazy access from `theseus_insight.__init__` and `api/tasks.py`.
- Verified the edited files with `python -m py_compile` and confirmed `import theseus_insight` succeeds in the `theseus` environment without triggering the previous eager podcast import path.
- Observed a second newsletter PDF-stage stall where the first PDF completed but the second paper hung before Docling reported detected formats.
- Updated the newsletter section generator to stop passing remote PDF URLs directly into Docling; it now downloads each PDF to a temporary local file with explicit HTTP timeouts and then converts the local file.
- This should allow slow or problematic remote PDF URLs to fail fast instead of blocking the entire newsletter section loop.
- Switched newsletter section PDF extraction away from Docling entirely and onto the existing `MarkitdownDocProcessor`, preserving the same downstream summarization flow while avoiding the Docling-specific hangs seen during section generation.
- Added a hard per-PDF subprocess timeout for newsletter section extraction using MarkItDown. Corrupted or wedged PDFs now time out and get skipped instead of blocking the entire newsletter generation task.
- The timeout is controlled by `PDF_CONVERSION_TIMEOUT_SEC` and currently defaults to 120 seconds.
- Documented the newsletter PDF extraction timeout behavior and the `PDF_CONVERSION_TIMEOUT_SEC` environment variable in `README.md`.
- Added bounded parallel PDF downloading for newsletter section generation. PDFs are now downloaded concurrently, processed in completion order, and once enough sections are produced the remaining pending downloads are cancelled.
- Download progress is now logged in the terminal before parsing begins, so slow network transfers are visible separately from the MarkItDown parsing timeout window.
- Added `scripts/send_test_email.py`, a small utility that loads Gmail settings from `.env` and sends a one-off test email for SMTP/DNS troubleshooting.
- Verified edited Python files with `python -m py_compile`.
- Observed that the user’s Hiddify DNS settings already prefer Google Public DNS (`8.8.8.8`) remotely, which matches the new SMTP fallback strategy.
- Added a direct DNS fallback in the Gmail mailer so SMTP hostname resolution can bypass flaky system/VPN resolver state and still connect to Gmail by IP while preserving TLS hostname validation.
- Extended the fallback resolver to understand `tcp://` nameserver entries after noticing the user's Hiddify configuration prefers Google DNS over TCP.
- Updated the standalone SMTP test script to use the same fallback-aware session helper as the main application.
- Expanded `scripts/send_test_email.py` into a fuller Gmail delivery diagnostic tool that now tests system DNS, fallback DNS, raw TCP connectivity, HTTPS reachability, SMTP STARTTLS on 587, SMTPS on 465, and optional authenticated send attempts.
- Updated `scripts/send_test_email.py` to accept a checked-in local OAuth client config file (`gmail-secret.json` by default, overridable via `GMAIL_CLIENT_SECRET_FILE`) in addition to the Google OAuth env vars.
- Ran the expanded diagnostics outside the sandbox on the user's active network/VPN path. Results:
  - system DNS resolution for `smtp.gmail.com` succeeded
  - fallback DNS resolution via `tcp://8.8.8.8` also succeeded
  - HTTPS to `https://mail.google.com` succeeded
  - raw TCP to `smtp.gmail.com:587` and `smtp.gmail.com:465` timed out
  - SMTP handshake and authenticated send attempts on both 587 and 465 timed out
- Conclusion from the live diagnostics: the current network/VPN path appears to allow Gmail over HTTPS but blocks or blackholes Gmail SMTP submission, so a DNS-only fix is insufficient.
- Completed the Gmail OAuth flow and created `gmail_token.json` with a refresh token and the `gmail.send` scope.
- Confirmed that a direct raw `requests.post()` call to the Gmail API send endpoint succeeds immediately on the current network path.
- Updated `theseus_insight/communication/communication.py` so newsletter email delivery now follows the requested sequence: SMTP first, Gmail API fallback second, then raise if both transports fail.

### 2025-12-31: Version 1.0 Bump
- Bumped versions in `setup.py`, `theseus_insight/__init__.py`, `theseus-ui/package.json`, and `theseus-ui/package-lock.json` from 0.9.x to 1.0.0.

### 2025-12-31: Landing Dashboard + Star Map Implementation
- Audited existing dashboard; identified it duplicated sidebar navigation and included a broken timeline path.
- Implemented widget-based dashboard with pinned shortcuts, recent outputs, insights strip, and star map preview.
- Added Star Map page (Canvas + d3-zoom + quadtree hit-testing) and wired routing/sidebar.
- Added backend cached point table + recompute task using TaskManager, plus `/api/profiles/{id}/star-map*` endpoints and websocket stream.
- Verified TypeScript lint on edited files and Python syntax via `compileall`.

### 2025-12-31: Star Map numerical stability fix
- Observed runtime warnings during projection (`divide by zero/overflow/invalid in matmul`) caused by non-finite or extreme embedding vectors.
- Hardened star map recompute to:
  - Skip embeddings containing NaN/Inf
  - Skip embeddings with extreme magnitudes
  - Normalize vectors to unit length before projection
  - Drop any points that still become non-finite after projection

### 2025-12-31: Star Map 3D + dashboard preview fit
- Fixed dashboard Star Map preview rendering so it fits/centers on the point cloud (correct DPR rendering + fit-to-bounds mapping).
- Extended star map cache + API to support a Z coordinate for 3D visualization:
  - Added migration `scripts/016_profile_star_map_3d.sql` (adds `z` column)
  - Updated star map recompute to project embeddings to **3D** (x/y/z) and normalize each axis.
- Updated Star Map page to support **3D (rotate/zoom)** and added an Insights panel:
  - Quick stats + top “constellations” (dominant interest labels + counts)
  - 2D/3D toggle

### 2025-11-25: Database Import Investigation
- Traced through `db_import.py` to understand profile mapping flow
- Found `_compare_profiles` only compared 3 fields, not interests
- Identified that `import_profile_research_interests` relied on `profile_id_mapping` but didn't handle merge case
- Implemented comprehensive fix with smart interest merging
- Updated API to expose merge_interests option
- No linting errors after changes

---

## Architecture Notes

### Profile Import Flow
```
1. import_from_archive() extracts tar.gz
2. import_from_directory() orchestrates import
3. Pre-loads interests data for smart merging
4. For each profile:
   a. ProfileMapper.map_profile() compares with existing
   b. If profiles match: map ID + queue new interests for merge
   c. If profiles differ: create new profile
5. apply_queued_interest_merges() adds new interests to matched profiles
6. import_profile_research_interests() handles remaining interests with ID mapping
```

### Profile Matching Logic
```
Profiles "match" if ALL of these are equal:
- arxiv_filters (JSON object)
- tags (JSON array)
- email_recipients (JSON array)

Note: Research interests differences do NOT prevent a match.
Instead, new interests are merged into the existing profile.
```
