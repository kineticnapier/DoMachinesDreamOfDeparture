# Consolidated Issue Fix Batch

This branch is the single working branch for accumulating verified fixes to open issues in DoMachinesDreamOfDeparture.

## Workflow

- Before each change, check open/closed issues and existing pull requests to avoid duplicate work.
- Implement fixes on this branch and add focused regression tests.
- Reference each addressed issue in the consolidated pull request and relevant commit messages.
- Do not claim an issue fixed until code and tests support it.
- Do not automatically merge the pull request.
- Preserve the repository's existing behavior and compatibility except for explicitly documented fixes.

## Applied on this branch — verification pending

- Refs #17 — `src/dmdod/research/stock_runtime_parity_v01.py`:
  normalize known Stock judgement aliases without changing runtime scoring.
  Added 12 parametrized cases in `tests/test_stock_runtime_parity_v01.py`.
  - Run: `python -m pytest -q tests/test_stock_runtime_parity_v01.py`
  - Status: code and tests committed; **tests not executed**. No live ADOFAI parity claim.
- Refs #22 — `src/dmdod/planning/persistent_fingering_scheduler_v01.py`:
  quarantine physical KeyDown before `arm()`, including a single axis,
  a two-contact thumb, and a two-axis chord.
  Added 3 parametrized cases in `tests/test_persistent_fingering_scheduler_v01.py`.
  - Run: `python -m pytest -q tests/test_persistent_fingering_scheduler_v01.py`
  - Status: code and tests committed; **tests not executed**. No physical Motor validation claim.

- Refs #8 — `src/dmdod/research/stock_runtime_trace_v02.py`:
  replace repeated future-event tuple slices with three reverse-pass lookups,
  preserving the earliest strictly later event in each category.
  Added two regression cases in `tests/test_stock_runtime_trace_v02.py`
  covering missing matches, same-session ordering, multiple sessions and
  a 2,400-event no-slice sequence.
  - Run: `python -m pytest -q tests/test_stock_runtime_trace_v02.py`
  - Status: code and tests committed; **repository pytest not executed**.
    Separate isolated reverse-lookup equivalence checks passed (2 tests);
    actual JSONL runtime and memory performance are unmeasured.
  - Classification: behavior-preserving evaluator-only refactor; no change to
    HUD-only observations, motor timings, checkpoints or frozen reports.

#17 and #22 are correctness changes, not behavior-preserving refactors; #8 is a behavior-preserving refactor.
Do not mark either issue fixed or use `Fixes #...` until regression checks pass.
Frozen research reports, HUD-only constraints, timing constants and checkpoint
formats were not intentionally changed.

## Additional changes awaiting repository test verification (2026-10-28)

- Refs #13 — Reject non-finite commands in 2K and N-Key Motor before physics advances,
  preserving existing finite-value clipping and integration. Added both-side,
  both-trace-path, and non-finite regression cases.
  - Run: `python -m pytest -q tests/test_motor_env.py tests/test_n_key_capacity.py`
  - Status: code and regression tests committed; **repository tests not executed**.
  - Classification: correctness fix for invalid inputs, not behavior-preserving.
- Refs #14 — Track finalized misses by target index rather than timestamp,
  preserving one judgement per equal-time target and repeated finalize() behavior.
  Added same-time, all-missed, and distinct-time regression tests.
  - Run: `python -m pytest -q tests/test_timing_evaluator.py tests/test_rhythm_env.py`
  - Status: code and regression tests committed; **repository tests not executed**.
  - Classification: evaluator record correctness fix; hit assignment unchanged.

No training, live ADOFAI, performance or GPU measurement was performed for these changes.
Do not use issue-closing keywords before repository regression tests and applicable verification.

## Additional fixes awaiting repository verification (2026-10-28)

- Refs #16 — `src/dmdod/features/stock_visible_judgement_feedback_v01.py`:
  reject privileged notifications for `DLL_INTERNAL_COUNT_ONLY` and
  `TEST_FIXTURE_ONLY` categories, while keeping confirmed counters and
  the existing frame schema unchanged.
  Added regression cases in `tests/test_stock_visible_judgement_feedback_v01.py`.
  - Run: `python -m pytest -q tests/test_stock_visible_judgement_feedback_v01.py`
  - Status: source and tests committed; **repository tests not executed**.
  - Classification: HUD-only observation contract correction; not behavior-preserving.
- Refs #7 — `src/dmdod/training/human_visible_curriculum.py`:
  verify recorded `early`, `keydowns`, and `overloaded` on resume,
  while allowing old checkpoints that did not record those fields.
  Added mismatch and legacy compatibility regression cases in
  `tests/test_human_visible_curriculum.py`.
  - Run: `python -m pytest -q tests/test_human_visible_curriculum.py`
  - Status: source and tests committed; **repository tests not executed**.
  - Classification: resume validation correctness fix; saved checkpoint
    schema, optimizer state and ranking rules unchanged.

No training, live ADOFAI verification, or performance measurements were performed.

## Additional unverified fix — Refs #18 (2026-10-28)

- `src/dmdod/envs/real_chart.py` and `src/dmdod/envs/rhythm.py`: stop automatic Miss accounting at the first failure, guard repeated expiry, and break the evaluator event loop after TooLate.
- `tests/test_real_chart_env.py` and `tests/test_rhythm_env.py`: four regression tests cover Fail and NoFail modes, repeated expiry, and remaining targets.
- Run: `python -m pytest -q tests/test_real_chart_env.py tests/test_rhythm_env.py tests/test_stock_floor_state_machine.py`.
- **Repository tests not run** (execution environment cannot resolve github.com to obtain the full repository); training, real ADOFAI parity and performance checks not run. Correctness change; preserve historical research results.

## Additional unverified fix — Refs #21 (2026-10-29)

- `src/dmdod/training/human_visible_curriculum.py`: accept empty AdamW
  moments only for a pristine `selected-best` checkpoint with
  `baseline-validated`, epoch=best_epoch=0 and an empty history.
  Keep missing/invalid states and all trained/progress empty states rejected.
- `tests/test_human_visible_curriculum.py`: three CPU regression tests for
  pristine baseline resume, invalid empty/malformed states, and a progress
  checkpoint referencing an unstepped baseline best.
- Run: `python -m pytest -q tests/test_human_visible_curriculum.py`.
- **Repository tests not executed** (the runtime cannot resolve github.com to
  retrieve the complete repository); no training, ADOFAI runtime checks,
  checkpoint migration, or performance measurements were performed.
- Classification: narrow resume correctness fix, not a behavior-preserving
  refactor. Existing optimizer parameter-group validation and checkpoint
  schema are unchanged. Keep `Refs #21` until verified.


## Additional unverified fix — Refs #3 (2026-10-29)

- `src/dmdod/training/config.py`: reject identical, resolved-path, symlink,
  and hardlink aliases between `[run].checkpoint` and `[run].output`
  before training begins, without changing stored path spellings.
- `src/dmdod/training/runner.py`: repeat the check at `_prepare()`
  for callers that construct `TrainingConfig` directly, before model loading.
- `tests/test_training_config.py`: test same path in three training modes,
  normalized `..` aliases, hardlinks, symlinks (skip when unsupported),
  distinct output, and direct-runner validation.
- Run: `python -m pytest -q tests/test_training_config.py`.
- **Repository tests not executed**: the container cannot resolve github.com
  to fetch the complete repository. No training, checkpoint overwriting,
  live ADOFAI runtime, GPU, or performance measurement was performed.
- Classification: source checkpoint protection / correctness change, not
  behavior-preserving for previously accepted destructive configurations.
  Training math, checkpoint format, saved path spelling, and HUD-only
  constraints remain unchanged. Keep `Refs #3` until verified.

## Additional unverified fix — Refs #4 (2026-10-29)

- `src/dmdod/adofai/chart.py`: share the existing UTF-8/UTF-16 BOM and
  loose-JSON decoder between full chart parsing and document inspection.
- `src/dmdod/data/tuf.py`: use that decoder for direct/ZIP chart acceptance;
  keep legacy `pathData` recognizable until parser preflight rejects it.
- `tests/test_tuf_dataset.py` and `tests/test_tuf_dataset_preflight.py`:
  add direct, ZIP, malformed, legacy and parser-compatible loose-JSON cases.
- Run: `python -m pytest -q tests/test_adofai_chart.py tests/test_tuf_dataset.py tests/test_tuf_dataset_preflight.py`.
- **Repository tests not executed** (GitHub DNS unavailable in the execution
  container); no real TUF downloads, training, ADOFAI runtime, GPU or
  performance checks. Dataset membership can change for previously rejected
  valid charts; do not overwrite historical selections or reports.
- Classification: input acceptance correctness fix. Full chart parsing,
  HUD-only constraints, checkpoint schema and judgement timing unchanged.


## Additional unverified fix — Refs #11 (2026-10-29)

- `src/dmdod/connectome/fly_policy.py`: clone the reused no-grad input
  projection before the in-place recurrent update, so a previously returned
  state is never overwritten by the next frame's input projection.
- `tests/test_fly_connectome_policy.py`: add CPU multistep no-grad and
  deterministic-action regression tests comparing grad-enabled steps and
  `forward_sequence()`, checking previous-state immutability and confirming
  the scratch buffer remains absent from the checkpoint state dict.
- Run: `python -m pytest -q tests/test_fly_connectome_policy.py`.
- **Repository pytest not executed** (container cannot resolve github.com to
  fetch the full repository). A separate minimal PyTorch matrix example
  reproduced the aliasing and verified that cloning avoids it; this is not
  a repository regression test.
- **Not run:** actual model training, ADOFAI runtime comparison, CUDA,
  performance/memory benchmarks. The clone adds an allocation per step;
  throughput and GPU memory impact remain unmeasured.
- Classification: **inference correctness fix, not behavior-preserving**.
  Model weights, checkpoint keys, HUD-only input, Motor and judgement timing
  are unchanged. Historical inference benchmarks must not be overwritten.
  Keep `Refs #11` until repository tests and impact checks pass.


## Refs #19 — Rhythm synthetic target cursor optimization (2026-10-29)

- Implementation: `src/dmdod/envs/rhythm.py` now uses monotone unresolved, per-key, and expiration cursors plus a remaining-target count. It preserves sorted-target ordering, strict expiration boundaries, TooEarly non-consumption, TooLate and auto-Miss accounting, fail-on-first-Miss semantics from #18, and existing reward/done logic.
- Regression coverage: four tests added in `tests/test_rhythm_env.py` for interleaved keys and non-consuming TooEarly, fail/no-fail and reset, 2,000 ordered targets, and per-step equivalence with a full-scan reference. Check against full-scan results and count invariants.
- Source commit: `441f2329dc3952aedc4cea456e33b61cd205c223`. Test commits: `59cf6985cc8b9f62010e2cccb759000205abe433`, `834ff65c5a524f277a3b4c37198021c25ef117f7`.
- Verify with: `python -m pytest -q tests/test_rhythm_env.py tests/test_perception.py`; compare step-by-step reward, cue, judgement history and termination with the old implementation. No benchmark has been run.
- **Unverified:** repository pytest, training, actual ADOFAI runtime, CUDA/GPU and performance/memory measurements were not executed. This is an intended behavior-preserving performance change, but equivalence still requires verification. Keep Issue #19 open and PR #23 Draft; use `Refs #19`, not `Fixes #19`.


## Batch update — Refs #15 (2026-10-29)

- **Refs #15** — Replace repeated sort-on-insert and full-queue scans in
  `StockFrameStateMachine` with a stable `heapq` priority queue, preserving
  timestamp then insertion ordering, the `1e-12` frame cutoff, and
  `pending_keydowns` presentation order.
- Source commit: `dcad5a032a59a944a556ff5ab54222c1e19b8066`.
  Regression tests commit: `d77a2aef9f49349a8a13e0d3cd94a20deabcbd09`.
- Tests added to `tests/test_stock_frame_state_machine.py`: prequeued
  out-of-order and equal-time inputs, frame cutoff tolerance, and 1,200
  prequeued inputs with stable ordering.
- Run: `python -m pytest -q tests/test_stock_frame_state_machine.py tests/test_stock_floor_state_machine.py tests/test_stock_runtime_parity_v01.py`.
- **Repository pytest NOT run** (GitHub host unavailable to execution container).
  An independent 1,000-scenario comparison of the old sorted-list selection
  and new heap selection passed; this is **not** a repository regression result.
  Live ADOFAI, training, GPU and performance/memory measurements NOT run.
- Classification: intended behavior-preserving performance refactor, but
  repository-level equivalence and actual speedup remain unverified.
  Keep Issue #15 open, PR #23 Draft, and use `Refs #15`, not `Fixes #15`.

## Partial implementation — Refs #5 (2026-10-29)

- Shared `RealChartMotorEnv._all_targets_resolved()` uses the existing
  monotone `_next_target_index()` cursor instead of scanning all resolved
  targets from index zero on each step.
- Applied in `src/dmdod/envs/real_chart.py`,
  `src/dmdod/motor/capacity.py` (8K), and
  `src/dmdod/envs/contact_chart_16k.py` (12-actuator/16-contact).
- Added 6 parameterized cases (2 tests × 3 environments) in
  `tests/test_real_chart_resolution_cursor.py` for partial resolution,
  all-resolved termination, mixed used/missed, reset, and cursor/full-scan
  equivalence after each monotone resolution.
- Source commits: `30466218`, `23d6f08a`, `1a85965f`;
  tests commit: `075ae5a9`.
- Run: `python -m pytest -q tests/test_real_chart_resolution_cursor.py tests/test_real_chart_env.py tests/test_prestige1_autoplay_coupled_chart_audit.py`.
- **Partial:** teacher capacity diagnostic `_unresolved_tail()` still
  materializes the entire suffix on every step; no change was made to its
  reservation/teacher input semantics. Issue #5 stays open.
- **Not verified:** repository pytest could not run (container cannot resolve
  github.com). Independent Python monotone cursor-vs-full-scan comparison
  passed 111,600 updates, which does not validate simulator integration.
  No training, live ADOFAI, GPU, memory or performance benchmarks were run.
- Classification: intended behavior-preserving performance refactor, pending
  repository regression and performance verification. No change to HUD-only
  inputs, judgement boundaries, rewards, checkpoints or frozen results.
  Keep PR #23 Draft and use `Refs #5`, not `Fixes #5`.


## Refs #1 — song-time parity must be explicitly calibrated (2026-10-29)

- `src/dmdod/research/stock_runtime_parity_v01.py`: an observed
  `song_time_s` no longer counts as a match. Without explicitly supplied
  alignment offset and tolerance, the time field is `UNKNOWN` and live
  evidence cannot become `RUNTIME_PARITY_CONFIRMED`. With both parameters,
  compare the time residual to the tolerance; mismatches are explicit.
- `scripts/audit_phase33_stock_runtime_parity_v01.py`: optional paired
  `--song-time-offset-s` / `--song-time-tolerance-s` arguments; report the
  supplied values. These parameters are **not proof of actual calibration**.
  Derive them from independent live timing evidence, never the virtual 10ms grid.
- Also preserve a valid `0.0` SwitchChosen song time instead of falling
  back to the category event's timestamp (`None`-only fallback).
- `tests/test_stock_runtime_parity_v01.py`: synthetic qualified-shaped live
  session exercises missing alignment, match, mismatch, missing timestamp,
  and invalid parameter pairs. This fixture is **not a live capture**.
- Run: `python -m pytest -q tests/test_stock_runtime_parity_v01.py tests/test_stock_runtime_trace_v02.py`.
- **Not verified:** repository pytest, Unity frame-phase calibration, actual
  ADOFAI runtime trace, training, CUDA, and performance/memory measurements
  have not been run. This is a fail-closed verdict correctness change, not
  behavior-preserving; no historical results or checkpoints are overwritten.
  Keep Issue #1 open and PR #23 Draft. Use `Refs #1`, not `Fixes #1`.


## Batch update — Refs #10 (2026-10-29)

- Source: `src/dmdod/training/human_visible_curriculum.py` records ordered
  Train chart SHA256 and start/end times, and checks them before restoring
  the optimizer or writing a resumed checkpoint.
- `src/dmdod/training/config.py` introduces
  `[human_visible].allow_legacy_train_identity_resume = true` as an explicit
  opt-in for old checkpoints with no Train identity or inherited unverified
  Train history. Default is false. Resumed checkpoints keep a persistent
  `human_visible_train_identity_verified = false` when prior Train history
  cannot be proven, even after current anchors are recorded.
- Regression tests: `tests/test_human_visible_curriculum.py` checks SHA,
  time window, order, count, tolerance, malformed identity, legacy opt-in,
  selected-best/progress pre-restore rejection, and serialized provenance; `tests/test_training_config.py` checks
  the new boolean configuration.
- Run:
  `python -m pytest -q tests/test_human_visible_curriculum.py tests/test_training_config.py`.
- **Repository pytest NOT executed** (GitHub DNS inaccessible in container).
  No actual training, checkpoint resume/save integration, live ADOFAI,
  GPU or performance measurements were executed.
- Classification: resume correctness and reproducibility fix, not a
  behavior-preserving refactor; it deliberately refuses formerly unverified
  resume configurations. Model weights, optimizer update math, HUD-only
  observation, timing and frozen research results remain untouched.
  Issue #10 stays open and PR #23 remains Draft; `Refs #10` only.

## Batch update — Refs #9 (2026-10-29)

- Source: `src/dmdod/adofai/stock_floor_state_machine.py`, `targets_from_playable_segment(segment, *, initial_floor=None)`.
- Explicit evaluator-only landed floor for a sliced segment is validated against the last input-bearing floor strictly before the segment start. Earlier ordinary floors no longer trigger a false omitted-floor flag when the caller provides the correct starting state.
- Default `None` preserves the old conservative behavior; actual in-segment missing ordinary floors, unsupported branches and Midspin bridges remain checked. Existing callers remain unchanged. Pass the same initial floor to `StockFloorStateMachine`.
- Added 5 regression tests to `tests/test_stock_floor_state_machine.py`: valid slice and Hit, invalid initial floors, omitted in-slice floor, full-chart equivalence, Midspin bridge.
- Run: `python -m pytest -q tests/test_stock_floor_state_machine.py tests/test_stock_frame_state_machine.py tests/test_stock_runtime_parity_v01.py`.
- **Not verified:** repository pytest, live ADOFAI runtime, training, GPU, performance/memory checks not executed. No parity or speed claims. This is an input-boundary correctness change, not a behavior-preserving refactor.
- Issue: https://github.com/kineticnapier/DoMachinesDreamOfDeparture/issues/9. Keep Issue open and PR #23 Draft. Use `Refs #9`, not `Fixes #9`.


## Batch update — Refs #2 (2026-10-29)

- **Partial implementation:** Added `src/dmdod/planning/due_event_window_v06.py` and replaced the repeated full-event `upcoming` scan in `scripts/run_dmdogo_v06_pattern_first.py::run_pattern_arm` with indexed time-window lookup. The selection and deferred-retry rules, exact expiry addition, +1e-9 deadline, and original event order are retained.
- Added `tests/test_due_event_window_v06.py` with expiry/prediction boundary, deferral, invalid timeline and 500-event/250-frame seeded equivalence cases.
- **Executed:** standalone helper regression: `PYTHONPATH=/mnt/data/dmdod_audit/src python -m pytest -q /mnt/data/dmdod_audit/tests/test_due_event_window_v06.py` → **3 passed**. This tested a local copy of the new helper, **not** the repository checkout or full runner.
- **Not executed:** repository pytest/integration for v0.6/v0.7/v0.8, frozen-output comparisons, actual physical Motor, training, live ADOFAI, GPU, memory/performance benchmarks.
- **Unimplemented:** post-run `decisions` × `keydowns` and `judgement_trace` scans; exact numerical/telemetry compatibility and performance gains remain unverified.
- Intended behavior-preserving selection optimization. Because the v0.8 research runner is shared, do not certify frozen v0.8 results until full before/after replay/telemetry comparisons pass. Keep Issue #2 open, PR #23 Draft and `Refs #2` only.

## Batch follow-up — Refs #2 (2026-10-29)

- The previously unimplemented post-run full scans are now replaced in `scripts/run_dmdogo_v06_pattern_first.py::run_pattern_arm` with `PostrunDiagnosticIndex` from `src/dmdod/planning/postrun_diagnostic_index_v06.py`.
- The helper indexes KeyDowns by physical axis and time, and TooEarly trace timestamps by target index. It preserves the original axis order, repeated-axis output, nearest-timestamp tie break by original KeyDown order, strict TooEarly timestamp comparison, and diagnostic-only (not scorer) ownership semantics. A widened search window is rechecked with the original `abs(time_ms - due_ms) <= 35.0` predicate for floating-point boundary equivalence.
- Added `tests/test_postrun_diagnostic_index_v06.py` with three tests covering ties, duplicate axes/ordinals, exact/near boundaries, empty input and 500 randomized old/new comparisons (1,000 KeyDowns, 900 trace rows).
- **Executed:** isolated local helper test, `PYTHONPATH=src python -m pytest -q tests/test_postrun_diagnostic_index_v06.py` in a temporary standalone directory → **3 passed**. This is NOT repository pytest or a full physical runner replay.
- **Not executed:** repository-wide or integrated pytest, all three planning modes, before/after frozen replay and telemetry comparison, physical Motor, training, ADOFAI runtime, GPU, memory/performance measurements.
- Classification: intended behavior-preserving diagnostic performance refactor; actual numerical/telemetry equivalence and speed gains are **not yet established**. Issue #2 remains open and PR #23 remains Draft. Use `Refs #2`, never `Fixes #2` until verified.


## 2026-10-29 — Refs #6 (partial provenance recording)
- Committed `scripts/dmdogo_h0_source_provenance.py` and wired `audit_source_provenance` into new H0.10/H0.11/H0.12 report JSON only. Records loaded local Python SHA-256, Git HEAD, dirty status and status digest; frozen v0.8 identity remains separate.
- Added `tests/test_dmdogo_h0_source_provenance.py` for changed source, Git metadata and report wiring.
- Commits: `ecdacf2`, `ee602e5`, `13a6e9d`, `d518352`, `422812f`.
- Run: `uv run pytest -q tests/test_dmdogo_h0_source_provenance.py tests/test_dmdogo_v09_h010_all_quick_seams.py tests/test_dmdogo_v09_h011_adaptive_release.py tests/test_dmdogo_v09_h012_synthetic_counterexamples.py`.
- **Not verified:** repository pytest, actual report generation, training, ADOFAI live runtime, GPU and performance measurements were not run; checkout is unavailable. H0.13+ are not present on this work branch and remain **unimplemented**; do not copy newer research scripts from the diverged base.
- Metadata-only intended behavior-preserving addition to newly generated reports; no frozen research outputs, timing, Motor, HUD-only observations, checkpoints or algorithms changed. Issue #6 remains open, PR #23 remains Draft; use `Refs #6`, not `Fixes #6`.


## Batch update — Refs #20 (2026-10-29)

- **Refs #20** — Added deterministic, bounded distinct-label enumeration in
  `src/dmdod/planning/judgement_label_orders.py` (64 visible increments,
  4,096 distinct-order expansions). Replaced factorial
  `set(permutations(labels))` in all three HUD-only research estimators:
  `judgement_state_estimator_v01.py`,
  `persistent_judgement_state_estimator_v02.py`, and
  `stock_persistent_judgement_state_estimator_v03.py`.
- v01 now treats hitting `max_hypotheses` as an incomplete search rather
  than claiming `MODEL_IDENTIFIED`; v02/v03 preserve an explicit UNKNOWN
  remainder and return TRUNCATED on oversized visible/tied input order sets.
  v02 preflights factorial permutations of tied KeyDowns.
- Added `tests/test_judgement_label_orders.py` and regression cases to
  `tests/test_judgement_state_estimator_v01.py`,
  `tests/test_persistent_judgement_state_estimator_v02.py`, and
  `tests/test_stock_persistent_judgement_state_estimator_v03.py`.
- **Executed only:** 2 standalone helper pytest cases in a local reconstructed
  copy; repository tests were **not run** (GitHub DNS unavailable in execution
  container). Run:
  `python -m pytest -q tests/test_judgement_label_orders.py tests/test_judgement_state_estimator_v01.py tests/test_persistent_judgement_state_estimator_v02.py tests/test_stock_persistent_judgement_state_estimator_v03.py`.
- **Research behavior change:** oversized searches now conservatively abstain,
  and enumeration order is deterministic; exact hypothesis IDs/order may differ.
  Do not compare newly generated research results with frozen reports without
  recording this change. No training, live ADOFAI, GPU, performance or memory
  measurements were run. Keep Issue #20 open and PR #23 Draft; no `Fixes`.


## 2026-10-29 — Refs #5: teacher capacity diagnostic tail (pending verification)

- `src/dmdod/motor/capacity.py`: replace per-frame full unresolved-tail tuple with a bisected due window; keep valid future reservations visible to `CenterFirstNKeyTeacher.pipeline_action()` so the reservation cleanup rule remains unchanged.
- `tests/test_n_key_capacity.py`: add three regression tests for future/stale reservations, stepwise old/new teacher and Motor equivalence, and end-to-end capacity-result comparison against the original full-tail behavior.
- Commits: `0948368f243c1cd69e14f8810c9bdf6f5a3621df`, `acc2ff2ab66829f4f32c2b8b2b542d22a8a9b919`.
- An **isolated** 5,000-case randomized comparison of due-token membership and future-reservation membership passed; this is not repository pytest.
- Run: `python -m pytest -q tests/test_n_key_capacity.py tests/test_real_chart_resolution_cursor.py`.
- **Unverified:** repository pytest, real N-Key capacity run, ADOFAI runtime parity, GPU, training, memory and performance benchmarks. Do not use `Fixes #5` or close the Issue yet. PR #23 remains Draft.


## 2026-10-29 — Refs #24: HUD counter reset clock

- Issue: https://github.com/kineticnapier/DoMachinesDreamOfDeparture/issues/24
- Source: `src/dmdod/features/stock_visible_judgement_feedback_v01.py` — `VisibleCounterRenderer.reset_visible_counters()` clears `_last_capture_s` as well as counters and pending notifications.
- Regression: `tests/test_stock_visible_judgement_feedback_v01.py::test_visible_reset_restarts_capture_clock_and_discards_old_notifications` covers previous capture at 10s, new episode capture at 0s, cleared notifications and within-episode backward-time rejection.
- Source commit `d52c07a9a97b2a7b1e2c8c53c2933bc32f82bff8`; test commit `b4d2a89564b3cfe1846268e45fdd3a12a9ea14b5`.
- Verification command: `python -m pytest -q tests/test_stock_visible_judgement_feedback_v01.py`.
- **Not executed:** repository pytest, live ADOFAI, training, performance or memory measurement. This is a reset-boundary correctness change, not a change to judgement timing or the HUD-only input contract. Issue remains open; PR remains Draft; use `Refs #24`, not `Fixes #24`.


## Batch update — Refs #12: floor-boundary FailBar integration (2026-10-29)

- **Partial correctness fix:** `src/dmdod/envs/real_chart.py` now integrates
  evaluator beat counts across compiled floor-entry times instead of using the
  BPM of one midpoint floor for the entire control interval.
- This uses chart time, so pitch conversion is applied exactly once.
  Equal-time Midspin floors are skipped by `bisect_right`; a single aggregate
  `OverloadCounter.advance_beats()` call retains the previous counter update
  cadence. Existing per-floor effective BPM remains unchanged.
- `tests/test_real_chart_env.py`: 5 new parameterized/individual test cases
  for 120→240 BPM floor transition (pitch 50/100/200%), one-step versus split
  updates, constant BPM across equal-time Midspin floor entries, and the
  six-beat Multipress reset boundary.
- Source commit: `1b96c2fce74382e9956a5e06bcf8e02fe18bbad1`;
  tests: `3f99f4abe58ed884c6781b64b6212b815630f301`, additional six-beat test `d3cdc88f0d5e167947e9a2bce3834a901ea2eac4`.
- Run: `python -m pytest -q tests/test_real_chart_env.py tests/test_n_key_capacity.py`.
- **Unimplemented:** within-floor `SetSpeed(angleOffset>0)` still uses the
  compiler's averaged floor BPM; preserving exact sub-floor speed requires
  new evaluator-only timing segments and Stock timing verification.
- **Not verified:** repository pytest, full-chart before/after replay,
  actual ADOFAI DLL FailBar behavior, training, GPU and performance/memory
  measurements were not run. The partial fix can change FailBar/reward near
  a floor BPM boundary; it is not a behavior-preserving refactor.
- Keep Issue #12 open, PR #23 Draft; use `Refs #12` only, not `Fixes #12`.


## Batch update — Refs #25 (2026-10-29)

- **Refs #25** — Change `VisibleCounterRenderer` pending visible judgement notifications from append+full sort / full scan per capture to a stable min-heap keyed by (event time, insertion sequence). Preserve confirmed-visible categories, display/capture latencies, 1e-12 cutoff tolerance, reset and monotonic capture clock, and HUD-only output.
- Source: `src/dmdod/features/stock_visible_judgement_feedback_v01.py` (commit `230d74facd5364b36b7a5f3d23a593327e2abb5a`).
- Tests: `tests/test_stock_visible_judgement_feedback_v01.py` (commit `c55cdcd7857cfd3a796bff23b353f33aa5ad7efe`), 3 additional cases for out-of-order/same-time delayed notifications, strict capture boundary, 2,500 pending events and reset.
- Run: `python -m pytest -q tests/test_stock_visible_judgement_feedback_v01.py tests/test_judgement_observer_v01.py`.
- **Not verified:** repository pytest (container DNS cannot resolve github.com), training, live ADOFAI runtime, GPU, and performance/memory measurements. Independent old-list vs new-heap comparison: 2 standalone pytest cases passed (200 randomized sequences × 500 operations and boundary case); not a repository test.
- Classification: intended behavior-preserving internal performance refactor. No judgement algorithm, HUD-only surface, Motor, checkpoint or frozen research result changes. Keep Issue #25 open, PR #23 Draft, and do not use `Fixes #25` pending full verification.
