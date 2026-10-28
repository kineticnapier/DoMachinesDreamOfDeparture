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
