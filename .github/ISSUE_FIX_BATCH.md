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

Both are correctness changes, not behavior-preserving refactors.
Do not mark either issue fixed or use `Fixes #...` until regression checks pass.
Frozen research reports, HUD-only constraints, timing constants and checkpoint
formats were not intentionally changed.
