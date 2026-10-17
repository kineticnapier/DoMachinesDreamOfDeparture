# Stage T — Prestige v2 causal JRP Final10 audit v1

## Locked protocol

Commit `983c396` fixed the protocol before opening Final: all ten Final charts,
the deterministic middle eight-second window, frozen `prestige_v2.pt`, CUDA
inference, unchanged 263D Student input, and identical Motor initial state.
Each window ran twice with only JRP visibility changed.  Labels were available
only to the post-hoc matcher.  No Final metric selected a threshold or model.

Run PID 45832 started 2026-10-17 11:09:39 JST and ended 11:10:56 JST with
exit code 1.  The full report is
`reports/prestige1-jrp-final-v1-983c396-r4.json`.

## Honest result: FAIL

| Metric | Timing v1.4, JRP hidden | JRP visible |
|---|---:|---:|
| target events | 862 | 862 |
| matched forecasts | 718 | 719 |
| tracking rate | 83.2947% | 83.4107% |
| false forecasts | 0 | 2 |
| JRP-only forecasts | 0 | 3 (1 correct, 2 false) |
| physical Hit / Miss | 147 / 715 | 147 / 715 |
| Too Early / Overload | 29 / 0 | 29 / 0 |

The extra correct event does not offset two new false detections, so the fixed
success condition failed.  Physical outcomes are identical because this audit
tests Timing observation/prediction while preserving the frozen Motor policy;
JRP does not silently alter the old checkpoint's 263D tensor.

## Counterexample and evaluator defect

The two false forecasts had confidence about 0.999 but missed by exactly
15.625ms and 31.250ms.  Both occurred when a physical keydown expired an older
Miss and then credited a later target in one scoring call.  The diagnostic JRP
hook captured `_next_target_index()` **before** scoring, so it rendered CBPM for
the missed floor rather than the floor whose Hit was actually credited.  This
made a stale interval look like a high-confidence visible update.

The correction diffs the scorer's evaluator-only `used` flags and sends the
single newly credited floor to the renderer.  Those flags remain inside the
renderer/evaluator boundary and never enter Student input.  A new synthetic
Miss-then-Hit regression reproduces the old association error.  All 56 related
CPU tests pass after the correction.

Any rerun after this point is a transparent evaluator-correction confirmation,
not a second untouched Final selection opportunity.  Its result must be
reported alongside this original FAIL.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --basetemp=artifacts/pytest-jrp-miss-credit-v1 `
  tests/test_jrp_overlay_diagnostic.py `
  tests/test_jrp_closed_loop_v1.py `
  tests/test_jrp_cbpm_bridge_physical.py `
  tests/test_jrp_cbpm_anchor_v1.py `
  tests/test_jrp_overlay_v2.py `
  tests/test_jrp_bpm_v1.py

.\.venv\Scripts\python.exe -u scripts/audit_prestige1_jrp_final_v1.py `
  --config configs/training/prestige1.toml `
  --checkpoint checkpoints/prestige1/prestige_v2.pt `
  --device cuda `
  --output reports/prestige1-jrp-final-v1.json `
  --status-file logs/jrp-final-v1/status.json
```

The audit refuses to overwrite report and status paths.  Frozen Stage A remains
FAIL; checkpoints, old 263D input, Train/Validation/Final membership, scorer,
and Motor physics are unchanged.
