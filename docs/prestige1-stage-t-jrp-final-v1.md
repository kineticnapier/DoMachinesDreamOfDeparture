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

## Counterexample investigation

The two false forecasts had confidence about 0.999 but missed by exactly
15.625ms and 31.250ms.  An initial hypothesis blamed the diagnostic JRP hook's
pre-score target capture.  The hook was correctly hardened to diff the scorer's
used flags and render the single target actually credited.  A synthetic
Miss-then-Hit regression now protects that invariant; all 56 related tests pass.

However, rerun `reports/prestige1-jrp-final-v1-314784d-r5.json` produced the
**identical FAIL**, disproving that hypothesis.  Post-hoc timeline tracing found
the actual cause: the one-frame early/late association rule is ambiguous at
high event rates.  A Hit updates JRP after a control-frame observation.  At the
next observation, that prior-Hit text can coincide with the next floor's HUD
transition and baseline elapsed event.  The tracker then applies the previous
floor's interval to the next event origin:

- Final chart 2: elapsed 47.324ms + stale 15.625ms interval = 62.949ms,
  while no target exists there (nearest target is 15.625ms away).
- Final chart 6: elapsed 1734.375ms + stale 31.250ms interval = 1765.625ms,
  while the nearest target is 31.250ms away.

The confidence score reflects only printed-CBPM quantization uncertainty.  It
does not measure temporal association ambiguity, so it was badly
overconfident on both counterexamples.  Removing or changing the adjacency
rule after seeing Final would be holdout tuning and is therefore not adopted.

The corrected rerun (PID 35380, 11:14:17–11:15:36 JST, exit 1) is a transparent
evaluator-confirmation run, not a second untouched Final selection opportunity.
It must be reported alongside the original FAIL.

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
