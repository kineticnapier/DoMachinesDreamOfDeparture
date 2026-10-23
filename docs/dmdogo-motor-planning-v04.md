# DMDOGO v0.4 — Motor compensation and multi-event planning

## Executive summary

The fixed bool P18 experiment completed all six independent Motor episodes.
Every arm produced 1,122/1,122 Hits, zero Miss, zero TooEarly and no overload.

State-aware Motor command timing is successful in this deterministic simulator:
Latency-only reaches 100% X Accuracy and reduces MAE from Free's 16.699 ms to
4.861 ms. Integrated v0.4 also reaches 100% X and 4.885 ms MAE. The bounded
Planner does not improve rolling. Planning-only increases left/right switching
to 553 and obtains only 4/381 fixed-window stability. No post-full-chart weight
or horizon adjustment was made.

The result is Pareto, not a single winner. Latency-only is the timing endpoint;
continuity v0.2 remains the naturalness endpoint. Integrated v0.4 improves
Free's thumb use and switching slightly, but does not approach v0.2 rolling.

## Motor latency model

The Phase A audit is documented in `docs/dmdogo-motor-latency-v04.md`. The
frozen Free replay has mean command-to-KeyDown latency 55.755 ms versus mean
scheduled lead 39.085 ms. Response rises to 59.940 ms at 1920 travel BPM and
66.453 ms for selected-axis reuse within 125 ms.

`MotorLeadCompensator` clones only the current 12-axis Motor state and predicts
the fastest isolated KeyDown under the frozen equations. Other axes are neutral
or released, making the prediction optimistic. The live command is latched only
when this earliest arrival is no earlier than the Oracle due. Unknown/unsafe
topology falls back to the unchanged 44 ms rule.

On bool, all 1,122 starts had valid predictions; fallback was not needed.
Prediction MAE was 0.445 ms for Latency-only and 0.452 ms for Integrated. The
actual compensated errors were:

- Latency-only: −2.772 to +12.170 ms, mean +4.828 ms;
- Integrated: −1.958 to +11.211 ms, mean +4.849 ms.

There were 24 and 34 negative timing errors respectively, but none crossed the
scorer's TooEarly boundary. The remaining positive bias is principally the
10 ms control decision grid: the controller waits until the fastest predicted
arrival crosses due, then can overshoot by one control period.

The latency arm plans up to 90 ms early, so its Network inputs and allocations
are not identical to baseline Free: 981/1,122 allocations differ. A posthoc
attempt to replay the exact Free allocations becomes live-ineligible at Event
109 because compensation has already changed Motor state. Therefore the strong
timing gain is established for the causal Latency-only policy, but the narrower
claim “identical fingering plus compensation” remains unproven.

## Short-horizon planner

The Planner is a deterministic candidate graph search with:

- four-event horizon;
- Network top three live-admissible candidates per event;
- beam width eight;
- 80 ms structural same-axis reuse floor;
- lexicographic rolling-break, Network-rank, thumb-use and hand-switch cost;
- live eligibility recheck at commit;
- cached future choices, with explicit discard/replan counters.

It never reads future physical states, KeyDowns, judgements or Teacher rollout.
Future eligibility is provisional and cannot authorize a commit.

The implementation stayed bounded (peak beam 8), but it failed the intended
naturalness objective. Planning-only evaluated 19,419 candidates, discarded
268 cached choices and replanned 76. Integrated evaluated 22,029, discarded
372 and replanned 166. The Network top-three restriction plus changing live
eligibility does not consistently contain a four-finger rolling path. This is
evidence against Hypothesis B for this fixed planner, not evidence that all
multi-event planning is ineffective.

## Full-chart comparison

| policy | Hit | X | MAE | signed | L/R | thumb | fixed fast windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Free | 1122 | 94.060% | 16.699 ms | +16.670 ms | 526 | 228 | 8/381 |
| continuity v0.2 | 1122 | 91.671% | 17.256 ms | +17.227 ms | 405 | 64 | 35/381 |
| continuity v0.3 | 1122 | 93.324% | 16.739 ms | +16.699 ms | 413 | 161 | 4/381 |
| Latency-only | 1122 | 100.000% | 4.861 ms | +4.828 ms | 527 | 209 | 2/381 |
| Planning-only | 1122 | 93.102% | 17.243 ms | +17.222 ms | 553 | 205 | 4/381 |
| Integrated v0.4 | 1122 | 100.000% | 4.885 ms | +4.849 ms | 517 | 197 | 3/381 |

All side-use counts are zero. Timing goals pass for the two compensated arms;
the v0.2-level continuity goals do not. Planning-only is dominated by other
arms and is not recommended.

## Computation

The final six-arm run took 111.178 s. Mean per-event selection time was 0.874
ms for Latency-only and 3.518/3.452 ms for Planning-only/Integrated. Mean
compensation-controller time was 2.928 ms and 2.745 ms. Peak observations were
12.237–13.056 ms for the controller and 149.983 ms for one Planning-only
selection (Integrated peak 7.895 ms). Mean load fits a 10 ms control interval,
but maxima do not establish hard real-time safety. The Python research runner
is therefore not certified as a production real-time controller.

## Viewer and artifacts

`artifacts/dmdogo-v04/bool-full-r2/replay-v04.html` remains a self-contained
Viewer v0.3 with six-policy switching, 16 contacts, KPS, KeyDown flash, planets,
seek and 12-axis state. v0.4 decision cards additionally show planned choices
and whether compensation is active. Existing replay fields remain optional and
older v0.3 payloads remain supported.

## Verification

- Focused v0.4 plus fingering/replay regression: 60 passed in 11.20 s.
- Full repository regression: 2,140 passed in 202.74 s.
- One pre-existing warning remains in `tests/test_onset_ablation.py`: PyTorch
  warns about converting a `requires_grad` tensor to a scalar. No new warning
  was observed.
- The chart, checkpoint and frozen Motor source hashes are recorded in
  `reports/dmdogo-v04-bool-r1.json`.
- The final six-arm evidence was generated once after the implementation was
  fixed. No full-chart weight, beam-width or horizon search was performed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\run_dmdogo_v04_comparison.py `
  --chart "data\DMDOD-v140-tuf-p11-p20\Train\04042 - bool - vo[ICE].adofai" `
  --checkpoint checkpoints\fingering\fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts\dmdogo-v04\bool-full-r3 `
  --status-file logs\dmdogo-v04-bool-full-r3.status.json `
  --wall-timeout-s 1200
```

Use a new revision because all writers refuse to overwrite existing evidence.

## Evidence limits and next questions

1. Separate early fingering planning from Motor compensation without importing
   a future baseline Motor state. A predictive selection-state model is needed.
2. Replace top-three beam admissibility with a safety-constrained rolling FSM
   or Pareto frontier; do not tune weights on bool.
3. Reduce the residual control-grid bias with a sub-control scheduling layer,
   while preserving the frozen 1 ms physics and TooEarly safety.
4. Bound controller wall time before any real-time use; mean runtime is not a
   deadline guarantee.
5. Validate on a predeclared Train-only density fixture. No Validation20 or
   FINAL data was opened in this phase.
