# DMDOGO v0.5 — Continuity × Motor compensation

## Result

The predeclared `continuity_compensated` arm causally composes the unchanged
Continuity v0.2 selector with the v0.4 Motor forward-model compensator. On the
fixed Train bool chart it completed 1,122/1,122 Hits with no Miss, TooEarly or
overload. X Accuracy was 100%, MAE was 4.638 ms and signed error was +4.557 ms.

The arm also improved aggregate continuity relative to v0.2: left/right
switches fell from 405 to 359 and thumb use from 64 to 39. It did **not**
preserve the fixed-window rolling result: only 4/381 windows were stable versus
35/381 for v0.2. v0.5 is therefore a strong Pareto candidate for timing and
coarse continuity, but it does not meet the declared rolling target.

## Causal integration contract

- The frozen v0.1 Network produces scores from the current Motor state.
- `ContinuitySelector(mode="continuity")` sees only scores, live eligibility,
  due time and its committed past selections.
- The selected allocation is checked against live eligibility before commit.
- `MotorLeadCompensator` then schedules the command from a clone of the current
  Motor state. It cannot alter the Oracle due time or the selected allocation.
- No future KeyDown, judgement, physical observation or Teacher rollout is
  available to either component.
- Disabling compensation dispatches through the original v0.2 runner; a TDD
  fixture verifies identical selections and KeyDowns.

The existing Free, v0.2, v0.3, Latency-only and Planner-v0.4 call paths retain
their defaults. The new arm is separate from the v0.4 Planner.

## Why the selection history changes

Compensation expands the selection horizon from 44 to 90 ms. Compared with the
independent v0.2 episode, v0.5 selected 40–50 ms earlier (mean 45.927 ms) and
changed 967/1,122 allocations. The Network Free argmin itself changed on 976
events because the observed Motor state and lookahead-relative context differ.

Of the 967 changed allocations, the original v0.2 allocation was no longer
live-eligible at the v0.5 selection point for 405 events. It was live-eligible
but not selected for 562; those cases combine changed Network ranks with the
already-diverged causal continuity history. Only 66 of 577 events following a
gap of at most 35 ms preserved the exact v0.2 allocation.

These are coupled independent episodes, so no individual changed event is
claimed as an isolated cause. The evidence does explain why “the v0.2 finger
sequence, just earlier” is not a valid implementation.

## Motor result

All 1,122 compensation predictions were valid and all completed; fallback was
never used. Command lead averaged 52.481 ms. Actual command-to-KeyDown latency
averaged 57.037 ms, while the isolated prediction averaged 57.130 ms.
Prediction error MAE was 0.506 ms and signed error was −0.093 ms.

At rounded travel BPM 1,920, v0.5 covered 384 events with mean timing error
+4.404 ms (range −2.392 to +10.863 ms). At 480 BPM it covered 737 events with
mean +4.639 ms (range −1.981 to +11.876 ms). There were no misses, early
judgements, incomplete commands or reservation failures in either region.

The controller mean was 2.810 ms per 10 ms frame, but one observed maximum was
161.304 ms. The Python research runner therefore does not establish a hard
real-time bound even though simulated episode time and results remain valid.

## Fixed full-chart comparison

| policy | Hit | X | MAE | signed | L/R | thumb | fixed fast windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Continuity v0.2 | 1122 | 91.671% | 17.256 ms | +17.227 ms | 405 | 64 | 35/381 |
| Latency-only | 1122 | 100.000% | 4.861 ms | +4.828 ms | 527 | 209 | 2/381 |
| Integrated v0.4 | 1122 | 100.000% | 4.885 ms | +4.849 ms | 517 | 197 | 3/381 |
| Continuity-compensated v0.5 | 1122 | 100.000% | 4.638 ms | +4.557 ms | 359 | 39 | 4/381 |

All arms had zero Miss, zero TooEarly, no overload and 1,122 physical KeyDowns.
The three baseline rows reproduce the saved v0.4 values.

## Verification

- Focused v0.5/continuity/compensation/viewer regression: 55 passed in 11.87 s.
- Full repository regression: 2,145 passed in 194.45 s.
- The sole warning is the pre-existing PyTorch tensor-to-scalar warning in
  `tests/test_onset_ablation.py`; no new warning was observed.
- The comparison JSON, telemetry and HTML hashes are fixed in
  `reports/dmdogo-v05-bool-r1.json`.
- The full chart was evaluated once after the design was fixed; no result-led
  selector or compensation tuning was performed.

## Artifacts and reproduction

- Comparison: `artifacts/dmdogo-v05/bool-full-r1/comparison-v05.json`
- Detailed telemetry: `artifacts/dmdogo-v05/bool-full-r1/telemetry-v05.json`
- Self-contained Viewer v0.3: `artifacts/dmdogo-v05/bool-full-r1/replay-v05.html`
- Cause audit: `reports/dmdogo-v05-continuity-compensation-cause-r1.json`

```powershell
.\.venv\Scripts\python.exe scripts\run_dmdogo_v05_comparison.py `
  --chart "data\DMDOD-v140-tuf-p11-p20\Train\04042 - bool - vo[ICE].adofai" `
  --checkpoint checkpoints\fingering\fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts\dmdogo-v05\bool-full-r2 `
  --status-file logs\dmdogo-v05-bool-full-r2.status.json `
  --wall-timeout-s 1200
```

Use a new revision: the runner refuses to overwrite existing evidence.

## Conclusion and next design boundary

The two mechanisms coexist safely and produce the best measured timing plus
coarse continuity on bool. Exact rolling does not survive the earlier causal
selection state. A future design should predict selection-time Motor
availability or preserve a rolling state under an explicit safety constraint;
it should not retune v0.2 weights on this evaluation chart. No v0.6 mechanism
was implemented in this phase.
