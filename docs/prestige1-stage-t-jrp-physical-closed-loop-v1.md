# Stage T — JRP physical closed-loop visibility audit v1

This is **not** trained Student evaluation. No trained Student `.pt` file
is mounted in the assistant CPU environment. This commit deliberately avoids
inventing such weights or reporting a Student achievement. It provides a
causally separated policy callback and real CPU Motor/JRP display loop, and
runs a fixed scripted Motor fixture only to measure HUD timing safety.

## Design

`src/dmdod/evaluation/jrp_closed_loop_v1.py` wraps the existing
`JrpOverlayDiagnosticNKeyEnv` (8-key, 10ms control, 1ms physics), the frozen
`CausalTimingTracker` v1.4, and the conservative `CausalJrpTimingBridge`.

A policy receives **only** an immutable `CausalJrpPolicyInput`:

- sequential HUD frame number;
- current 263D HUD tuple;
- the three *displayed* JRP strings or None (possibly delayed/stale);
- derived baseline/JRP pending Timing updates.

The policy receives **no** target times, chart floor identifiers, oracle
renderer diagnostics, true current time, reward/hit counters, or chart object.
The evaluator checks each returned physical 8-key action and advances the
unchanged Motor/scorer. Target timestamps are read only in posthoc matching.

The embedded `ScheduledMotorFixture` is not a Student: its pulse windows
were chosen from nominal chart timing before rollout. `IdlePolicy` is the
all-miss control. The harness is ready for an actual `CausalJrpPolicy`
implementation, once an existing compatible Student checkpoint is available.
It does not claim to convert frozen Stage A GRU timing estimates into a Motor
controller or change the Stage A gate.

## Fixed CPU test

Synthetic 120 BPM seven-target chart, Pause 2 beats at first floor.
Preselected fixture pulses: 0.30–0.70s, 2.24–2.64s, 2.80–3.20s,
3.55–3.95s, and 4.55–4.95s. These are *scripted actions*, not
HUD-only decision outputs. Score/HUD/predictor all advance closed loop with
actual 1ms physical stepping. Visible JRP text is independently delayed by
0, 10 or 30ms; the game's Hit hooks themselves are not rescheduled.

| Scenario | Physical Hit / Miss | JRP Hit hooks | Unique eligible CBPM candidates | Correct / false candidates | NEW additional Timing events |
|---|---:|---:|---:|---:|---:|
| Scripted, text delay 0ms | 5 / 2 | 5 | 2 | 2 / 0 | 0 |
| Same actions, delay 10ms | 5 / 2 | 5 | 2 | 2 / 0 | 0 |
| Same actions, delay 30ms | 5 / 2 | 5 | 0 | 0 / 0 | 0 |
| Skip first scripted press | 4 / 3 | 4 | 1 | 1 / 0 | 0 |
| No presses at all | 0 / 7 | 0 | 0 | 0 / 0 | 0 |

**Important**: the 2 correct CBPM candidates are already predicted by
baseline Timing v1.4. They are **not 2 new hits** or 2 extra covered
notes. The causal guard abstains when text is 30ms late or not refreshed
by a missing Hit. These results support the visibility/safety contract
for this one fixture, not general native-JRP parity or learned-Student
performance. Delay values are three fixed diagnostic cases, not a tuned
hyperparameter study.

51/51 related CPU regression tests PASS, including the new loop tests.

## Reproduction

```sh
PYTHONPATH=src python -m pytest -q \
  tests/test_jrp_closed_loop_v1.py \
  tests/test_jrp_cbpm_bridge_physical.py \
  tests/test_jrp_cbpm_anchor_v1.py \
  tests/test_jrp_overlay_diagnostic.py \
  tests/test_jrp_overlay_v2.py \
  tests/test_jrp_bpm_v1.py

PYTHONPATH=src python scripts/audit_jrp_closed_loop_v1.py \
  --output diagnostics/jrp-physical-closed-loop-v1.json
```

Audit refuses to overwrite. The real DMDOD scorer/physics are run; no
Student training, checkpoint writes, protected holdouts, stage-gate edits, or
old 263D format changes are performed. Frozen Stage A remains **FAIL**.

## Outstanding blocker

Obtain the *actual compatible trained Student checkpoint* and its exact
inference/actuation adapter before claiming Student closed-loop results.
A frozen timing-only GRU should not be falsely represented as an 8-key
Motor policy. Follow-up must compare JRP-hidden vs JRP-visible using the
same weights, initial conditions, and independently held-out chart.
