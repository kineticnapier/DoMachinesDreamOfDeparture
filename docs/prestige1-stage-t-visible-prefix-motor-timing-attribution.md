# Stage T — Visible-prefix persistent Motor: intended vs physical timing

Status: **read-only deterministic diagnostic**. No changes to the current
Student, frozen Stage A (FAIL), original Geometry/Hybrid, MaleCNS4096, motor
physics, policy, checkpoints, Validation or FINAL. No action tuning.

## Motivation and method

The previous Stage T visible-prefix planner improved 4,000 BPM 18/72/144-note
results from 15/61/124 Hits to 17/65/127, but did not resolve the remaining
Misses. This experiment runs that **exact unchanged planner** and physical
8-Key simulator, with 10ms held actions and 1ms physics, to distinguish:

- Causally HUD-derived reservation and local-clock due time.
- The first actual positive action issued to the assigned key, its motor
  position, velocity and muscle activation at the time, plus the cold-finger
  calibration-based deficit. These physical states are **diagnostic-only**.
- An actual physical KeyDown from the 1ms simulator and the evaluator target
  that KeyDown was offered to, with its timing margin.
- Per-target Hit/Miss and any offset between the **intended** floor and the
  floor the evaluator was prepared to score.

Planner and its inputs remain unchanged. Only current 263D HUD and previous
planner state are passed into the policy. Exact target timestamps and floor
ordinals are consulted **only after actions have been chosen and executed**.

The scorer pairs each physical KeyDown to the *earliest previously commanded,
unmatched intent for the same key*. This is a diagnostic association, **not
proof of causation** or a globally optimal intent/event matching algorithm.

## Fixed results: 1,200 BPM omitted; focus on 2,000 and 4,000

| BPM | Notes | Hit | Miss | X-Accuracy | Mean absolute intended KeyDown error | Miss association |
|---:|---:|---:|---:|---:|---:|---|
| 2,000 | 72 | 72 | 0 | 88.89% | 21.10ms | — |
| 2,000 | 144 | 144 | 0 | 88.54% | 21.95ms | — |
| 4,000 | 18 | 17 | 1 | 63.06% | 31.35ms | 1 evaluator-target mismatch |
| 4,000 | 72 | 65 | 7 | 42.43% | 38.32ms | 7 evaluator-target mismatches |
| 4,000 | 144 | 127 | 17 | 41.25% | 39.07ms | 16 evaluator-target mismatches, 1 without a reservation |

At 4,000 BPM/144 notes, the first failed intended event is note #20: it is
paired with a KeyDown **38ms after its intended due time**, while the
simulator evaluator was still waiting for note #19. Many later pairings are
one or more targets behind. This supports a **sequence-timing alignment
problem**, beyond the previously measured terminal lookahead issue.

**Important interpretation:** mean intended KeyDown error is calculated from
planner reservations and diagnostic same-key associations, not the simulator's
judgement timing on whichever target happened to be scored. The two quantities
must not be equated. The cold-latency shortfall metric assumes a static finger
from rest and cannot by itself predict a motor already in motion; it should
not be treated as a physical minimum latency or definitive explanation.

## How to run

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_prefix_motor_timing_audit.py
uv run python -u scripts/audit_prestige1_autoplay_prefix_motor_timing.py --config configs/training/prestige1.toml
~~~

Report (refuses overwrite):

    diagnostics/prestige1/stage_timing_visible_prefix_motor_timing_attribution_readonly.json

Five fixed experiments: 2,000 BPM × (72, 144 notes) and 4,000 BPM × (18, 72,
144 notes). Full per-target intent and per-KeyDown timing details are included
in the JSON, as are 10ms command frame logs. No repeated random trials,
hyperparameter sweeps, training, Validation or FINAL.

## Interpretation and next research question

A simple cold-press-latency scheduler is not an accurate multi-key dynamical
controller. We should next investigate whether **prearmed trajectory,
release/retrigger timing and coupled finger dynamics** can reduce the actual
KeyDown vs intended-event error, on an isolated controller with unchanged
physics, before modifying or training the real Student. Do not silently
optimize the existing frozen baselines or select hyperparameters using
Evaluation/FINAL.
