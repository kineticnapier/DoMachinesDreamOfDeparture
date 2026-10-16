# Stage T — HUD event forecasts to physical eight-key motor (shadow diagnostic)

Status: **read-only feasibility baseline, not a deployed scheduler and not Student training**.

## Motivation

The fixed Stage T HUD-only predictor can identify three events at relative
+2ms/+5ms/+8ms on the ordinary 20,000 BPM straight-floor chart. That is **not**
proof that the 8-Key Motor can press the keys at those times. The original
Student continues to produce one action **every 10ms**, held for 10ms;
physical integration runs at **1ms**.

The motor's calibration from a **resting finger** (unchanged motor physics,
1ms stepping) is approximately **39–42ms until the first KeyDown**. This is
longer than the 10ms forecaster horizon. For normal 180-degree floors at
20,000 BPM, the visible 12-floor geometry extends only ~36ms ahead. Even
with longer forecasting, repeated presses also require release/recovery.

These are physical preparation issues, distinct from correctly reading BPM
or planning the next event. They do *not* prove that an optimized predictive
controller could never run this BPM.

## What the probe actually does

1. Measure all eight resting-key latencies with the existing frozen
   `calibrate_n_key_press_lead` motor simulator, never chart target timestamps.
2. Run the unchanged Stage T single-HUD forecast on **each actual 10ms HUD**
   produced by the deterministic 8-Key simulator in 26 fixed real-chart
   fixtures (13 BPM points, CW/CCW).
3. If the forecaster abstains, produce **no motor plan**. Otherwise allocate
   an unused, unpressed finger per predicted event, ordered by measured
   single-finger cold latency. This is **not** a globally optimized fingering.
4. For each event, set the nominal press command onset to
   `max(0, predicted_event_time - cold_latency)`. Record the positive
   deadline deficit if the expected actuation lead is unavailable.
5. For each fixture, take the first eligible nonempty plan and simulate a
   **fresh-from-rest** unchanged physical N-Key motor, with reference actions
   updated every 1ms for up to 80ms. The reference executor holds a finger
   command until actual physical KeyDown, then brakes it.
6. Record true motor KeyDown event times and the number that occurred inside
   the **first 10ms**. Multiple simultaneous finger commands can compete via
   bilateral hand dynamics: isolated-finger latency is *not* guaranteed to
   predict simultaneous-key events.

The predictor/planner never receives exact floor indices, chart time, target
oracle timestamps or evaluator labels. Motor calibration is a public model
property independent of the chart; the evaluator does not choose actions.
The physical reference starts from rest and is *not* a continuation of the
actual Student or trained finger positions.

## Interpretation limits

- Counts of planned events in the report are **per-observation plan entries**,
  not unique target IDs over the whole chart. The same target can appear in
  several overlapping 10ms planning snapshots.
- The 1ms executor is **only an investigative counterfactual**. Its use does
  not modify or grant 1ms action access to the existing Student.
- The naive multi-key executor is not an optimal motion plan. Failure at
  20,000 BPM from rest is not a proof of absolute physical impossibility.
- The original HUD predictor remains conditional (ordinary floor geometry,
  unsupported special events result in abstention).
- A proper learned/prepared controller will need to plan farther ahead,
  maintain persistent per-finger motor intentions across observations, and
  account for simultaneous-key mechanics and release/recovery delays.
- Stage A FAIL, original Geometry/Hybrid, fixed MaleCNS4096, all checkpoints,
  Validation, FINAL and existing learning pipelines are untouched.

## Commands

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_motor_schedule_audit.py
uv run python -u scripts/audit_prestige1_autoplay_motor_schedule.py --config configs/training/prestige1.toml
~~~

New immutable report:

    diagnostics/prestige1/stage_timing_forecast_motor_shadow_20k_readonly.json

The report refuses replacement if the file already exists.

## Recommended next investigation

Before changing a Student, test a **persistent 10ms observation / 1ms
motor intent buffer** with sufficiently early, HUD-visible predictions and
full per-key dynamics. Evaluate on deliberately feasible chart speeds as
well as the extreme 20,000 BPM case, and avoid oracle timestamps in the
planner. If a target cannot be seen early enough to prepare a new finger,
record it as infeasible under that information contract rather than using
hidden simulator truth.
