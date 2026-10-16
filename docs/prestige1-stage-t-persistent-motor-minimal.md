# Stage T — Persistent Motor Planning, minimal 10ms-control experiment

Status: **isolated, read-only deterministic baseline**; not an adopted policy.
No Stage A/Geometry/Hybrid/4096-CNS/Student/checkpoint changes, no training,
no Validation or FINAL.

## Research question

What happens when an 8-Key Motor continues operating across multiple notes,
rather than being reset before every proposed event?

### Simulator and policy input

- Genuine `DiagnosticHudNKeyRealChartMotorEnv`: 10ms held actions, 1ms
  physics substeps. One reset at the start, **no reset between notes**.
- Fixed deterministic 18-target ordinary straight-floor charts at 600, 900,
  1200, 2000, 4000 and 20000 Tile BPM. No SetSpeed, Pause, Twirl, Midspin
  or MultiPlanet; this is *not* a general chart/flight benchmark.
- Planner input is **only the current 263D v2 HUD, persistent reservations,
  local 10ms policy tick count, and frozen 8-Key per-finger cold-press
  calibration**. No privileged chart time, target times, absolute floor
  ordinal or hidden direction are passed into the planner.
- Extend the existing conservative current-floor/forward-floor geometry
  forecast from 10ms to a fixed **70ms lookahead**. CW/CCW ambiguities and
  unsupported visible geometry still cause abstention. This 70ms is a
  *fixed diagnostic horizon*, not a hyperparameter search.
- Persist predicted event deadlines and finger assignments across control
  steps. Deduplicate predictions via locally clocked predicted time tokens;
  start finger activation when a predicted deadline is within its measured
  cold latency. Release after observing KeyDown through the HUD's motor
  press flag, not evaluator feedback. No microstep command access is
  introduced to the frozen Student.
- Evaluate actual KeyDown and ADOFAI hit/miss only **after** issuing action;
  neither evaluator signal adjusts the planner.

## Interpreting results

This baseline is deliberately simple. It is not an optimized fingering
planner and does not learn; the repeated use of a cold latency ignores
hand coupling, acceleration state and finger fatigue. It may hit despite
latency prediction error because the evaluator has nonzero judgement windows.

The 20,000 BPM 70ms preview is longer than the 12 ordinary ahead floors
(approximately 36ms), so this conservative estimator abstains. A zero hit
result under this fixed horizon is **not** a physical impossibility proof.

Even on slower fixtures, these deterministic 18-target results do not
establish generalization or real chart completeness. Tests report count,
errors, physical actions and evaluator outcome separately.

## Run (Windows/Linux)

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_persistent_motor_audit.py
uv run python -u scripts/audit_prestige1_autoplay_persistent_motor.py --config configs/training/prestige1.toml
~~~

Output, never overwritten automatically:

    diagnostics/prestige1/stage_timing_persistent_motor_minimal_readonly.json

## Next steps, not yet implemented

1. Verify longer-running physical fatigue/release dynamics at a moderate
   stable BPM, with more than 18 events.
2. Separate error sources: forecast abstention vs insufficient early
   visibility vs physical actuation/release vs fingering choice.
3. Investigate data-driven motor compensation and persistent trajectory
   planning **without using privileged target times in the controller**.
4. Keep extreme-BPM motor training isolated until the observation/control
   contract is established, especially when more than 8 notes fall inside
   the available motor preparation window.
