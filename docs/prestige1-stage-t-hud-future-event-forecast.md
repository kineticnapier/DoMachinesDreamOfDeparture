# Stage T — HUD-only future 10ms event-sequence forecast (read-only)

Status: **deterministic research diagnostic, not an adopted Student planner**.
Stage A FAIL, frozen Geometry/Hybrid, original MaleCNS4096 and all
checkpoints remain unchanged. No Student training, Validation or FINAL.

## Question

Can the currently observable **263D HUD** predict **how many playable
floor-entry events occur in the next 10ms, in what relative order, and
at what offsets** — especially when a 20,000 BPM chart crosses three or
four floors during one frozen 10ms Student control step?

## Causal baseline

- Input: a **single 263D HUD** frame, with v2 unclipped Tile BPM,
  normalized orbit sin/cos, and ordinary current/ahead floor geometry.
  No absolute chart index, privileged time, target schedule, hidden
  direction, simulator state or future observation is supplied.
- For both CW and CCW hypotheses, derive the current orbit's phase from
  the signed entry-to-orbit angle, and the target arc lengths from each
  floor's visible entry/exit angles. Use `pi * TileBPM / 60` as angular
  rate. Emit future event offsets with `0 < delta_t <= 0.010s`.
- A direction is ruled out **only if incompatible with the current
  observed arc**. Failure of the opposite direction on a future floor
  does **not** prove this direction. If both remain possible, accept
  only identical event schedules; otherwise abstain.
- Deliberately abstain for full-turn/zero ambiguous arcs, near-boundary
  phase, insufficient lookahead, MultiPlanet/Midspin, speed changes,
  pauses and other unsupported visible special events. A Twirl already
  applied on the current floor can be handled if current phase resolves
  the resulting direction; future Twirls abstain.
- This is a *conditional coverage* predictor: a zero-error eligible
  subset is not a claim of complete chart or end-to-end control.

## Deterministic verification

- Use real compiled ADOFAI chart fixtures and **8-key 1ms physics/HUD
  simulator**, not synthetic sin/cos. Independently compare each 10ms
  decimation with a second simulator driven at 10ms control/1ms physics.
- Exactly the original 26 fixed fixtures: 13 BPM points (600 through
  20,000), CW and CCW.
- Probe fixed 10ms frame offsets of 0, 1, 2, 5 milliseconds from the
  1ms-rendered trace. Shifted decimations are *diagnostic only*; the
  original Student's clock remains 10ms.
- All forecasts are computed from HUD before consulting any evaluator
  timestamps. The posthoc scorer checks the predicted count and
  ordered time offsets against the next actual target timestamps.
- Separately test curved-path ordinary geometry, CW/CCW ambiguities,
  unsupported visible future events, truncated lookahead and corrupt
  inputs. An earlier error at the initial ambiguous phase demonstrated
  why *an unsupported future path must not eliminate a direction*.
- Store eligible/abstained, correct/wrong event sequences, multi-event
  interval coverage and conditional timing errors independently.

## What this does not solve

1. No planned finger assignment and no intra-10ms multi-key Motor command.
2. No promise of recovery for floor-boundary ties, future SetSpeed,
   Pause, Midspin, MultiPlanet, hidden acceleration, or incomplete HUD.
3. The 1ms measurements are a diagnostic observation source, not a
   change to Student inputs or policy inference rate.
4. No training on Train20/Train-heldout/Validation/FINAL.
5. Passing one family of straight/ordinary fixtures does not establish
   20,000 BPM end-to-end playability or generalization to unknown maps.

## Commands

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_future_events_audit.py
uv run python -u scripts/audit_prestige1_autoplay_future_events.py --config configs/training/prestige1.toml
~~~

Output is immutable on re-execution:

    diagnostics/prestige1/stage_timing_future_events_10ms_simulator_readonly.json

Next research question: whether this predicted **relative event schedule**
can be represented/learned by the fixed Student and executed by physical
8-key actuators without privileged control or manual oracle timing.
