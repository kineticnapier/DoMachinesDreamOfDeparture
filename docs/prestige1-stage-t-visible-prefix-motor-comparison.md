# Stage T — Visible-prefix forecasting for the persistent 8-Key Motor

Status: **isolated deterministic diagnostic**; does **not** modify the
frozen 70ms baseline, Stage A FAIL, original Geometry/Hybrid,
MaleCNS4096, checkpoints, or any training procedure. No Validation/FINAL.

## Question

The existing `forecast_visible(hud, horizon=70ms)` requires enough
ordinary, visible floor geometry to establish the **entire** 70ms window.
Near the chart end it returns `insufficient_lookahead`, discarding
otherwise determinable targets within that window. Can we safely preserve
those targets *without* assuming that unseen future floors contain no notes?

## Separate, conservative forecaster

`forecast_visible_prefix()` inspects the same **one 263D HUD frame**:
normalized orbit, floor entry/exit angles and unclipped Tile BPM. For each
CW/CCW candidate it walks visible ordinary floors, accumulating floor-entry
events in order until the 70ms horizon, a missing floor or unsupported
geometry/event is encountered.

- If the whole 70ms horizon is observed, return `eligible_complete`.
- If the horizon is **not** completely visible but there is a safe known
  event sequence, return `visible_prefix` with a stop reason. This does
  **not** claim any knowledge of later events.
- If the current direction is geometrically impossible, eliminate it.
  If **both** directions are still viable, accept only when their *entire
  supported predicted event lists* agree within the existing 100ns
  comparison tolerance; otherwise abstain.
- Do not infer direction from a future Twirl, SetSpeed, Pause, Midspin,
  MultiPlanet, full-turn ambiguity or missing geometry. A safe prefix
  preceding an unmodelled event may be published, but not the event itself.
- If zero future events are safely visible and the preview is incomplete,
  abstain. Never convert an incomplete prefix to "no upcoming events."

The new `PrefixPersistentMotorPlanner` duplicates **only** the unchanged
baseline assignment, persistent intent, cold-latency and physical action
logic, substituting the new forecaster. Motor still controls every
**10ms**, integrates physics at **1ms**, and resets **once** at chart start.
This is a diagnostic A/B comparison, not a Student policy upgrade.

Neither the forecaster nor the planner sees chart truth, absolute floor
index, evaluator target timestamps, exact elapsed chart time, hidden
rotation direction or evaluator Hits. Those are accessed **only after**
the physical run to compare outcomes.

## Fixed A/B experiment (Linux CPU simulator)

All nine combinations of 18/72/144 straight-floor targets and
1200/2000/4000 Tile BPM, same unchanged key physics and fixed 70ms preview:

| BPM | Targets | Full 70ms baseline Hit | Safe visible prefix Hit | Delta |
|---:|---:|---:|---:|---:|
| 1,200 | 18 | 18 | 18 | 0 |
| 2,000 | 18 | 18 | 18 | 0 |
| 4,000 | 18 | 15 | **17** | +2 |
| 1,200 | 72 | 72 | 72 | 0 |
| 2,000 | 72 | 72 | 72 | 0 |
| 4,000 | 72 | 61 | **65** | +4 |
| 1,200 | 144 | 144 | 144 | 0 |
| 2,000 | 144 | 144 | 144 | 0 |
| 4,000 | 144 | 124 | **127** | +3 |

At 4000 BPM, the visible-prefix forecaster generates 2/4/4 additional
unique commitments on 18/72/144-note charts. The 18-note run still misses
its final target. The longer-chart mid-sequence physical timing/target
mismatch remains unsolved. This is **not** a claim that the new forecaster
solves 4,000 BPM or that it should be adopted as a Student component.

One-frame prediction tests independently compare **every claimed time**
against actual simulator events on ordinary 1,200/2,000/4,000/20,000
BPM charts, and ensure the reported status differentiates partial from
complete. Additional tests require safe abstention for ambiguous direction,
future Twirl and unsupported events; a future Pause can be preceded by
known safe events, but the pause itself is not predicted.

## Commands

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_visible_prefix_audit.py
uv run python -u scripts/audit_prestige1_autoplay_visible_prefix.py --config configs/training/prestige1.toml
~~~

Immutable output (refuses replacement):

    diagnostics/prestige1/stage_timing_visible_prefix_motor_comparison_readonly.json

## Next question (not implemented)

Diagnose **why the remaining 4,000 BPM misses occur even with an event
reservation**. Separate late first command, hand coupling, finger re-use,
physical KeyDown error and evaluator target mismatch, using the previously
established endurance attribution approach. Do not tune the motor while
measuring and do not use hidden simulator truth to schedule actions.
