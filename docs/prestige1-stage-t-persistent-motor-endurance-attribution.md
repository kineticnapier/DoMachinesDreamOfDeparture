# Stage T — Persistent Motor endurance and miss attribution

Status: **read-only, fixed deterministic simulator diagnostics**. This does
not change any trained Student, Motor policy, frozen Stage A/Geometry/Hybrid,
checkpoints or MaleCNS4096. No Validation/FINAL.

## Purpose and experimental contract

Investigate the original 18-note 4,000 BPM 3-Miss result and run the same
unchanged persistent 10ms-control/1ms-physics 8-Key Motor on longer charts.
Do **not** fix or optimize the planner inside this experiment.

- Straight ordinary 180-degree-floor fixtures, `bpm = 1200/2000/4000`,
  fixed lengths 18, 72 and 144 targets; nine fixed runs.
- `PersistentMotorPlanner` with the existing **70ms full-horizon** guarded
  forecast, local monotonic control clock, persistent per-key reservations,
  unchanged cold-finger calibration and single 10ms held Action per tick.
- A single physical Motor reset at chart start. Never reset between notes.
- Planner sees **only the current 263D HUD plus its own past state**.
  The oracle's exact note timestamps, indices and evaluator state are read
  strictly for **posthoc** scoring after physical Action execution.
- Independently distinguish (a) no committed note, (b) planned but never
  commanded, (c) commanded but no matched KeyDown, and (d) a physical
  KeyDown whose evaluator target differs from the intended note. Match a
  physical down to the oldest issued, not-yet-matched intent on the same key;
  this is attribution, not an optimal causal explanation.
- Record the actual simulated KeyDown timings, evaluator Hit/Miss,
  X-Accuracy, intended-event timing errors, plus per-floor evidence.

## Results (local Linux CPU simulator, deterministic)

| BPM | Targets | Hit | Miss | X-Accuracy | Intended KeyDown MAE | Miss attribution |
|---:|---:|---:|---:|---:|---:|---|
| 1200 | 18 | 18 | 0 | 95.28% | 13.94ms | — |
| 2000 | 18 | 18 | 0 | 85.00% | 18.33ms | — |
| 4000 | 18 | 15 | 3 | 58.61% | 29.33ms | 3 unplanned |
| 1200 | 72 | 72 | 0 | 98.47% | 16.19ms | — |
| 2000 | 72 | 72 | 0 | 88.89% | 21.10ms | — |
| 4000 | 72 | 61 | 11 | 40.21% | 38.15ms | 4 unplanned; 7 evaluator-target mismatch |
| 1200 | 144 | 144 | 0 | 99.24% | 16.63ms | — |
| 2000 | 144 | 144 | 0 | 88.54% | 21.95ms | — |
| 4000 | 144 | 124 | 20 | 39.27% | 39.59ms | 4 unplanned; 16 evaluator-target mismatch |

### Why the final 3/4 notes were missed

The existing `forecast_visible(..., 70ms)` requires **complete geometry
through the entire 70ms horizon** to return an eligible schedule. Near chart
end that geometry is unavailable, even though the next few real notes are
visible. The baseline therefore discards **all** partial known events and
cannot create commitments for the last 3/4 notes at 4,000 BPM. These misses
are **not proof of motor incapacity**. They reveal an overly strict
full-horizon/terminal-abstention rule. It must be addressed in an isolated
successor design, not silently changed in this diagnostic.

### Why extra mid-chart misses appeared at 4,000 BPM

The physical Motor generated a KeyDown for many planned notes, but the
**evaluator was waiting on a different target** (7 such misses on 72 notes,
16 on 144 notes). For example, the planned note at 300ms had a matched
physical KeyDown around 338ms and the evaluator associated that press
with an earlier target. The repeated latency/error causes a sequence-level
misalignment. Simply counting Hit or planned events hides this behavior.

The simulator's evaluator has timing windows and sequential target-claim
semantics; a press intended for one target may earn a Hit on another.
Therefore *Hit count is not intended-action precision*, and these diagnostic
associations do not prove a unique physical cause. There is no evidence here
of generic full-chart 4,000 BPM mastery or 20,000 BPM feasibility.

## Reproduction

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_persistent_motor_endurance_audit.py
uv run python -u scripts/audit_prestige1_autoplay_persistent_motor_endurance.py --config configs/training/prestige1.toml
~~~

Immutable new report (refuses overwrites):

    diagnostics/prestige1/stage_timing_persistent_motor_endurance_attribution_readonly.json

## Next investigations (NOT implemented)

1. A bounded *visible-prefix* forecaster that publishes the known portion
   of a future event schedule without falsely claiming complete 70ms coverage.
2. Quantify whether physical/hand coupling, command latency, finger reuse or
   per-target alignment causes the mid-chart mismatch. Avoid training or
   changing the original motor profile in the diagnostic.
3. Only then test persistent trajectory control (as an isolated experiment),
   separating evaluator Hits from intended KeyDown time accuracy.
