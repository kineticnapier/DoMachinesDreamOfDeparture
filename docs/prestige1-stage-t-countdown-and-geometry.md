# DMDOD Prestige I — Does the Timing network actually count down?

## Research question and frozen status

Stage T exact-Autoplay supervised GRU produced heldout near-MAE 47.328ms,
worse than a constant near-time predictor (41.924ms), despite beating
the single-frame MLP (59.649ms). That does not establish whether the GRU
tracks elapsed time or merely predicts a distributional average.

Before changing the training loss, evaluate **frozen trained weights**
on identical chart-SHA256-disjoint Train20 expert-controlled visible HUD.
The entire experiment is read-only and research-only. No Stage A gate
change, Motor/Fingering training, Validation/FINAL, or new checkpoint.

## Countdown metric

At two consecutive control observations k, k+1 with the **same true
event ordinal** and both oracle-visible:

    true_remaining[k+1] - true_remaining[k] = -10ms

Measure:

    countdown_residual_ms =
       1000 * (predicted_remaining[k+1] - predicted_remaining[k])
       - 1000 * (true_remaining[k+1] - true_remaining[k])

Ideal residual is 0ms. A constant predictor gives +10ms. Report
absolute MAE, signed residual (bias), median/p95, within ±2ms and ±5ms,
and mean predicted time step. Also report the near-only subset where
both consecutive frames have oracle true remaining <=200ms.

**Important:** low countdown error alone is NOT enough: a model can
count down perfectly while remaining consistently 20ms late. Compare
with the absolute near-MAE and signed bias already available.

The oracle event ordinal is used *only after inference* to segment
the evaluator's pair metrics. It NEVER enters either model or HUD
geometry heuristic.

## Target switching metric

For each pair k,k+1 with a different oracle event ordinal, separately
measure how close the predicted jump is to the true next-target jump.
Report jump residual MAE, and upward-jump recall where truth actually
jumps upward. Events <10ms apart can have a **nonpositive** inter-sample
true jump, even though the target changed; report these separately
rather than falsely marking all non-increasing changes as failures.

## Visual plots (no optional plotting dependencies)

Five write-once **SVG** traces, one per heldout Train chart (#02, #07,
#13, #14, #20), show an approximately 61-frame window surrounding the
first visible event switch:

- black: oracle next-event remaining time
- gray: training-fit constant baseline
- orange: frozen MLP
- blue: frozen GRU
- dashed vertical lines: **evaluator-only** target switch

SVG y-axis is *display-limited* to 350ms so close events remain
legible. The underlying oracle targets and predictions are not
truncated and all computed errors use full values.

## Conditional geometry baseline

Input **only from successive 263D HUD frames**:
- orbit xy (features 24,25),
- relative floor 0 and +1 geometry (15D slots),
- visible Tile BPM in the appended HUD block.

It determines orbit phase using atan2(x,y), infers spin direction
from the prior and current orbit observations, and estimates:

    angular_velocity = pi * tile_BPM / 60
    delta_time = forward_arc(current_orbit_phase, current_floor_exit_angle)
                 / angular_velocity

The HUD encoder does **not** include an explicit clockwise/CCW flag.
The Tile BPM, not Real BPM, matches the simple-floor orbital angular
speed in this renderer. This inference is intended only as a baseline
against the same simulated HUD, not validated native ADOFAI parity.

The baseline abstains when current/next regular floor is unavailable,
current tile changes across frames, Twirl/SetSpeed/Pause/Checkpoint/
MultiPlanet/event flags appear, a midspin/multiplanet is visible,
angular motion is absent or inconsistent with BPM, or the floor
rotational arc is ambiguous due to a possible 360-degree turn.

Eligibility is decided from **HUD features alone**, before checking
oracle timestamps. On eligible frames, compare Geometry, Constant,
MLP, and GRU against the same hidden oracle times. Report eligible
frame count/coverage and exclusion reasons; results only represent
this restricted subset. Ineligible frames do not get fabricated
geometry guesses. The heuristic may still be wrong on otherwise
eligible cases because it ignores full runtime subtleties.

## Code and commands

Files:
- src/dmdod/training/prestige1_autoplay_countdown.py
- scripts/audit_prestige1_autoplay_countdown.py
- tests/test_prestige1_autoplay_countdown.py

The new command uses the already existing SHA-protected frozen Stage T
replay path. It crosschecks captured frame counts, aggregate near/full
MAE, FPR and original saved checkpoint weights before diagnosis.

Run **first**, before attempting the already implemented near-only
loss ablation:

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_countdown.py tests/test_prestige1_autoplay_distance_and_ablation.py
uv run python -u scripts/audit_prestige1_autoplay_countdown.py --config configs/training/prestige1.toml
~~~

Write-once files in diagnostics/prestige1:
- stage_timing_autoplay_countdown_train20.json
- stage_timing_autoplay_countdown_holdout_02.svg
- stage_timing_autoplay_countdown_holdout_07.svg
- stage_timing_autoplay_countdown_holdout_13.svg
- stage_timing_autoplay_countdown_holdout_14.svg
- stage_timing_autoplay_countdown_holdout_20.svg

Next, inspect the countdown vs near MAE and frame-matched geometry
coverage, then run the separate, fixed loss ablation already provided.
No additional hyperparameter search is authorized by this audit.

If any source is changed or preexisting output exists, stop rather
than reset/stash/clean, overwrite, retrain, or use Validation/FINAL.
