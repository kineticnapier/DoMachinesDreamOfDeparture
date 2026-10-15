# DMDOD Prestige I — Saved near-only GRU countdown comparison

## Why

The controlled Train20 near-only regression ablation (no far >200ms
regression) reduced heldout near MAE from **47.328ms** to **36.926ms**,
beating the same heldout constant's **41.924ms**. However, it did not
meet the fixed 20%-better research criterion and the original Stage A
gate is **still FAIL**.

The missing question: Did the better GRU actually learn to count down
at ~10ms per 10ms control step, or merely shift its mean prediction?

No new training or parameter selection is allowed in this audit.

## Procedure

1. Reconstruct the original Train20 chart-disjoint Train-fit and
   Train-heldout HUD streams via the existing frozen replay guard.
2. Verify SHA256 for frozen original Stage T report, constant/MLP/GRU
   checkpoints, Autoplay truth, v3.0 source plans, **prior distance
   diagnostic**, saved near-only GRU checkpoint and its report, and
   **prior Countdown diagnostic**.
3. Verify saved near-only checkpoint metadata/architecture and match
   *all* re-evaluated heldout and Train-fit near MAE, p50/p95,
   all-visible MAE, classification counts and per-anchor scores.
   Abort on drift. The source HUD, time labels and model parameters
   remain immutable throughout.
4. Measure the **same-event true countdown** against the frozen
   constant, MLP, old GRU, and near-only GRU on exactly the same HUD
   frames, using the existing Countdown definition.
5. Separately measure **target switches**. An actual switch is defined
   by evaluator-only event identity; predictions never see the ID.
6. Compare old GRU and near-only GRU to the HUD-only geometry heuristic
   on identical eligible near-event frames, and verify eligibility and
   original GRU geometry MAE against the older countdown audit.
7. Produce write-once JSON and 5 SVG overlays. No checkpoint is
   selected or adopted from these metrics.

## How to interpret

For same-event adjacent HUD observations 10ms apart:

    actual_remaining_next - actual_remaining_now = -10ms

A true countdown predicts approximately -10ms / control step.
A constant predictor predicts 0ms / step and has a countdown error
of +10ms. The new report records:

- **near predicted step** (mean change in predicted remaining time,
  ideal -10ms),
- **near countdown MAE / P95** (absolute step error against -10ms),
- near-time MAE and all-visible MAE, which detect **absolute timing
  offset**, even for a model with a perfect countdown slope,
- target-switch jump MAE and upward-jump recall (only when a positive
  true jump is present),
- same-frame HUD geometry subset near MAE for old and near-only GRUs.

If the near-only GRU remains ~0ms/step, it has **not** learned to
count down even though its absolute near MAE improved. If countdown
approaches -10ms/step but absolute near MAE remains high, its time
rate is improving without adequate time alignment. If countdown
and absolute near MAE improve, we have stronger temporal evidence,
though this does not establish independent closed-loop timing.

The model output is one next-event estimate at each 10ms control
frame: multiple real targets inside one frame are not represented
separately. Data are expert-controlled visible trajectories.

## Files

- src/dmdod/training/prestige1_autoplay_near_countdown_audit.py
- scripts/audit_prestige1_autoplay_near_only_countdown.py
- tests/test_prestige1_autoplay_near_countdown_audit.py

Inputs: all previously saved Stage T, exact-Autoplay, distance,
Countdown and near-only GRU reports/checkpoints, SHA-checked.

Write-once outputs in diagnostics/prestige1:
- stage_timing_autoplay_near_only_countdown_train20.json
- stage_timing_autoplay_near_only_countdown_holdout_02.svg
- stage_timing_autoplay_near_only_countdown_holdout_07.svg
- stage_timing_autoplay_near_only_countdown_holdout_13.svg
- stage_timing_autoplay_near_only_countdown_holdout_14.svg
- stage_timing_autoplay_near_only_countdown_holdout_20.svg

## Windows steps

Use the existing branch, preserve local uncommitted work and first
inspect git status. A fast-forward pull is safe only when Git says so.

~~~powershell
git status --short
git pull --ff-only

uv run pytest -q tests/test_prestige1_autoplay_near_countdown_audit.py

# Read-only inference with already saved models; NO train.
uv run python -u scripts/audit_prestige1_autoplay_near_only_countdown.py --config configs/training/prestige1.toml
~~~

Do not overwrite generated outputs, reset/clean/stash, retrain models,
use Validation/FINAL, or reinterpret the research score as PASS.
