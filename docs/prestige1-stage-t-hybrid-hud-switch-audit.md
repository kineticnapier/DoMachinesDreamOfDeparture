# Prestige I Stage T — Stateless HUD Hybrid Timing and transition continuity

## Question

Can a non-learning, inference-only selector improve timing accuracy
without importing hidden chart knowledge, and does switching Geometry
to/from GRU introduce timing discontinuities?

Frozen heldout Train5 near-event MAE:
- Constant 41.924ms;
- near-only GRU 36.926ms;
- geometry on eligible 1,046 frames: 0.675ms;
- previous arithmetic reconstruction of immediate framewise Hybrid:
  ~18.72ms (rounded-aggregate prediction, NOT direct measurement).

This audit obtains a measured value and mode-switch errors on identical
Train20 captured HUD frames. It cannot change frozen Stage A FAIL.

## Exact inference contract

Inputs to infer_hybrid(visible, frozen_near_gru):

1. Causal 263D visible HUD in chronological order.
2. Frozen, SHA-verified, near-only GRU weights.

For frame k:

- Run the strictly forward-only GRU over visible HUD.
- If geometry_one(HUD[k-1], HUD[k]) is eligible, use Geometry.
- Otherwise use frozen near-only GRU output.

Frame zero falls back to GRU because no prior HUD exists.
Eligibility is computed from past/current HUD alone; it is NEVER
conditioned on event_index, oracle remaining_s, oracle near/mask,
privileged absolute time, chart future or hit outcome.

This is a STATELESS framewise baseline (apart from two HUD samples).
It has NO event tracker, smoothing, resync timer, oracle-triggered
reset, teacher forcing, model retraining or hyperparameter selection.

## Evaluation protocol

After predictions are finalized, the evaluator may use oracle
timestamps/IDs/masks to score:

- near and all-visible MAE, median, P95, signed bias and counts;
- Geometry eligibility, abstention reasons and per-anchor coverage;
- Constant, GRU, Geometry (eligible subset), Hybrid (all frames),
  always compared on the very same eligible frames for subset metrics;
- Train-fit and independently chart-disjoint Train-heldout;
- switching stats for G->G, G->GRU, GRU->G, GRU->GRU;
- each mode pair separately for oracle **same-event** and **event-changed**;
- error immediately before and after switching, raw predicted step,
  error in predicted step vs actual event-time step, GRU-only
  counterfactual step on those very same transition pairs;
- fixed diagnostic threshold |step residual| >20ms (not tuned);
- event-change positive-jump direction recall, and spurious upward
  steps while the oracle target is unchanged;
- near-pair subset of the mode-transition timing residual.

Importantly, a true event switch may correctly cause the remaining
time to jump upward; it is never labeled as a SAME-event anomaly.
Event IDs are only used to label results after inference. Jump
direction recall is NOT evidence of an autonomous event-ID tracker.

The script does NOT report tracking-loss duration or resync latency:
neither concept is defined for a stateless baseline. A stateful
event-tracker requires a later separate design.

Five SVG overlays show true Auto time, Near-only GRU, Hybrid, and
Geometry eligible points. Vertical plot values cap at 350ms **for
display only**; full timestamps are used for numerical metrics.

## Frozen data safeguards

Re-capture only Train20 on identical expert-driven HUD trajectories,
and verify SHA256/format/metadata of Stage A FAIL, exact Autoplay,
saved old Stage T and near-only GRU, distance report, original
Countdown report, and saved near-only Countdown report.
Recalculate old near/full MAE and countdown metrics and require
the same Geometry eligibility and near error. The experiment is
write-once and refuses overwrites. No Train-heldout tuning,
Validation, FINAL, Motor, Fingering, or new training.

## Windows commands

Check the existing work tree first; do not discard local changes.

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_hybrid_audit.py
uv run python -u scripts/audit_prestige1_autoplay_hybrid_switch.py --config configs/training/prestige1.toml
~~~

Outputs in diagnostics/prestige1:

- stage_timing_autoplay_hybrid_switch_train20.json
- stage_timing_autoplay_hybrid_switch_holdout_02.svg
- stage_timing_autoplay_hybrid_switch_holdout_07.svg
- stage_timing_autoplay_hybrid_switch_holdout_13.svg
- stage_timing_autoplay_hybrid_switch_holdout_14.svg
- stage_timing_autoplay_hybrid_switch_holdout_20.svg

The earlier ~18.72ms calculation is a consistency reference, NOT a
pass condition. Do not change Stage A to pass on these results.
