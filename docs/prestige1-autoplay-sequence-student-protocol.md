# Prestige I — Autoplay-supervised multi-event Timing protocol

Status: preregistered independent Timing research. It does not alter the
frozen Stage A FAIL gate, Motor/Fingering learning, 263D HUD layout, or an
existing checkpoint.

## Causal contract

- Teacher/evaluator: exact `AutoplayTimingTeacher` onset times, retained as
  Python doubles before float32 loss conversion. No 10ms/1ms label rounding.
- Student: forward-only history of the unchanged 263D visible HUD. No chart
  clock, target ID, floor ID, compiled schedule, future HUD, or teacher action
  is concatenated to model input.
- Output: up to 16 strictly ordered events in the next 80ms, plus one
  existence probability per slot. Low-confidence output is an explicit
  abstention. Equal-time notes remain one chord/event.
- Multiple events inside one 10ms controller interval remain separate labels.
  Repeated predictions across frames will be reconciled by a causal tracker
  before scoring, so one target can be credited at most once.

## Frozen study order

1. Reproduce the saved Train20 exact-Autoplay readout and hashes.
2. Train only on the prior Train-fit chart hashes. Use its five prior
   Train-heldout hashes for checkpoint/threshold selection. These are not a
   new unknown-chart claim.
3. Use the already-opened, checkpoint-unseen Development20 for the first
   matched comparison. The already-opened Final10 is retrospective stress
   evidence only and can never be relabeled as holdout.
4. Open Qualification20 only if the new Student beats Timing v1.4 on
   Development20 at the preregistered practical endpoint: unique-event F1 at
   10ms tolerance and 10–80ms lead improves by at least 0.02, false forecasts
   per target do not increase, MAE is at most 10ms, and median first-prediction
   lead is at least 20ms. Reserve20 remains unopened.

Primary comparisons use the same captured frames and exact targets for Timing
v1.4, saved MLP, saved GRU, and the new Student. Report unique matching at
1/5/10/20ms tolerances, MAE/p95, false forecasts, abstention, lead bins
(0–10, 10–20, 20–40, 40–80ms), event-count error, confidence calibration,
10ms multi-event frames, and visible Pause/SetSpeed/Twirl/full-turn cohorts.
High-BPM simulator fixtures are a separate stress test, not unknown-chart
generalization evidence.

The experiment fails rather than relaxing these criteria after seeing
Development results. Qualification/Reserve are not used for threshold or
model selection.
