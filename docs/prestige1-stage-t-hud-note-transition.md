# Prestige I Stage T — 263D HUD changes across note switches

## Objective

The saved stateless Hybrid baseline lowered chart-disjoint Train-heldout
near-MAE from 36.926ms to 18.723ms (2,002 near frames), but observed
Geometry-to-GRU at 87 true target changes with ~80ms arrival error.
Does the visible 263D HUD expose a reliable transition signal, and how
soon is ordinary-floor Geometry eligible again after a target change?

This is a read-only diagnostic, NOT an event detector, new architecture,
training run or deployment. It should precede any stateful tracker.

## Information boundary

- Frozen expert-driven Train20 visible HUD inputs only.
- Geometry/GRU predictions are produced by unchanged
  infer_hybrid(visible, frozen_near_only_gru) before oracle evaluation.
- hud_delta(HUD[k-1], HUD[k]) uses exclusively causal, same-time features.
- Auto next-event index, exact remaining time, near flag and observability
  mask are used afterwards, only to label pairs, construct post-switch
  strata and score prediction errors.
- No privileged floor index, chart schedule, unobservable target, future
  frame, oracle-triggered inference or model selection.

## Feature groups

Use descriptive fixed epsilon=1e-6 to mark changes between adjacent
10ms HUD frames, using the current 263D representation:

- motor state (24 features) and visible orbit xy (2 features)
- previous/current/next floor slot changes (15 features per floor),
  and any other visible floor slot change
- current floor x/y, entry+exit sin/cos, Midspin/multiplanet/event
  flags, and slot-present bit
- next floor entry+exit and event flags
- HUD Tile BPM, Real BPM and visible feedback/judgement flags

Report change frequency on true note-switch pairs versus pairs where
Auto's current target did not change, without fitting thresholds.
For each fixed cue report TP, FP, precision, recall and specificity
as posthoc descriptive rates only. They are not autonomous detector
results. Orbit angular step and current floor max-delta are also
summarized.

## First frames after a true note switch

For each evaluator-labelled switch at frame k, without altering Hybrid:

- same-event +0,+10,+20,+30,+40ms offsets, truncated if next target
  change occurs, or the HUD becomes unobservable or ends;
- Geometry/GRU mode at each offset, timestamp errors for Hybrid,
  frozen fallback, and eligible Geometry;
- first Geometry-eligible frame inside new oracle event, observed
  delay in 10ms control steps, timestamp error and explicit reasons
  for no recovered Geometry before the event/segment ends;
- top 5 worst first-frame absolute timing errors per split with
  HUD changes and fallback reason.

First Geometry eligibility delay is NOT event-tracking re-synchronization
latency; no stateful tracker exists. Oracle event identity is only used
to censor the evaluator window after all inference.

## Frozen provenance checks

Before writing output:

- Original Stage A FAIL, exact Autoplay and Stage T provenance;
- SHA256 of frozen near-only GRU checkpoint and reports;
- prior near-only distance and countdown audits;
- replay the stateless Hybrid and verify frozen scores,
  pair counts, Train20 partition and source hashes;
- no Validation, FINAL, Motor/Fingering training, or checkpoint writes.

A new unique JSON output is refused if it already exists. No existing
training or diagnostic artifact is overwritten.

## Windows commands

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_switch_hud_audit.py
uv run python -u scripts/audit_prestige1_autoplay_switch_hud.py --config configs/training/prestige1.toml
~~~

Output:

- diagnostics/prestige1/stage_timing_autoplay_hud_note_transition_train20.json

Train-fit and Train-heldout results remain separate and nothing is
adopted automatically. Stage A remains FAIL throughout.
