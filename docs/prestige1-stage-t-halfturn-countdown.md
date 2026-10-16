# Prestige I Stage T — Is the 180-degree correction also a valid countdown?

## Starting observations (frozen)

The Train-heldout 180-degree angle audit found that, among 148
oracle-labelled note-switch pairs, **51** returned the ORIGINAL
Geometry abstention inconsistent_orbit_motion. All 51 had:

- The current-floor 15D HUD slot unchanged through the switch.
- |wrap(crossing step - prior step - pi)| within 1 degree.
- Half-turn-corrected speed within the ORIGINAL [0.5,1.5] ratio band.
- Prior speed matching current Tile BPM, with no Tile BPM changes.
- Angle residual of approximately 0.0000049 degrees.

This demonstrates mathematical consistency with a 180-degree coordinate
jump, **not** by itself correct target-time inference or deployable switch
detection.

## Fixed, HUD-only counterfactual estimators

Both proposals are made on EVERY frame (not only oracle switches).
Both require original, unchanged geometry_one(previous,current) to return
exactly inconsistent_orbit_motion; other original abstentions (floor
change, missing visible floor, special event, midspin, full turn) are
NOT overridden.

Given only HUD[k-2], HUD[k-1], HUD[k]:

1. cross = wrap(orbit-angle[k] - orbit-angle[k-1])
2. prior = wrap(orbit-angle[k-1] - orbit-angle[k-2])
3. corrected = wrap(cross - pi)
4. angular-rate expectation = pi * visible Tile BPM / 60, multiplied
   by the original control tick duration.

Two fixed candidates:

- halfturn_speed_only: corrected angle-step magnitude must satisfy the
  frozen Geometry [0.5,1.5] speed ratio guard.
- halfturn_prior_agreement: in addition, preceding angular step must
  satisfy the same speed ratio band against current Tile BPM and
  |wrap(corrected-prior)| must be <=1 degree.

The corrected-step calculation supplies a synthetic PREVIOUS orbit
angle to the unchanged geometry_one() implementation; current HUD and
every visible floor feature remain untouched. No true target, absolute
floor index, privileged timestamp, future HUD or labels can influence
the candidate.

The 1-degree prior-agreement threshold is fixed before this evaluation.
It is NOT swept or tuned on heldout. Neither candidate is adopted.

## What is actually scored

Use *after inference* oracle event IDs to separate all pairs with valid
current and preceding HUD frames:

- target-switch pairs (candidate eligibility, timing-error MAE, P95,
  matched same-frame frozen GRU and unchanged Hybrid MAE);
- same-event pairs (candidate eligibility constitutes a detection
  false positive, rate relative to all same-event pairs, and errors);
- per-split descriptive precision across the two observable classes;
- near-frame errors, abstention reasons and worst errors.

This addresses two distinct questions:

A. Does the corrected Geometry predict the correct next target time
   where the old Geometry returns inconsistent_orbit_motion?
B. Would the same HUD-only guard spuriously activate inside an oracle
   event, where no target switch occurred?

The actual motor decision is NOT tested. Even an accurate pointwise
countdown does not implement a stateful note tracker, and does not
justify new Motor training or changes to Stage A.

## Frozen-artifact guards

Recompute frozen Hybrid traces, validate near-only GRU and Hybrid scores,
and compare exact first-frame/halfturn original Geometry switch reasons
and pair counts to saved reports; verify SHA256-protected original
checkpoints, source manifests, and Stage A FAIL.

New output is written exclusively; existing diagnostics, original
Geometry, Hybrid, Motor and Stage T weights remain untouched.
All scoring uses Train20 fit and repeatedly-inspected chart-disjoint
Train-heldout only. No Validation or FINAL.

## Windows execution

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_halfturn_countdown_audit.py
uv run python -u scripts/audit_prestige1_autoplay_halfturn_countdown.py --config configs/training/prestige1.toml
~~~

New output (once):

- diagnostics/prestige1/stage_timing_autoplay_halfturn_countdown_train20.json

Stage A remains FAIL. No model training, hyperparameter optimization,
guard removal, adoption, Motor/Fingering training, Validation or FINAL.
