# DMDOD Prestige I / Stage T — Projective Geometry timing error attribution

## Why

The read-only double-angle (RP1) experiment reconstructed countdowns at
92/148 Train-heldout true switch frames with conditional Timing MAE
0.000007ms; all 51 original motion-abstention cases succeeded.
On Train-fit, 351/555 eligible switch frames had MAE 0.346312ms, and
the 236 eligible original floor_changed cases had MAE **0.515064ms**.

The discrepancy MUST NOT be dismissed by looking only at heldout mean.
The next question is whether a few large outliers dominate the Train-fit
average, whether BPM changes or geometry/phase boundaries correlate
with error, and which anchors contain the errors.

## What this immutable audit reports

On both Train-fit and repeatedly used chart-disjoint Train-heldout:

- Recreate the SAME original projective_speed / projective_prior_agreement
  proposals from the same causal 263D HUD stream. No threshold/algorithm
  changes and no oracle gating.
- After all HUD predictions, use evaluator-only event IDs/masks/time to
  identify true note switches and conditional Timing errors.
- For each accepted original floor_changed candidate, keep signed error,
  absolute error, control-frame equivalent, phase equivalent in degrees,
  oracle remaining time, predicted time, same-frame frozen GRU/Hybrid error.
- Record HUD Tile BPM / Real BPM before and after, whether each changes,
  present/current floor angles, relative floor center, orbit angle,
  projective step/previous step, projective speed ratio, and markers.
- Report MAE, bias, median/P95 absolute error, counts exceeding fixed
  descriptive thresholds 0.01/0.1/1/5/20/100ms; absolute error
  concentration by anchor; Tile/Real BPM splits; 16 worst individual
  errors per candidate.
- Account for EVERY noneligible candidate frame by the unchanged
  projective reason and original Geometry reason. In Train-heldout this
  should partition the 56 remaining original switch frames.

This is **not** a new hyperparameter experiment or tuned classifier.

## Frozen verification

The previous projective report must exist. The new audit replays the
original Stage T, GRU, Hybrid and RP1 diagnostics; it rejects if chart
partition, expected source SHA256, prior report hash, original Geometry
reason counts, candidate eligibility, per-cohort MAE, Stage A gate or
frozen checkpoint data differ. Report names never overwrite previous
artifacts.

All original models and thresholds remain immutable. Read-only,
Train20 only, previously inspected heldout is NOT clean FINAL.
Stage A FAIL remains. No Motor/Fingering, Validation or FINAL.

## Windows

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_projective_error_audit.py
uv run python -u scripts/audit_prestige1_autoplay_projective_error.py --config configs/training/prestige1.toml
~~~

New immutable report:

    diagnostics/prestige1/stage_timing_autoplay_projective_error_train20.json

Inspect worst anchor/frame errors, not just the mean. Even if outliers
are isolated, correcting them is a separate proposed change requiring
new safety evaluation. The diagnostic does NOT modify Geometry/Hybrid.
