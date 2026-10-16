# Prestige I Stage T — Read-only special marker / motion abstention breakdown

## Research question

The previously saved first-frame Geometry audit reported, for chart-disjoint
Train-heldout (148 oracle-labelled target-switch pairs):

- **Original** frozen Geometry: floor_changed=96,
  inconsistent_orbit_motion=51, special_event=1.
- **Carried-step COUNTERFACTUAL after relaxing only floor_changed**:
  eligible=37, special_event=53, inconsistent_orbit_motion=4,
  ambiguous_full_turn=1, midspin_or_multiplanet=1,
  not_floor_changed=52.

These are *different reason spaces*. In particular, 53 is the count of
the **carried-step candidate's** special_event abstention, not the count
of original Geometry special_event failures. Likewise, 51 is an
**original** inconsistent_orbit_motion count. Do not add counts across
the reason spaces.

This experiment separates those reasons without lifting a guard,
retraining the network, or changing the Hybrid output.

## What it extracts from the 263D HUD

Two precommitted cohorts (evaluated after all causal HUD analysis):

A. carried_special_event:
   original frozen counterfactual carried-step predicts no time and
   returns status special_event at a true switch.
B. original_inconsistent_orbit_motion:
   unmodified original Geometry returns inconsistent_orbit_motion
   at a true switch.

Also report an independent auxiliary cohort
floor_changed_then_carried_motion_inconsistent, distinct from B.

For **each current and next visible floor** inspect the six fixed marker
bits, in their actual HUD layout order:

- Twirl
- SetSpeed
- MultiPlanet
- Pause
- Checkpoint
- SetFloorIcon

Record marker occurrence by slot (OVERLAPPING by definition), identical
marker on both slots, and exclusive combinations such as
current:Twirl,next:Checkpoint. Also report current/next Midspin and
multiplanet structural flags independently.

The same HUD-only motion snapshot reports:

- crossing orbit angular displacement from k-1 to k;
- prior orbit angular displacement from k-2 to k-1;
- signed angular difference and direction change;
- current and previous Tile BPM and Real BPM, decoded directly from
  **visible HUD log-BPM fields** (not the privileged simulator clock);
- |crossing step| / expected step from CURRENT Tile BPM;
- |prior step| / expected step from CURRENT Tile BPM;
- |prior step| / expected step from PREVIOUS Tile BPM;
- where each ratio falls relative to original fixed [0.5,1.5] guard;
- whether visible Tile BPM or Real BPM changed at the boundary.

The three ratios help tell apart a floor-crossing angular coordinate
reset and a legitimate BPM discontinuity. They are descriptive
evidence, not certified causes of an out-of-band ratio.

For traceability, output a crosstab of the original vs carried reason,
per-anchor counts, a handful of worst crossing ratio samples, and
marker arrangements for the special cohort.

## No oracle leaks / no change to baseline

The code calls the frozen first-frame candidate generator and produces
motion snapshots on **every HUD frame first**, using only past/current
263D frames. After that pass, the evaluator uses oracle event IDs and
observability masks only to stratify switch vs same-event samples.

The original Geometry and all previous diagnostic JSON/checkpoints remain
byte-for-byte untouched. Before any write, revalidate frozen Stage T,
near-only GRU, stateless Hybrid, HUD-switch and first-frame reports,
including checkpoint SHA256 and original Stage A FAIL. The new output
uses an exclusive write and refuses to overwrite an earlier artifact.
No Validation/FINAL, Motor/Fingering learning, hyperparameter search,
or model adoption. The formerly unseen Train-heldout charts have
now been used repeatedly for research, **not** as an independent
final acceptance holdout.

## Windows commands

Inspect the existing worktree and preserve local changes:

~~~powershell
git status --short
git pull --ff-only

uv run pytest -q tests/test_prestige1_autoplay_special_motion_audit.py

uv run python -u scripts/audit_prestige1_autoplay_special_motion.py --config configs/training/prestige1.toml
~~~

New report:

- diagnostics/prestige1/stage_timing_autoplay_special_motion_train20.json

The output is classification-only. Even if some marker types correlate
with low timing error elsewhere, no event guard is removed until a
separate justified, independently evaluated inference design exists.
