# Prestige I Stage T — Projective orbit / double-angle read-only diagnostic

## Why

Original Geometry sometimes interpreted pi-radian jumps as angular
acceleration. A fixed 180-degree counterfactual matched 115/115
Train-fit and 51/51 Train-heldout original motion-abstention cases.
However, of the original floor_changed switches, 131/403 fit and
35/96 heldout failed the prior 180-degree current-only step-agreement
test. The preceding step may itself have crossed a planet swap.

## Quotient representation

The HUD stores (sin(theta), cos(theta)), rather than the conventional
Cartesian order. Normalize and convert to z=(cos(theta),sin(theta)).

For successive frames compute D=z_current*conjugate(z_previous),
then:

    projective_step = 0.5 * atan2(
        2 * Re(D)*Im(D),
        Re(D)**2 - Im(D)**2)

Equivalent: half the principal argument of D**2. The result is invariant
under theta -> theta+k*pi for any integer k.

Evaluate projective steps at t-1 and t using HUD[t-2:t+1]. All candidate
calculations run on every HUD frame before oracle labels are used.

### Aliasing

Signed projective step lies in [-pi/2,pi/2]. A physical angular
advance >=90 degrees per frame can alias. The nominal stock bound
for a 10ms tick is Tile BPM 3000. Since original Geometry permits a
speed ratio of up to 1.5, this diagnostic abstains conservatively when
1.5 * expected step >= pi/2, i.e. at Tile BPM >=2000.
This bound comes from the pre-existing guard, not heldout tuning.

## Frozen counterfactual options

Apply only when ORIGINAL Geometry first reports floor_changed or
inconsistent_orbit_motion:

- projective_speed: current projective step must pass original [0.5,1.5]
  angular-speed check; all the other original guards remain active.
- projective_prior_agreement: previous projective step must pass the
  same check, and current-vs-previous projective step difference must
  be <=1 degree.

For floor_changed only, the synthetic previous frame's current-floor
slot is replaced with the current HUD slot to pass that one guard.
For both groups, the previous orbit sin/cos is synthesized so that the
original, frozen geometry_one evaluates the new angular step. The
CURRENT HUD, floor entry/exit angle, marker bits and network are not
changed.

## Posthoc evaluation: do not cherry-pick

Classify predictions only AFTER all causal HUD inference:

- ALL true switches and ALL same-event pairs, including false-positive
  eligibility, matched GRU/Hybrid MAE and projective Geometry MAE/P95.
- Original motion-abstention 51 Train-heldout cases.
- Original floor_changed 96 Train-heldout cases.
- Within floor_changed: OLD halfturn explained 61, unexplained 35.
- Angular consistency within 1 degree in each subgroup.
- Existing carried_previous_step coverage/accuracy for each subgroup.

Old 61/35 membership depends on posthoc oracle switch classification,
and is not a candidate input or trigger. The quotient itself destroys
planet parity. It is NOT an online event detector or stateful tracker.

### Provenance / safety

Source manifest, config hash, near-only GRU checkpoint, frozen Hybrid
scores, original Geometry switch reasons and frozen halfturn report
(61/35) are verified. Stage A FAIL remains frozen; no training,
new adopted guard, Motor/Fingering learning, Validation or FINAL.
The previously viewed Train-heldout charts are exploratory, not a
fresh independent evaluation set.

Output is exclusive and will not overwrite previous diagnostic files:

    diagnostics/prestige1/stage_timing_autoplay_projective_orbit_train20.json

The previous halfturn COUNTDOWN diagnostic is optional, not required.

## Windows

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_projective_orbit_audit.py
uv run python -u scripts/audit_prestige1_autoplay_projective_orbit.py --config configs/training/prestige1.toml
~~~

Interpretation: accurate angular speed does NOT imply accurate
next-target countdown, and neither establishes safe Motor adoption.
