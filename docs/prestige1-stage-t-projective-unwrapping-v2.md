# Prestige I / Stage T — Read-only high-BPM projective angle unwrapping

## Motivation and input contract

Previous Stage T RP1 uses z=exp(2i*theta) from a causal 263D HUD.
The projective angular difference is constrained to [-90,+90] degrees;
at 4000 BPM over the existing 10ms control tick, real +120 degrees appears
as -60 degrees. At 8000 BPM, real +240 degrees appears as +60 degrees.

The old HUD v1 saturated both BPM entries at 1920. The new **default**
v2 encoder instead returns log2(BPM)-log2(120) with no BPM clamps.
This experiment must independently re-capture expert HUD v2. It must NOT
feed v2 HUD into the frozen Stage A or Stage T neural checkpoints.

The Stage T v1 preparation is used only as a protected source of chart
manifest, Train20 labels and trajectory-provenance baseline. The v1
capture function remains explicitly pinned to legacy BPM coding for
artifact reproducibility. The separate v2 capture drives the SAME
expert (same input motor actions / 10ms physics/control clocks) and
rejects if ANY non-BPM HUD element differs from the frozen v1 trace.

## Algorithm (no oracle labels, zero training)

On each observed frame pair, compute the usual RP1 step r in
[-pi/2,pi/2]. Compute expected absolute angular step

    expected = pi * TileBPM / 60 * 0.010

from the human-visible unclipped v2 Tile BPM feature.

Build candidates r+k*pi around the two closest signed-speed branches,
k~round((+expected-r)/pi) and k~round((-expected-r)/pi), plus their
fixed +-2 neighbors (no loop proportional to BPM). If the phase count
exceeds floating-point exact resolution, abstain as numerical ambiguity.

The candidate with minimum |abs(r+k*pi)-expected| is accepted ONLY if:

1. Tile BPM was constant across the two observed HUD frames (otherwise
   current BPM cannot determine the preceding interval's angle integral);
2. best mismatch <= 10% of expected;
3. runner-up mismatch exceeds best mismatch by MORE THAN 10% of expected;
4. original Geometry guards for visible floors, special events, full
   turns, midspin, multiplanet and stopped/invalid orbit remain effective
   in the diagnostic counterfactual.

These are fixed, conservative diagnostic choices, **not** heldout-tuned
thresholds. They are not adopted in Geometry or Hybrid.

An optional prior-agreement candidate additionally requires the
independently unwrapped prior frame's signed step to agree within 1deg.
There is no explicit clockwise/counterclockwise feature in the existing
263D geometry HUD; thus an equally plausible clockwise and
counterclockwise candidate is **ambiguous** and abstains.

The counterfactual retains all old Geometry checks except deliberately
examining the old floor_changed guard and replacing the old wrapped
speed calculation with the chosen full signed step. The predicted
remaining time still uses the old simple single-target floor-angle
assumption, which may fail on multi-event transitions.

### Previously false successes at high BPM

An original geometry_one() status of eligible_simple_floor is NOT
necessarily safe above the alias threshold. For instance,
+225 degrees per frame is observed as -135 degrees by its ordinary
[-180,+180] wrap. The old Geometry can pass its speed band (135/225=.6)
but predict the WRONG rotation direction.

Accordingly, this diagnostic ALSO evaluates old eligible_simple_floor
and stopped_orbit cases when *visible* TileBPM>=3000, as well as the
previous floor_changed and inconsistent_orbit_motion cohorts.
Ordinary low-BPM eligible Geometry predictions are left untouched.

A 3000 BPM +/-90deg step, 6000 BPM +/-180deg step, or 12000 BPM
+/-360deg step has symmetric possible signs in RP1 for constant
speed. The diagnostic abstains rather than guessing. Information
theory does not permit unconditional signed recovery from the current
3-frame HUD where all successive samples share that symmetry.

## Measured quantities

Without using labels in predictor selection, evaluate after inference
on Train20 fit / previously reviewed chart-disjoint heldout:

- every true event-switch pair and every same-event pair;
- pairs where unclipped Tile BPM exceeds 1920 / 3000;
- candidate eligibility and abstention reasons across BPM bands
  under1920, 1920-3000, 3000-6000, 6000-10000, >=10000;
- conditional Timing MAE/P95, thresholds, and worst errors with actual
  vs projective vs unwrapped angle;
- conditional same-event eligibility (this is NOT by itself a switch
  detector false positive), matched event-pair precision as a
  descriptive diagnostic ONLY;
- replay verification that exact per-event oracle counts and floor
  original guard counts agree with frozen v1 reports.

A better angular estimate does not imply a correct next-note target
when multiple notes may fall within 10ms. Old HIGH-BPM Geometry may
give wrong signs despite previously passing its own guard. A formal
stateful event tracker is out of scope.

## Safety and provenance

Stage A FAIL stays frozen. No training, no Motor learning, no new
Student checkpoint, no alteration to v1 Geometry/Hybrid or previous
reports, no Validation or FINAL. All new report writes refuse overwrite.

The v2 capture input is still exactly 263D. The projective estimator
and threshold are audit-only, not a policy action or production
replacement. Train20 heldout has been examined repeatedly and is NOT
a clean final evaluation.

## Windows commands

~~~powershell
git status --short
git pull --ff-only

uv run pytest -q tests/test_prestige1_autoplay_projective_unwrap_audit.py
uv run python -u scripts/audit_prestige1_autoplay_projective_unwrap.py --config configs/training/prestige1.toml
~~~

Generated audit:

    diagnostics/prestige1/stage_timing_autoplay_projective_unwrapped_v2_train20.json

The audit intentionally replays all 20 expert sequences on Windows
to independently recapture v2 HUD and validate trajectory identity.
