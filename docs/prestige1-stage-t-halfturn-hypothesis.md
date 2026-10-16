# Prestige I Stage T — Is the 9x orbit step a planet-swap half-turn?

## Previous observations (NOT yet an explanation)

In the saved Train-heldout special/motion audit, 51/148 true note-switch
pairs were rejected by the **original** Geometry guard
inconsistent_orbit_motion. The observed cross-boundary orbit step magnitude
was 9.000x the Tile BPM-based expected step on average. The preceding
within-floor step was 1.000x, there were 51/51 apparent direction flips,
and 0/51 Tile/Real BPM changes.

The original Geometry checks changed *current floor slot features* BEFORE
its angular speed consistency guard. Thus each of those 51 switches must
have the **same 15D current floor slot** on the two adjacent HUD frames.
That does NOT imply the simulator's absolute current floor index was the
same. The moving window may shift to the next similarly-encoded tile.

Separately, floor coordinates in the encoder are translated relative to
the current floor, the orbit angle is based on the current floor's entry
angle, and the geometry path ignores TrackPosition/decorative transforms.
The orbit coordinates are NOT globally continuous across planet swaps.

## Fixed half-turn hypothesis

Compute the following from HUD[k-2], HUD[k-1], HUD[k], and nothing else:

- prior_step = wrap(theta[k-1] - theta[k-2])
- crossing_step = wrap(theta[k] - theta[k-1])
- corrected_step = wrap(crossing_step - pi)
- residual = wrap(corrected_step - prior_step)

The correction is only a hypothetical 180-degree **coordinate jump**,
not an approved clock/geometry estimator. In normal same-direction floor
motion, residual should approach zero if one step of orbit motion plus a
planet exchange accounts for the observation.

Separately test the **floor angle relationship**:
new-current entry angle minus old-current exit angle should be congruent
to 180 degrees if the current-floor progression uses the stock
next-entry = previous-exit + pi relation. That is not an oracle time
signal: entry/exit sin/cos are already in the visible 263D HUD.

Compare, using decoded **current Tile BPM**:

- crossing speed ratio (raw existing guard input)
- prior speed ratio
- halfturn-corrected crossing speed ratio

Report exact descriptive counts with absolute residuals <=0.1, <=0.5,
<=1.0 and <=5 degrees. No threshold is used for inference/adoption.
Report signed/absolute degree residuals, BPM changes and representative
examples with anchor/frame indices. Keep the original frozen Geometry
classification intact.

## Research protocol

- Train20 expert-driven HUD, chart-disjoint Train-heldout subset.
- HUD-only transformations on **every frame** first.
- Oracle event IDs, masks and exact timings are NEVER used by the
  transformation; they label switch cohorts only after computation.
- Separate original inconsistent_orbit_motion from original floor_changed.
- Replay-verify sample counts and original rejection reasons against
  saved frozen special/motion report, its config SHA and source SHA set,
  the checkpoint files and original Stage A FAIL.
- Write only a new JSON report, refuse overwrite; no changes to any
  prior checkpoint, model, Geometry guard, policy or training config.
- No Validation, FINAL or Motor/Fingering training.

## Windows commands

~~~powershell
git status --short
git pull --ff-only

uv run pytest -q tests/test_prestige1_autoplay_halfturn_audit.py
uv run python -u scripts/audit_prestige1_autoplay_halfturn.py --config configs/training/prestige1.toml
~~~

New report:
- diagnostics/prestige1/stage_timing_autoplay_halfturn_diagnostic_train20.json

**Interpretation caution:** Near-zero angular residual confirms the
*mathematical compatibility* of a 180-degree jump and preceding orbit
motion on this captured HUD stream. It does not on its own prove the
full physics mechanism or establish a safe new online predictor.
