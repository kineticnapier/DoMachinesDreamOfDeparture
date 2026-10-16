# DMDOD HUD BPM encoding v2 — no clipping

## Problem

The historical v1 263D visible HUD uses two log2 BPM features
(Tile BPM and Real BPM) with a clamp to [-4,4]:

    clamp(log2(BPM / 120), -4, 4)

As a result, all BPM >= 1920 map to 4.0, including 2380 and 4000.
This produced a misleading Geometry time conversion in the frozen RP1
Train-fit replay: 30 errors >1ms all occurred on Train anchor 01,
with HUD Tile BPM saturated at 1920 while observable orbit speed
corresponded to 2380 BPM (ratio = 2380/1920).

The clipping also hides a high-tempo phase-aliasing hazard: original
projective Geometry's alias-risk condition uses the saturated HUD BPM
and can fail to see true speeds above the intended safety range.

## New default v2

Both 2-key and N-key HUD encoders now encode **finite positive** Tile
and Real BPM as:

    log2(BPM) - log2(120)

There are **NO upper or lower BPM clamps**. Zero, negative, NaN and
infinite BPM fail loudly with ValueError, not silently replaced with a
fabricated small positive BPM. Subtracting logs avoids underflow for
positive subnormal inputs.

The output layout and dimensionality are unchanged: 8K = 263D.
High-BPM values differ:
- 120 BPM: 0
- 1920 BPM: 4
- 2380 BPM: ~4.309855263
- 4000 BPM: ~5.058893689
- 8000 BPM: ~6.058893689

The names HUD_FEATURE_VERSION and N_KEY_HUD_FEATURE_VERSION now identify
the v2-unclipped encoding. **Old checkpoints trained with v1 values are
not input-distribution compatible with v2 values**, even though tensor
shapes are still 263D. Never feed v2 data into an old checkpoint
silently; retraining/independent assessment is required for new models.

## Historical artifact replay is explicitly pinned

For frozen Stage A and Stage T reproducibility, the v1 codec remains
available only as an explicitly selected LEGACY compatibility path
via legacy_bpm_encoding=True on HUD encoder functions.

Prestige I frozen collectors and Stage T Autoplay captures explicitly
request legacy_bpm_encoding=True so old report and checkpoint hash
verification sees its original inputs, even if BPM exceeds 1920.
The Stage C historical alignment capture also remains pinned to v1.

This historical compatibility does not authorize clipping **new**
data, new observations or future model training. The default
new encoder is v2 without any BPM clipping.

## Temporal sampling still matters

With 10ms control frames, the RP1 double-angle representation yields
a signed angular step only modulo 180 degrees. The nominal unique
directional interval is (-90,+90) degrees per frame, which under the
stock angular speed formula corresponds to Tile BPM <3000. E.g. the
4000-BPM physical advance is +120 degrees/frame but appears as -60
degrees through RP1 alone.

Removing the BPM clamp restores the information needed to potentially
unwrap these aliases using the separately observed BPM, but does not
by itself fix the existing frozen Geometry/Hybrid or license unsafe
Motor action. No estimator or neural checkpoint is adopted in this
change.

## Windows checks

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_hud_bpm_unclipped.py
uv run pytest -q tests/test_real_chart_hud_features.py tests/test_prestige1_autoplay_projective_orbit_audit.py tests/test_prestige1_stage_a_v2.py
~~~

All existing Stage A FAIL, Motor training prohibition, Validation/FINAL
exclusion and historical Stage T report hashes remain unchanged.
