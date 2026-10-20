# DMDOD Phase 15 — Future Checkpoint Causal Timing Recovery

## Status

**CHECKPOINT_LOCAL_GEOMETRY_PROBE_PASS / LIFECYCLE_FULL_AUDIT_PENDING**

HUD263 / HUD278 and the frozen Timing v1.4 geometry are unchanged. No image
recognition, Hold mesh sensor, Fingering, Motor, Scorer, checkpoint or
learning changes were made. The H7–H9 image research stays frozen.

The only non-clean event in Phase 14 fixed Train16 is ヒアソビ Event 54
(Floor 55; chart SHA256
`baaf6bff3cd012d8b0e4f123f58868c1cd4f262a35394e32eb2ebcba898caab9`).
That floor has **SetSpeed ×0.5 + Checkpoint tileOffset -1**.

## Stock Checkpoint control-flow evidence

Audited `ffxCheckpoint.StartEffect` is triggered on a hit, updates checkpoint
state, can revive players, and triggers visual/audio effects; unlike Pause
or SetSpeed, it does **not** add a duration to the chart-time map.
`scnGame` attaches the effect component to the floor. This conclusion is
**code-path parity**, not a measured Unity gameplay judgment.

A SetSpeed on the **destination floor** does not change its incoming arrival
time. The *previous* observed speed is 268 BPM; the destination floor's
resolved BPM is 134.0. Prediction beyond that destination still requires a
fresh speed observation (Phase 13 is unchanged).

## Isolated HUD-only timing probe

A local copy of the prior repository ZIP and the SHA-pinned Train80 chart was
executed with the frozen v1.4 geometry. The calculation-only HUD copy cleared
the **future** Checkpoint marker on the eligible ordinary two-planet
floor while retaining its visible SetSpeed marker.

- Target truth (posthoc only): `11.641791069972934s`
- First safe candidate: step 1157 (`11.570s`)
- Predicted due: `11.641791080586330s`
- Error: approximately **0.0000106ms**
- First target-relative forecast lead: **71.791ms** (>44ms)
- Next seven 10ms control frames produced the same target due.
- Candidate returned `eligible_partial`, no post-SetSpeed event.
- Unmodified v1.4 returned `special_future_floor` for these eight frames.

This does **not** yet establish recovered Lifecycle IDs, 1068/1068 coverage,
zero false-positive IDs, or no regression across all fixed charts.

## Opt-in code

`HUDV3CompositeForecastSource` adds:

```python
HUDV3CompositeForecastSource(
    lifecycle_version="v1.5",
    midspin_recovery=True,
    twirl_recovery=True,
    checkpoint_recovery=True,  # default False
)
```

- Future Checkpoint floor must be visible, ordinary, and two-planet.
- Permitted marker set: Checkpoint, optionally SetSpeed.
- Disallowed Twirl, Pause, MultiPlanet, SetFloorIcon, Midspin combinations
  remain blocked.
- Calculation-only marker normalization cannot affect the displayed HUD.
- Predicted windows are explicitly truncated at that Checkpoint event.
- No next-floor speed is inferred before the SetSpeed marker is entered.
- All three special recovery flags are independent and remain disabled by
  default; v1.4 lifecycle combinations reject them.

Files:
`src/dmdod/planning/hud_timing_tracker_checkpoint_v18.py`,
`tests/test_hud_timing_tracker_checkpoint_v18.py`,
`scripts/audit_hud_timing_phase15_checkpoint_v01.py`,
`tests/test_audit_hud_timing_phase15_checkpoint_v01.py`.

Isolated Checkpoint normalization tests: 11 passed locally against the
previous uploaded source ZIP. Full post-merge Lifecycle/Composite tests and
the fixed full audit have **not yet run**.

## Required tests / acceptance gate

First (local Windows project root):

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_hud_timing_tracker_checkpoint_v18.py `
  tests/test_audit_hud_timing_phase15_checkpoint_v01.py `
  tests/test_hud_timing_tracker_twirl_v17.py `
  tests/test_hud_timing_tracker_midspin_v16.py `
  tests/test_hud_v3_composite_forecast_source_v30.py `
  --basetemp artifacts/pytest-phase15-first-r1
```

Target-only audit:

```powershell
.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase15_checkpoint_v01.py `
  --manifest reports/hud-timing-phase11-selection-20261019-r1.json `
  --output reports/hud-timing-phase15-target-20261020-r1.json `
  --target-only
```

If both pass, run the fixed Train16 + reference5 audit **without**
`--target-only` and with a new output file. Do not run training,
Validation20, FINAL, or real Motor. Outputs are exclusive-create, never
overwrite existing evidence.

**Acceptance criteria before declaring Phase 15 success**: Train16
1067/1068→1068/1068, reference5 200/200 unchanged, original 1067 success
events without regression, False Positive ID 0, Phase12 fixed13 maintained
13/13 and 9/13 Motor-ready, and target Event 54 lead ≥44ms. If any fail,
disable `checkpoint_recovery` and report `CHECKPOINT_INTEGRATION_BLOCKED`.

Do not equate any Timing/Lifecycle success with physical Motor Hit rate.
