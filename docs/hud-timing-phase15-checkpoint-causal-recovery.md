# DMDOD Phase 15 — Future Checkpoint Causal Timing Recovery

## Status

**CHECKPOINT_CAUSAL_RECOVERY_PASS (full fixed audit, user-run stdout verified as supplied)**

HUD263 / HUD278 and the frozen Timing v1.4 geometry are unchanged. No image
recognition, Hold mesh sensor, Fingering, Motor, Scorer, checkpoint or
learning changes were made. The H7–H9 image research stays frozen.

The only non-clean event in Phase 14 fixed Train16 is ヒアソビ Event 54
(Floor 55; chart SHA256
`baaf6bff3cd012d8b0e4f123f58868c1cd4f262a35394e32eb2ebcba898caab9`).
That floor has **SetSpeed ×0.5 + Checkpoint tileOffset -1**.

## Phase 15 full Train16 + reference5 audit (reported user run)

**Final reported verdict: `CHECKPOINT_CAUSAL_RECOVERY_PASS`**

The fixed full audit was run in the user's Windows environment and its stdout
aggregate was supplied to the project chat. The source stdout has been
preserved as a structured, provenance-labelled report in
`reports/hud-timing-phase15-checkpoint-user-run-summary-20261020-r1.json`.
The full per-event audit JSON was **not** supplied, so these run-level values
are attributed to the user-run stdout, not an independent local replay.

| Fixed set | Baseline (Checkpoint OFF) | Candidate (Checkpoint ON) |
|---|---:|---:|
| Train16 first 12 seconds | **1067/1068** (99.906%) | **1068/1068** (100%) |
| Reference5 | **200/200** | **200/200** |
| Train16 category 2/3/4/5 | 0/1/0/0 | **0/0/0/0** |
| False-positive event IDs | 0 | **0** |
| Existing Clean event regressions | — | **0** |
| Train16 conditional MAE | 0.019592905ms | 0.019574570ms |

On ヒアソビ Event 54, baseline category 3
(`special_future_floor`) changed to candidate category 1 with a unique
matching event ID `54` and `eligible_partial` status. The first forecast
lead is **71.79107ms** (44ms Motor readiness threshold met), first forecast
error **0.000011ms**, wrong candidate count **0**.

The report's ten checks are true, including preservation of Phase 12 fixed
13 events. All reference5 metrics, including conditional MAE, are unchanged.
Train16 mean MAE changed over a **different clean event population**:
this is not an estimator-accuracy improvement on a shared cohort.

The fixed audit is **Timing/Lifecycle**, not physical Motor A/B, and is not a
validation of full-chart Hold behavior or a general 100% success guarantee.
The special recoveries are still opt-in; the frozen future SetSpeed safety
gate remains in force.

Test caveat: the stdout contains the full audit verdict and checks but does
not include a separate pytest command/result for the merged Phase15 suite.
No test-suite PASS beyond earlier isolated tests is inferred here.

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

This isolated local probe alone did not establish Lifecycle coverage, but the later user-run full fixed audit reported 1068/1068 clean, no false-positive IDs and no existing success regressions (see summary above).

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
previous uploaded source ZIP. The full fixed audit was subsequently run on the
user's Windows system (aggregate stdout supplied); separate post-merge pytest
suite results have not been provided.

## Reproduction commands and accepted checks

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

The full fixed Train16 + reference5 audit was subsequently completed by the
user **without** `--target-only`; use a new output file for any independent rerun. Do not run training,
Validation20, FINAL, or real Motor. Outputs are exclusive-create, never
overwrite existing evidence.

**Acceptance criteria checked against the supplied stdout**: Train16
1067/1068→1068/1068, reference5 200/200 unchanged, original 1067 success
events without regression, False Positive ID 0, Phase12 fixed13 maintained
13/13 and 9/13 Motor-ready, and target Event 54 lead ≥44ms. All criteria passed in the supplied aggregate. If a later independent run
fails, disable `checkpoint_recovery` and investigate the discrepancy.

Do not equate any Timing/Lifecycle success with physical Motor Hit rate.
