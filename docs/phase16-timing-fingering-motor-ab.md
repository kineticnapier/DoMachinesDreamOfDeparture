# DMDOD Phase 16 — HUD Timing v1.5 + Special Recovery → Fingering → Motor A/B

## Status

**RUNNER_COMMITTED / PHASE16_LATEST_SOURCE_MOTOR_AUDIT_PENDING**

This is the next ordinary-press physical-integration stage after fixed
Phase 15 Timing/Lifecycle coverage reached 1068/1068 on frozen Train16.
**The 1068/1068 result is not a Motor Hit score** and cannot license a
physical-performance claim.

No image recognition, new Hold mesh sensing, new Fingering/Motor policy,
checkpoint training, Validation20, Reserve20 or FINAL will be included.

## Controlled comparison

Both arms run independent complete simulations with the same:

- SHA256-pinned existing Fingering Network checkpoint
  `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`;
- `TimingFingeringMotorPolicy`, selector `network`, style alpha 0.5;
- physical 12-axis/16-contact Motor, scorer, 10ms controller, 1ms physics;
- 69ms decision and 44ms nominal preparation rules;
- exact same SHA256-pinned Train chart / horizon.

Only the Timing opt-in flags differ:

| Arm | Lifecycle | Midspin | Twirl | Checkpoint |
| --- | --- | --- | --- | --- |
| `v15_recovery_off` | v1.5 | OFF | OFF | OFF |
| `v15_midspin_twirl_checkpoint_on` | v1.5 | ON | ON | ON |

The future SetSpeed/Pause safety gates stay unchanged; the Hold video/pixel
research H7–H9 remains frozen. The evaluator uses target truth only in the
posthoc `target_outcomes_posthoc` join after both complete replays.

## Scope and reproducibility

Existing Phase 9 Train bool first 5 seconds (9 press targets) is the frozen
initial Motor baseline. In the assistant's existing-source **CPU** replay,
v1.4 and v1.5 each reproduced 8/9 Hit, 1 Miss, 0 TooEarly, 0 Overload,
8 physical KeyDown and timing MAE 2.625440ms. The second target was omitted
because only insufficient observable lead was available; this is not
a failed Finger choice.

The assistant's local ZIP predates the newest Phase 15 source, so the
newest full-flag A/B is **NOT locally verified**. Do not treat the legacy
8/9 replay as a measured result for the candidate arm.

The new runner may run:

1. `bool` 0–5s **default smoke** (fixed SHA256 in Phase11 REFERENCE).
2. Any explicitly named **frozen Train16** chart for 0–12s (verifies the
   entire deterministic Phase11 manifest and frozen Train80 hashes).
3. All fixed Train16, each in two independent Motor replays. There is no
   chart filtering or metric-driven reselection.

Existing 5 reference charts and Phase15 timing-only 1068/1068 are kept as
independent baselines; Phase16 does not overwrite/relabel them.

## Output and safety gates

The Phase16 report records every target's posthoc physical Hit/Miss,
physical/logical KeyDown, Fingering decisions, insufficient-lead counts,
reservation collisions, in-flight cancellations, source conflicts,
frame count, and Timing status counts for each arm.

The gate `MOTOR_AB_NONREGRESSION_PASS` requires:
- target population unchanged;
- **no previously Hit event lost**;
- no increase in TooEarly, Overload state/counter, reservation collisions, or in-flight
  cancellations;
- zero source conflicts;
- zero lifecycle ID contract violations or physical reservation violations.

A pass does **not** prove that recovered Timing predictions caused improved
physical performance or that all 16 charts are playable. An event gaining
a new Hit is recorded separately from a previously Hit event being lost.
Differences caused by the Motor's discrete stepping require posthoc trace
analysis; do not automatically retune any timing constant.

## Commands

Run in the current Windows repository root, after
`git fetch`, `git status` and `git pull --ff-only` if safe. Preserve the
user's uncommitted `configs/training/prestige1.toml` and all untracked
artifacts; never reset, clean or force checkout.

Focused unit tests:

```powershell
uv run python -m pytest -q `
  tests/test_run_hud_timing_phase16_motor_ab_v01.py `
  tests/test_run_countdown_hud_v3_motor_phase3.py `
  tests/test_run_hud_timing_phase9_lifecycle_motor_ab.py `
  tests/test_hud_timing_tracker_checkpoint_v18.py `
  tests/test_hud_timing_tracker_twirl_v17.py `
  tests/test_hud_timing_tracker_midspin_v16.py `
  --basetemp artifacts/pytest-phase16-v01
```

Initial 5-second physical smoke:

```powershell
uv run python -u scripts/run_hud_timing_phase16_motor_ab_v01.py `
  --output reports/phase16-bool5-motor-ab-20261020-r1.json
```

First-twelve-seconds single-chart test, including Phase14 Twirl targets:

```powershell
uv run python -u scripts/run_hud_timing_phase16_motor_ab_v01.py `
  --chart-name "01645 - First town of this Journey remix - Irin.adofai" `
  --output reports/phase16-firsttown12-motor-ab-20261020-r1.json
```

If these runs are safe, process all precommitted Train16 charts:

```powershell
uv run python -u scripts/run_hud_timing_phase16_motor_ab_v01.py `
  --all-unseen `
  --output reports/phase16-train16-motor-ab-20261020-r1.json
```

The result writer uses exclusive creation and does not overwrite.
The all-unseen run can take longer than the Timing-only audit because
the frozen Fingering Network and physical Motor execute on every control step.
If a safety gate fails, retain the report, investigate causal traces,
and do not enable the candidate by default.

## Implemented changes and limits

- `scripts/run_countdown_hud_v3_motor_phase3.py`: optional three Timing
  flags; defaults unchanged, and non-v3 arms reject them.
- `scripts/run_hud_timing_phase16_motor_ab_v01.py`: frozen selection,
  independent A/B replays, compact physical results and fail-closed checks.
- `tests/test_run_hud_timing_phase16_motor_ab_v01.py`: deterministic
  input/provenance/contract tests.

No production Fingering or Motor implementation was changed.
**Merged newest-source test runs remain pending.** The older-source local
execution cannot establish candidate readiness.
