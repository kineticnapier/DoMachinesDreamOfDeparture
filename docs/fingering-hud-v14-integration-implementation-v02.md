# DMDOD — HUD-only Timing v1.4 × Fingering × 12-axis Motor

## Status

Runnable integration candidate, not yet a Windows-verified HUD-only PASS.
All code changes are on develop/v0.1-simulator and require the user's
local tests before success can be asserted.

Frozen Fingering checkpoint, original 8-key HUD263 contract, 10ms control,
1ms physical integration, Motor parameters and Timing v1.4 are unchanged.

## Added components

- planning/hud_v14_forecast_source_v02.py: causal v1.4 local forecasts,
  keeps unconfirmed tokens while the predictor abstains.
- planning/oracle_forecast_source_v02.py: isolated truth-aware comparator;
  output only contains local clock deadlines, no target ordinal.
- planning/timing_fingering_context_v02.py: truth-free forecast tokens to
  frozen 65D Fingering Network context.
- planning/contact_fingering_candidates_v02.py: frozen-order physical
  finger candidates and CenterFirst control policy.
- planning/timing_fingering_motor_policy_v02.py: shared selector,
  physical reservation and continuous 12-axis Motor commands.
- envs/contact_hud_chart_v02.py: contact renderer with *current displayed*
  Tile and Real BPM.
- scripts/run_hud_v14_fingering_motor_integration_v02.py: three-arm
  simulation, posthoc coverage analysis and explicit PASS/FAIL gates.
- tests/test_timing_fingering_motor_integration_v02.py: integration tests.

Previously added contact_timing_hud_v01 and
persistent_fingering_scheduler_v01 are reused.

## Three arms

1. Oracle Timing + frozen Fingering Network + persistent scheduler + Motor.
2. HUD v1.4 Timing + same frozen Network, scheduler and Motor.
3. HUD v1.4 Timing + CenterFirst, same scheduler and Motor.

Oracle truth is read only by the Oracle source constructor and posthoc
evaluator in the *runner*. The shared policy only accepts a
TimingFrameUpdate and the current physical 12-axis Motor, and does not
receive chart time, target ordinal, scorer Hit/Miss, or future fingering.

The HUD263 adapter uses a neutral first 24D and unchanged geometry
and unclipped v2 BPM layout. Neural Timing v3/GRU/MLP is not supported
by this temporary adapter. Timing v1.4 emits single-contact ordinary
events; unsupported MultiPlanet/midspin/Pause ambiguity abstains.

## Physical reservation semantics

Hit resolution never releases a physical axis. Likewise, simply passing
a predicted event due time must not abort a physical stroke that may
KeyDown several milliseconds afterward. Explicit cancellation is
different: it brakes and quarantines the axis until physical reset.
Unconfirmed events after abstention cannot create new reservations.
Armed times are frozen with the verified v1.4 0.75ms reconciliation
tolerance for incoming predictions, without retiming the physical intent.

## Declared constants

- Timing horizon 80ms, decision horizon 69ms.
- Motor normal lead 44ms, additional ordinary-finger lead max 25ms.
- Postdue positive-command tail 35ms.
- Newly observed events with under 44ms of preparation are reported as
  insufficient lead, not covered successes.
- No retraining, physical approximation or Validation20 tuning.

## Windows test and evaluation commands

Run from F:/dev/DoMachinesDreamOfDeparture after inspecting git status
and executing git pull --ff-only.

First run tests in one command:

    .venv\Scripts\python.exe -m pytest tests/test_timing_fingering_contract_v02.py tests/test_persistent_fingering_scheduler_v01.py tests/test_contact_timing_hud_v01.py tests/test_timing_fingering_motor_integration_v02.py -q

Then use new, create-only output folders, in order:

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --fixtures --output-dir artifacts/fingering_network_v02/hud-v14-ordinary-r1

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --special-fixtures --output-dir artifacts/fingering_network_v02/hud-v14-special-r1

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --train-chart "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai" --end-s 5 --output-dir artifacts/fingering_network_v02/hud-v14-train-smoke-r1

The ordinary fixtures use fixed 120, 240, and 480 BPM patterns; the
special fixtures cover SetSpeed, Twirl and Pause. No Validation20,
Qualification, Reserve or Final chart is used in these default runs.

Each result contains integration-summary.json. A declared FAIL yields
process exit code 1 after preserving its full report. Gate constraints
are not tuned on the validation set.

## Interpretation / limitations

- The posthoc supported-only score matches predicted local due times to
  truth targets, then uses the ordered scorer target Hit. It is not proof
  that the chosen finger caused the Hit.
- Full-chart coverage and miss counts are separately reported; unobserved
  targets are not called supported successes.
- Visible BPM comes from the current-floor renderer (like the preexisting
  eight-key HUD view). The policy cannot query segment targets.
- A lack of early enough v1.4 forecast is a genuine limitation; do not
  backfill the correct due from Oracle.
- The 16-contact scorer still has incomplete same-frame Multipress logic.
- Oracle arms may differ from the old standalone baseline due to the NEW
  shared persistent reservation scheduling.
- Python unit tests do not establish successful Windows HUD-only motor
  behavior: run the fixtures, inspect failures, and stop evaluation
  expansion if a gate fails.

Never report the integration as validated or the HUD-only system as
fully successful before actual fixture and Train smoke results exist.
