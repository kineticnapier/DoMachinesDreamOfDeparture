# DMDOD Phase 8 — Lifecycle v1.5 integration and regression audit

## Adoption decision

**`INTEGRATION_READY`**

`QuantizedLifecycleCausalTimingTracker` is now an explicit option of
`HUDV3CompositeForecastSource`.  The default remains lifecycle `v1.4`; callers
must pass `lifecycle_version="v1.5"` to opt in.  Any other value raises
`ValueError`.

This result means the Forecast Source, adapter, and Motor receiving contract
are ready for a later controlled integration.  It does **not** establish an
actual Motor Hit/Miss improvement: no real-chart Motor evaluation was run.

## API

~~~python
from dmdod.planning.hud_v3_composite_forecast_source_v30 import (
    HUDV3CompositeForecastSource,
    LIFECYCLE_V14,
    LIFECYCLE_V15,
)

legacy = HUDV3CompositeForecastSource(lifecycle_version=LIFECYCLE_V14)
improved = HUDV3CompositeForecastSource(lifecycle_version=LIFECYCLE_V15)
~~~

Omitting `lifecycle_version` is exactly the legacy v1.4 path.  Both paths use
the same HUD v1.4 forecast geometry, Countdown estimator, 80 ms horizon,
adapter, integer ID mapping, due interval representation, and uncertainty.

## Fixed regression audit

The audit replays identical current/past HUD observations into two independent
Forecast Sources before target truth is read:

1. Phase 7 fixed Train four, first 12 seconds;
2. Train bool, first 5 seconds.

Validation20 and FINAL were not used.  The final result is
`reports/hud-timing-phase8-lifecycle-integration-20261019-r2.json`.

### Phase 7 reproduction

| Metric | v1.4 | v1.5 |
| --- | ---: | ---: |
| Events | 191 | 191 |
| Clean lifecycle events | 167 | 189 |
| Clean lifecycle coverage | 87.43% | 98.95% |
| Lifecycle inconsistencies | 23 | 1 |
| False-positive IDs | 0 | 0 |

This exactly reproduces the declared Phase 7 clean-event improvement.

### Phase 7 plus Train bool

| Metric | v1.4 | v1.5 |
| --- | ---: | ---: |
| Events | 200 | 200 |
| Clean lifecycle events | 176 | 198 |
| Clean lifecycle coverage | 88.00% | 99.00% |
| Lifecycle inconsistencies | 23 | 1 |
| Justified abstentions | 1 | 1 |
| False-positive IDs | 0 | 0 |
| New / reused IDs | 201 / 1,523 | 201 / 1,523 |
| Canceled / elapsed | 24 / 176 | 2 / 198 |
| Existing success regressions | 0 | 0 |

The remaining lifecycle classification is the declared Triple Cross first
event `source_conflict`.  The remaining abstention is the declared event after
a Midspin.  Phase 8 does not alter either case.

Reported Timing MAE is 0.102686 ms for v1.4 and 0.091277 ms for v1.5; mean
forecast lead is 65.163 ms and 65.701 ms respectively.  These differences are
**not timing-estimation improvements**.  v1.5 moves 22 already-accurate events
from invalid lifecycle membership into the clean metric population, changing
the set over which MAE and lead are averaged.  Per-frame due values are equal.

## Forecast invariance

Across all five fixed cases:

- forecast due differences: 0;
- uncertainty differences: 0;
- Countdown due-interval differences: 0;
- pending integer-ID set differences: 0;
- adapter ID-contract violations: 0;
- existing successful-event regressions: 0.

The adapter continues to map each Composite string ID once to a monotonically
increasing integer ID.  `new` and `reused` partition pending IDs; `canceled`
and `elapsed` are disjoint and retire only live IDs.

## Motor receiving contract

A synthetic contract test creates and arms a physical reservation, then sends
the v1.5-style `elapsed` update 0.0002 ms before its continuous due value at the
quantized 10 ms frame boundary.

Observed result:

- reservation remains `ARMED_IN_FLIGHT`;
- `retired_reason` becomes `elapsed`;
- no physical KeyDown is fabricated;
- the occupied axis is not made available for reuse;
- the physical stroke remains governed by actual KeyDown/reset and the
  existing due+tail safety window.

Thus `elapsed` means only that the forecast clock/lifecycle passed.  It does
not mean Hit, successful KeyDown, or permission to advance physical state.
Abstention tests also confirm that an unconfirmed future event is not finalized
early.

## Gates

| Gate | Result |
| --- | --- |
| Phase 7 improvement reproduced | PASS |
| Explicit legacy v1.4 selection | PASS |
| Default remains v1.4 | PASS |
| Forecast due unchanged | PASS |
| Uncertainty unchanged | PASS |
| Countdown interval unchanged | PASS |
| Adapter event-ID violations | PASS, 0 |
| False positives increased | PASS, no |
| Existing success regressions | PASS, 0 |
| Motor receiving contract | PASS |
| Countdown first prediction regression | PASS, none |

## Reproduction

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_hud_timing_phase8_lifecycle_integration.py `
  tests/test_hud_v3_composite_forecast_source_v30.py `
  tests/test_hud_timing_tracker_lifecycle_v15.py `
  tests/test_hud_timing_composite_v20.py `
  tests/test_persistent_fingering_scheduler_v01.py `
  tests/test_timing_fingering_motor_integration_v02.py `
  tests/test_countdown_phase_tracker_v01.py `
  tests/test_countdown_first_event_v01.py

.\.venv\Scripts\python.exe -u `
  scripts/audit_hud_timing_phase8_lifecycle_integration_v01.py `
  --output reports/hud-timing-phase8-lifecycle-integration-YYYYMMDD-rN.json
~~~

The verified test run completed with **99 passed**.  No learning, checkpoint
update, hyperparameter search, Validation20/FINAL access, Fingering change,
Motor physics change, Scorer change, or real-chart Motor run was performed.
