# DMDOD Phase 7 — HUD Timing generalization

## Outcome

Phase 7-A fixed four Train charts before evaluation and audited their first
12 episode seconds.  The final Phase 7-C change is deliberately limited to
event lifecycle reconciliation at a quantized 10 ms control-frame boundary.
It does not change forecast due times, Countdown estimation, visible-BPM
interpretation, geometry, Motor, Fingering, Scorer, or any checkpoint.

The improved composite raised clean event coverage from **167/191 (87.43%)**
to **189/191 (98.95%)**, reduced lifecycle failures from **23 to 1**, and kept
false-positive IDs at **0**.  Aggregate midpoint Timing MAE changed from
0.081272 ms to 0.071812 ms and mean first forecast lead from 64.923 ms to
65.514 ms.  This is not a 191/191 claim: one Midspin-adjacent event remains a
documented abstention and one first-event Countdown/v1.4 source conflict remains.

## Fixed audit set

These charts were not used by the Countdown Phase 2 Train-bool gate.  No
Validation20, Qualification20, Reserve20, or Final chart was opened.

| Train chart | SHA256 | events <=12s | audited constructs |
| --- | --- | ---: | --- |
| `02334 - Time Leaper - Jipper.adofai` | `f4f71701065c986da6fd999cbc0a659d8fe47434793b5898e099ec2df83a5f70` | 21 | Countdown, ordinary rotation, 2 SetSpeed |
| `02521 - Triple Cross - Milkyway.adofai` | `5571481d5a0eee9dee9fe70c81702b3bdf6dfd0491a6904c80d19e22610d86d3` | 65 | Countdown, 2 SetSpeed, 24 Twirl, 1 Midspin |
| `07322 - Runaway - smy258.adofai` | `2e67ce666b72d72e61f353be22467eef60a10a6cbc935adc73baf833d3cdb27e` | 97 | Countdown, 2 SetSpeed, 28 Twirl |
| `07827 - 24eeev0-$ - hotduck _ COMMON.adofai` | `3e93ec4afae58c516b367a45e1e504cf31628afd205e9f87b0f7f8de465782b3` | 8 | Countdown, 9.3 s Pause |

Policy collection uses only the current/past HUD263 and the observed Countdown
suffix.  Exact target times are materialized only after a chart's complete HUD
stream has been collected, then used for association and metrics.

## Phase 7-A classification

The five categories are mutually exclusive.  Coverage below means category 1
with a continuous, non-canceled ID; it intentionally does not count an accurate
forecast whose lifecycle is invalid.

| System | 1 success | 2 observable failure | 3 justified abstention | 4 unverified | 5 lifecycle | clean coverage | FP IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw Timing v1.4 | 165 | 0 | 4 | 0 | 22 | 86.39% | 17 |
| HUD v3 Composite | 167 | 0 | 1 | 0 | 23 | 87.43% | 0 |
| Composite + lifecycle v1.5 | 189 | 0 | 1 | 0 | 1 | 98.95% | 0 |

The raw-v1.4 false IDs are Countdown orbit cycles.  Composite suppresses them,
so they are not playable candidates in either the baseline or improved output.
No category-2 timing-regression case was found in this fixed prefix.  The
highest-count repairable failure was category 5 and was therefore selected
instead of inventing a category-2 optimization target.

Concrete failures:

- Triple Cross event 16: truth `4.050000111 s`, retained ID `v14:33`, predicted
  due approximately `4.050000103 s`.  At local frame `4.050 s`, old logic
  canceled the ID because due was about 0.000103 ms later than the frame clock,
  while the new eligible HUD forecast had already advanced past the event.
- The same boundary pattern affected 15 Triple Cross and 7 Runaway events.
- Triple Cross event 23 at `5.125000148 s` follows a zero-duration Midspin.
  Statuses are `eligible_partial`, `special_event`, and `special_future_floor`;
  it remains category 3.  Midspin traversal was investigated but intentionally
  not included because Phase 7-C permits only one selected cause.
- Triple Cross event 0 at `1.200000042 s` still exposes both `countdown:0` and
  `v14:17`, with one `source_conflict`; it remains category 5.
- The long Pause produces 930 `pause_motion_unobservable` frames.  The system
  does not guess the hidden Pause duration.  Predictions resume from visible
  motion, which is the intended causal behavior.

## Phase 7-B selection

Selected cause: **near-boundary accurate tokens were canceled instead of
elapsed because the continuous due clock and 10 ms HUD clock differ by
sub-frame floating-point amounts**.

The correction is mathematically bounded by the existing 0.75 ms identity
matching tolerance.  A token can be reconciled as elapsed only when:

1. a new frame yields an eligible HUD forecast;
2. the old token is absent from that new forecast; and
3. `due <= now + match_tolerance`.

An invalid/stale/special-event abstention cannot trigger early elapsed.  The
rule uses no target truth and neither moves a due time nor creates an event.
Default `CausalTimingTracker` behavior remains unchanged; the rule is opt-in
through `QuantizedLifecycleCausalTimingTracker`.

## Phase 7-C gates

| Gate | Result |
| --- | --- |
| Default v1.4 behavior/tests preserved | PASS |
| Due times unchanged by lifecycle rule | PASS |
| Existing ordinary success cases regress | PASS (0 regressions) |
| Composite false-positive IDs | PASS (0 -> 0) |
| Lifecycle failures | PASS (23 -> 1) |
| Clean coverage | PASS (87.43% -> 98.95%) |
| Abstention cannot expire a future token early | PASS |
| Validation20/FINAL/training/checkpoint/Motor run | NOT RUN, by design |

## Reproduction

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_audit_timing_generalization_phase7_v01.py `
  tests/test_hud_timing_tracker_lifecycle_v15.py `
  tests/test_hud_timing_tracker_transition_v14.py `
  tests/test_hud_timing_composite_v20.py `
  tests/test_countdown_phase_tracker_v01.py `
  tests/test_countdown_first_event_v01.py `
  tests/test_countdown_hud_v3.py

.\.venv\Scripts\python.exe -u `
  scripts/audit_timing_generalization_phase7_v01.py `
  --output reports/hud-timing-generalization-phase7-audit-YYYYMMDD-rN.json
~~~

The fixed r6 result is
`reports/hud-timing-generalization-phase7-audit-20261019-r6.json`.

## Remaining work

The next bounded investigations should treat the two remaining failures
separately: reconcile Countdown/v1.4 source identity at the first playable
event, and prove a causal visible-geometry rule for zero-duration Midspin nodes.
Neither should be combined with this lifecycle change or selected using
Validation20.
