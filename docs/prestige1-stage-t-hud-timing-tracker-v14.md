# Stage T — Causal Timing Tracker v1.4: SetSpeed / Pause visibility boundaries

## Scope

Independent read-only Timing research on 263D HUD v2 at 10ms control
sampling and 1ms simulation. This is **not a trained Student**, and the
Stage A v2 FAIL / may_run_stage_c=false gate remains unchanged.
No truth timestamp, floor index, planned action, future HUD or hidden
SetSpeed angleOffset / Pause duration enters the predictor. Evaluator
truth is accessed only after all forecasts and tracker identities exist.

The v1.4 module is **separate** from v1/v1.1/v1.2/v1.3. Those modules,
frozen Stage A/T artifacts, Motor, and evaluation conditions remain intact.

## Why prior naive expansion was unsafe

Allowing a future SetSpeed marker without limiting the *subsequent*
prediction initially increased real-chart recall, but caused 3 phantom
events in the opening countdown. A countdown floor cannot be treated as
a normal preceding floor. A visible empty behind-floor slot now blocks
speculative arrival into an upcoming SetSpeed or Pause there.

A more important adversarial result: a SetSpeed event with
`angleOffset=90` may cause the current HUD Tile BPM to represent the
floor's average speed, while the **next floor** runs at a different
effective BPM. Extrapolating that average beyond the current floor
created several wrong event sequences and false tracker events in the
unrestricted 160-case stress test. This was **FAIL**, and is not adopted.

## v1.4 inference contract

- **Approaching SetSpeed or Pause floor:** if the preceding current
  non-countdown floor is geometrically valid, predict only its next
  floor-entry event. Return `eligible_partial` with
  `covered_until_s` exactly at that boundary. Do not assume the next
  floor's speed or pause duration.
- **Inside SetSpeed floor:** use currently visible Tile BPM and geometry
  only until the current floor's exit. This respects the simulator's
  average-speed HUD behavior for mid-floor `angleOffset`; do not
  extrapolate past its exit. A combined current SetSpeed+Pause abstains.
- **Inside Pause floor:** while orbit is stationary, the remaining
  duration is unobservable: abstain. Resume forecasts *only if* consecutive
  10ms HUD observations have identical visible floor geometry and Tile
  BPM, and the signed angular displacement agrees with the visible BPM.
  If the phase/step is ambiguous, aliased above the conservative 90°
  threshold, or speed changed, abstain.
- **Future multi-planet, midspin, other unverified special actions:**
  abstain as before.
- Keep previous v1.3 full-turn signed-motion guards and local event-ID
  reconciliation. No synthesized completion time is invented.

A test creates two charts with different Pause durations that share
identical HUD prefixes, demonstrating that duration cannot be recovered
from those observations alone. A post-pause motion observation provides
new evidence and can then permit a forecast.

## Reproducible CPU evidence (fixed inspected charts)

| Evaluation | v1.3 | v1.4 | Unmatched v1.4 |
|---|---:|---:|---:|
| 命日 (Nerfed), first 65s, 890 targets | 859 | **887** | **0** |
| level-recovered.adofai, first 50s, 863 targets | 816 | **852** | **0** |

Counts are distinct elapsed local tracker events matching target timestamps
in the specified segment (50µs matching tolerance). They are NOT Motor
Hits or full-chart SAFE. The first chart has 3 target times not tracked:
first countdown/SetSpeed arrival, one untracked SetSpeed-floor transition,
and one high-BPM full-turn motion-alias case. They are deliberately not
fabricated.

- All old/new regression tests: **63/63 PASS** on CPU.
- New SetSpeed(angleOffset), normal speeds, Pause durations, Twirl
  combinations: **160 generated cases**, 18,114 HUD frames,
  14,968 eligible/partial predictions, 2,986 elapsed events;
  all 2,986 matched, 0 mismatches. An earlier *unsafe* implementation
  failed this matrix; only the corrected implementation passed.
- Re-ran original **252 geometric/Twirl cases** on v1.4:
  53,651 eligible forecasts, 10,912 correct elapsed events,
  no mismatch.

Matrix scoring treats events within 1 microsecond of the 10ms forecast
window boundary as boundary-indeterminate for event-count coverage,
and verifies emitted times at 50 microseconds. Testing is limited to
these fixed simulated chart families; no independent untouched FINAL
dataset or native ADOFAI parity claim is made.

## Commands

```shell
PYTHONPATH=src python -m pytest -q \
  tests/test_hud_timing_tracker_v1.py \
  tests/test_hud_timing_tracker_partial_v11.py \
  tests/test_hud_timing_tracker_current_speed_v12.py \
  tests/test_hud_timing_tracker_fullturn_v13.py \
  tests/test_hud_timing_tracker_transition_v14.py

PYTHONPATH=src python scripts/audit_hud_timing_tracker_transition_v14.py \
  PATH_TO_NERFED.adofai --end-s 65 \
  --output diagnostics/timing-tracker-v14-nerfed65s.json

PYTHONPATH=src python scripts/audit_hud_timing_tracker_transition_v14_matrix.py \
  --output diagnostics/timing-tracker-v14-special160.json
```

Audit files refuse overwrite. The exact checked outputs were generated
in the assistant's Linux CPU runtime; no Windows or CUDA run required.

## Remaining work

Keep the countdown floor timing separate from normal floors. Investigate
the one missed SetSpeed transition through tracker identity histories,
then investigate high-BPM direction alias observability separately.
Do not lift Stage A FAIL or train Fingering/Motor until the necessary
causal contracts and independent evaluation are ready.
