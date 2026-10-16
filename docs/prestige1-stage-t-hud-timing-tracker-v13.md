# Stage T — Causal Timing v1.3: visible full-turn observability

## Research question

The frozen v1.2 HUD tracker abstained on any floor whose visible entry and
exit angle coincide. In the 2-planet, non-Midspin **stock simulator**, such a
normal noninitial floor represents a full 360° rotation, not a zero-duration
arc. This is *partly* observable from the 263D HUD and recent orbit history.

## Experimental v1.3 behavior

This is a **separate module** from v1/v1.1/v1.2; those remain unchanged.

1. Forecast the arrival **into** a future full-turn floor as a safe prefix
   while the current normal-floor geometry is valid. End the forecast exactly
   at that arrival; don't speculate about travel through the new floor.
2. On a current normal full-turn floor, require an immediately preceding HUD
   sampled 10ms earlier, identical visible current and next floor slots, and
   unchanged observed Tile BPM. Check the signed orbit increment against
   +/- `π * TileBPM / 60 * 0.010`.
3. Accept a signed direction only when that 10ms increment is strictly below
   a conservative 90° alias threshold and agrees with the observed increment
   within `max(0.02 rad, 2.5% step)`. If two directions remain possible, or
   if the floor changes, aliasing, pause/midspin/multiplanet or unsupported
   future speed changes intervene, **abstain**.
4. With a unique direction and equal visible entry/exit angle, use a **2π**
   current-floor arc and the orbit's observed phase to predict the next
   visible event within the 30ms horizon. Event IDs are local to the tracker.

All inference uses causal HUD; chart target timestamps and floor ordinals
are accessed *only after* inference to score prediction quality. The
2π convention is a simulator contract, not a claim to have inferred an
unobservable absolute number of turns from sin/cos alone. Direction at high
BPM is not reliably recoverable from two 10ms snapshots: no guessed signs.

## CPU evidence, deterministic read-only research

- v1, v1.1, v1.2 and v1.3 regressions together: **42/42 PASS**.
- **命日 (Nerfed), first 65s**: v1.2 **829/890**, v1.3 **859/890**
  distinct matched elapsed events; **0** unmatched within segment. This is
  96.5% of this already-inspected chart, not total Student recall in use.
- **Separate real-chart fixture (`level-recovered.adofai`), first 50s**:
  v1.2 **796/863**, v1.3 **816/863**, **0** unmatched.
- **252 deterministic geometries** = 14 patterns × 9 BPM values
  (496, 992, 1984, 2380, 4000, 6000, 8000, 12000, 20000)
  × 2 Twirl conditions. 62,357 visible 10ms frames, **53,651 eligible**
  frame forecasts, **10,912 exact-matched** elapsed tracked events,
  **18,014** ID reuses; no mismatched predictions within their declared
  coverage, and no false elapsed events. Physics steps are 1ms.
- One forecast-vs-label discrepancy in the *first* raw matrix scorer was
  an event approximately **0.13 microseconds** after the exact 30ms forecast
  boundary, with zero emitted events. The final matrix treats events in a
  **1-microsecond band around either window boundary as indeterminate** for
  event-presence recall; all emitted predictions are still checked against
  true chart event times (<=50µs tolerance). This boundary policy is stated
  explicitly rather than silently counting the edge case as either success
  or failure. No other scored discrepancies appeared.

## Remaining real-chart misses (posthoc)

31 of 890 chart events remain untracked: 27 future special-floor transitions,
2 current special-floor transitions, 1 high-BPM rotation-alias case,
1 tracker miss on an otherwise eligible HUD. These are correlated statuses,
not necessarily individual causal explanations. The remaining 31 need
separate study; don't generalize 360° handling into unsupported SetSpeed /
Pause handling.

## Run on CPU

```shell
PYTHONPATH=src python -m pytest -q \
  tests/test_hud_timing_tracker_v1.py \
  tests/test_hud_timing_tracker_partial_v11.py \
  tests/test_hud_timing_tracker_current_speed_v12.py \
  tests/test_hud_timing_tracker_fullturn_v13.py

PYTHONPATH=src python scripts/audit_hud_timing_tracker_fullturn_v13.py \
  PATH_TO_NERFED.adofai --end-s 65 \
  --output diagnostics/timing-v13-nerfed65s.json

PYTHONPATH=src python scripts/audit_hud_timing_tracker_fullturn_v13_matrix.py \
  --output diagnostics/timing-v13-252-chart-matrix.json
```

All reports refuse to overwrite. No prior frozen code/Stage A gate,
checkpoints, MaleCNS4096, Student model, Motor/Scorer, Validation, FINAL or
learning configuration changed. Physical hit correctness, 16-contact
execution and held-out generalization remain out of scope.
