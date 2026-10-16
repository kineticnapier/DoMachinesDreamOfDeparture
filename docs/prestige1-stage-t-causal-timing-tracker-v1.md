# Prestige I — Causal HUD Timing Tracker v1 (experimental)

**Independent Timing research, not a trained Student or an adoption of Stage A.**

- Causal input: 263D HUD v2, 10ms frames, no oracle timestamps, target IDs, hidden direction or future chart information.
- Forecast window: 30ms within visible ordinary geometry, explicitly abstaining on unsupported markers, full turns, unknown direction and insufficient lookahead.
- Stateful event IDs: monotonic local identities reconciled across overlapping forecasts (0.75ms matching tolerance).
- Abstentions create no events. Carry-over plans become unconfirmed. An elapsed prediction is **not a confirmed physical Hit**.
- Exact current floor exit events are skipped as already elapsed for the 30ms horizon; the 10ms legacy forecast boundary behavior is preserved.
- The previously frozen Stage A v2 remains FAIL. No Motor/Fingering/Student training or new checkpoint.

## Verified CPU results

14 direct CPU regression tests passed. Separate read-only fixed experiments:

- 26 fixture BPM/direction combinations (600 to 20000 BPM): no incorrect elapsed events.
- 72 varied-angle and Twirl fixtures, 7734 observed frames: 4410 eligible forecasts, 0 wrong eligible event sequences.
- The same 72 cases statefully: 3746 elapsed events matched posthoc, 5403 ID reuses, 0 wrong elapsed events.
- 命日 (Nerfed) first 65 seconds: 6505 HUD frames, 890 truth events, 605 matched elapsed predictions, 0 mismatches, 285 events not tracked. One prediction beyond 65s is excluded from this deliberately clipped score.

These are **conditional-coverage** reports on inspected charts; not an unbiased unseen-chart test or complete prediction, and not 12-axis/16-contact Student integration.

## CPU commands

Run: uv run pytest -q tests/test_hud_timing_tracker_v1.py

Run: uv run python -u scripts/audit_hud_timing_tracker_v1.py PATH_TO_NERFED.adofai --end-s 65 --output diagnostics/timing-tracker-v1-nerfed-65s.json

The audit refuses to overwrite previous outputs.

## Next

Explore supported visible-special-event transitions with explicit abstention.
Then version the Timing/Fingering interface for the 12-axis/16-contact system.
Leave old checkpoints, Validation and FINAL untouched.
