# Stage T — HUD Timing Tracker v1.1: Twirl safe-prefix experiments

Separate research variant of the conservative v1. The *incoming* Twirl
note is forecast using current visible geometry, but the forecast terminates
at that event and is marked eligible_partial with an explicit covered_until_s.
Event identity matching and cancellation are limited to this verified prefix.
No prediction is permitted beyond the new Twirl until a new HUD arrives.

IMPORTANT: An attempt to permit all future special markers was UNSAFE on
the real chart: it increased event coverage but introduced three incorrect
event predictions near SetSpeed changes. Therefore future SetSpeed, Pause,
MultiPlanet and other unverified event types continue to abstain.

Fixed CPU results: v1 + v1.1 regression 22/22 PASS. On 72 generated varied
charts, 4456 eligible/partial frame forecasts were correct within their
declared covered interval, with 0 mismatched prefixes.

On 命日 (Nerfed) first 65s: 890 true target notes,
805 correct elapsed predictions, 0 incorrect inside segment (90.4% recall
on this already inspected chart). The idle Motor produces HUD; Timing
receives only current HUD frames. Scorer truth is used only posthoc.
The system still fails to track 85 notes; no physical Motor execution,
12-axis Student, Stage A gate change, checkpoint, Validation or FINAL.

CPU command:
uv run pytest -q tests/test_hud_timing_tracker_v1.py tests/test_hud_timing_tracker_partial_v11.py

Audit command:
uv run python -u scripts/audit_hud_timing_tracker_partial_v11.py PATH_TO_NERFED.adofai --end-s 65 --output diagnostics/timing-tracker-v11-nerfed-65s.json

Neither v1 nor any frozen Stage A/T baseline is modified.
