# Stage T — Timing Tracker v1.2: current-floor SetSpeed

Experiment-only successor to v1.1. A SetSpeed marker on the **currently
visible floor** is permitted: its already-observed unclipped HUD Tile BPM
is used to forecast subsequent ordinary floor exits.

A future SetSpeed marker STILL causes abstention. In exploratory v1.1
testing, relaxing the future speed guard created three incorrect target
predictions in the real chart. Pause / MultiPlanet / full-turn remain
unsupported. Twirl safe-prefix semantics, chronological tracker IDs and
explicit window truncation are inherited unchanged from v1.1.

On the fixed first 65s of 命日 (Nerfed), with no Motor presses,
the chart contains 890 targets. The causal 263D HUD tracker produced
829 correct elapsed predictions, **zero** incorrect in the scored
segment (93.1% measured recall on this chart). Remaining missed: 61.
One or more predictions beyond the 65s clip are excluded posthoc,
not counted as wrong. Independent SetSpeed fixtures: 48 combinations
with 4295 eligible frame predictions and zero incorrect sequences.
v1/v1.1/v1.2 CPU tests: 27/27 PASS.

NO stage A change, NO Student learning or use of oracle target identity
at inference, NO Validation/FINAL, NO Motor/Teacher edits. This is not
a clean unseen-chart generalization result, a deployable Timing model,
or proof of physically executable future events.

Run regression tests:
uv run pytest -q tests/test_hud_timing_tracker_v1.py tests/test_hud_timing_tracker_partial_v11.py tests/test_hud_timing_tracker_current_speed_v12.py

Run read-only chart audit:
uv run python -u scripts/audit_hud_timing_tracker_current_speed_v12.py PATH_TO_NERFED.adofai --end-s 65 --output diagnostics/timing-tracker-v12-nerfed65s.json


## Missed-target attribution on the fixed real-chart clip

Read-only posthoc diagnosis of the 61 untracked target times, assigning the
tracker status at each due instant:

- 34: full_turn_or_zero_arc
- 24: special_future_floor
- 2: special_event
- 1: eligible despite the target remaining untracked

These are status correlations, not a proof of exact failure cause.
Last-30ms context grouped 18 all-full-turn, 16 mixed full-turn,
24 future-special, 2 current-special, and 1 other. Next high-value work is
a conservative HUD-only full-turn interpretation, then special-event
timing with separate speed/pause contracts. Do not invent progress when the
input does not resolve the ambiguity.
