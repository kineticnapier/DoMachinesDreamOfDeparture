# DMDOD Prestige I — Timing Teacher = Autoplay (exact compiled chart timestamps)

## Decision

**Replace the research Timing Teacher's target with Autoplay-equivalent
exact input-event timestamps.** Finger choice and 8-Key Motor dynamics remain
separate. The exact time is the teacher label; there is no extra "learned
timing correction" or intentional 10ms/1ms quantization of this truth.

\`\`\`text
Compiled ADOFAI chart (.adofai)
   └─ build_playable_segment: input-bearing floors ONLY
        (floor 0 excluded; Midspin floors excluded)
        └─ AutoplayTimingTeacher (privileged oracle)
             event_index, episode_time_s, chart_time_s, chord_size
             ├─ Timing Student training labels ONLY (future design)
             └─ Physical Motor teacher target time (future wiring)
                    └─ Fingering Planner + actuator lead + 1ms physics
                         └─ actual KeyDown at target +/- physical error

Student inference: 263D visible HUD HISTORY ONLY -> predicted timing;
never the oracle timeline, episode clock, floor index, or chart schedule.
\`\`\`

"Autoplay" here is explicitly the **compiled simulator's** target schedule:
\`PlayableChartTarget.episode_time_s\` already incorporates pitch, segment
offset, and playable-floor exclusion. This is not a verified replay of the
game's internal native Autoplay method. Native timing parity is separate.

## Exact time and loss masks

- \`episode_time_s=(chart_time_s-segment_start_chart_s)/pitch_ratio\`.
- Retain Python float (double) precision; do **not** round a target to
  a 10ms action frame or a 1ms physical tick.
- For time \`t\`, the label is \`target_episode_time_s - t\`.
  Far future labels are **not clipped** at 200ms. A separate
  \`near_200ms\` flag retains the current Stage A 200ms classification
  concept; \`visible_floor_window\` is an evaluator-only label mask.
- \`signed_delta_s(event_index, now_s)\` also represents overdue events
  as a **negative** remaining time without silently clamping to zero.
  \`future_event(now_s)\` intentionally skips past onsets; it is not
  interchangeable with the physical simulator's *next unresolved target*.
- Simultaneous playable targets are grouped (<=1e-7s onset difference)
  into one event with chord size 1–8. An invalid >8-input chord fails
  closed. Chords do **not** assign fingers.
- The Teacher exports the target time exactly; a Motor command must start
  **earlier** to compensate for actuation latency. This work does not
  change the safe v3.0 850/850 physical action plans or their measured
  12.20ms MAE. The oracle's zero reference timing error cannot be
  mistaken for perfect physical Motor performance.

## What is actually implemented

- \`src/dmdod/training/prestige1_autoplay_timing.py\`:
  immutable exact evaluator-side event timeline, unclipped near labels,
  explicit signed query, pitch/chord handling and visibility mask.
- \`src/dmdod/training/prestige1_autoplay_export.py\`:
  write-once Train20-only compiled event schedule, strictly matching the
  frozen Stage A and v3.0 report manifests and the 850 Hit ordinals.
  Each compiled timestamp is independently checked against
  \`physical_keydown_time_s - physical_error_ms/1000\` (tolerance 1µs).
- \`scripts/export_prestige1_autoplay_timing.py\`:
  export CLI, no Student/model execution.
- \`tests/test_prestige1_autoplay_timing.py\`:
  exact non-grid timestamp, pitch, chords, 200ms mask, signed overdue
  query, visibility and cross-source corruption checks.

Output, created once:
\`diagnostics/prestige1/stage_timing_autoplay_exact_train20.json\`.

The export contains **privileged timestamps** and may only be read by
training-label/evaluator code. Do not ingest the exported JSON as an
ordinary Student observation, as that would leak target truth.

## Existing code is intentionally not overwritten

\`prestige1.py\` Stage A v2 still uses the old 0..200ms clipped label
interface, with frozen FAIL gate. Its saved pretrained checkpoints,
source manifest, and existing v3.0 Teacher plans have not been changed.
This new exact oracle is the canonical source for any **future new**
Timing training path, not a claim that an existing Student has been
retrained or that Stage A is now PASS.

The already-separated \`FingeringPlanner\` continues to choose fingers;
the Motor policy continues to learn/execute physical trajectories and
must satisfy 1ms true physics (all Hit, zero Miss/TooEarly/Overload/
extra KeyDown) independently.

## Windows commands

\`\`\`powershell
git status --short
git pull --ff-only

uv run pytest -q tests/test_prestige1_autoplay_timing.py

# Only if tests pass. Train20 data export, no Student training.
uv run python -u scripts/export_prestige1_autoplay_timing.py --config configs/training/prestige1.toml
\`\`\`

Never overwrite existing report/checkpoints, alter original Stage A
gate, use Validation/FINAL for selection, or claim simulator physical
precision equals the exact theoretical schedule.
