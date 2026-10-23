# DMDOGO v0.8 — Adaptive Gesture Chunks

## Research scope

v0.6 used a global three-key periodic schedule. The physical Train bool replay
completed 1,122/1,122 Hits, but required 233 local replans and 626 deferred
events; X Accuracy was 92.362%.

v0.7 segmented timing into 16-event windows but still applied an eight-axis
cyclic pattern. Its 10-second physical prefix completed 57/57, X 100%,
Timing MAE 4.855 ms; no full-chart v0.7 physical result exists.

v0.8 generates **short local gestures**, not chart-wide key cycles.
This does not establish human imitation.

## Algorithm

1. Detect salient inter-onset gaps from a 17-gap local context. Separate
   chords and 450ms pauses. Split continuous streams into at most four-note
   planning units; this cap is not a measured musical phrase boundary.
2. Generate local single strokes, same-hand consecutive rolls and reversals,
   hand exchanges, paired exchanges, cross-hand interleaving and short rebounds.
3. Stitch candidate gestures with a deterministic, bounded width-24 beam.
   Require at least 80ms abstract same-axis reuse and distinct chord axes.
   Score local movement, hand transitions, reuse margin and time-decayed
   recent hand workload. The 750ms hand exposure constant is **not a
   calibrated fatigue model**; the beam is not globally optimal.
4. Hand selected axes to the existing physical Motor unchanged: eight
   ordinary uncoupled axes, live eligibility, 90ms prediction horizon,
   44ms local fallback boundary and the existing lead compensator.

This is offline Oracle timing, **not HUD-only Student observation**.
No Network scoring, physical future outcomes, checkpoint retraining,
Validation20 or FINAL are used.

## Frozen bool offline comparison

The already uploaded full v0.6 bool replay (SHA256
04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1)
contains 1,122 events.

v0.8 prototype results:
- 392 gesture chunks; 79 single, 94 doublet, 21 triplet, 198 four-note.
- 160 same-hand rolls, 153 paired exchanges, 79 single-stroke chunks.
- Abstract reuse violations: zero; minimum axis reuse 156.224ms.
- Identical assigned axis eight events later: 66.697% versus v0.7 94.883%.
- v0.8 proposed axes differ from v0.7's offline choices in 882 events.

The first 20.83ms doublets are assigned as independent L1-L2 and R1-R2
gestures with alternating hand workload, rather than a global eight-axis
pattern. Repetition itself is not disallowed or proof of poor fingering.

These offline results do not establish human-like style.
Subsequently, the user ran a complete frozen v0.8 physical episode on the
same Train bool chart: 1,122/1,122 Hit, X 100%, timing MAE 4.500752ms,
0 local replans, 0 deferrals, 0 overload (termination chart_complete).
See docs/dmdogo-v08-frozen-baseline.json and
docs/dmdogo-v08-freeze-generalization.md for the reference and protocol.
These outcomes are still single-chart observations, not generalization proof.

## Windows verification

Stay on develop/dmdogo and preserve all local modified/untracked files.
Inspect git status before a fast-forward pull. Never reset, stash, clean,
merge or rebase to force a pull.

Focused tests:
- tests/test_gesture_chunk_planner_v08.py
- tests/test_gesture_chunk_audit_v08.py
- tests/test_gesture_chunk_physical_v08.py
- tests/test_human_phrase_physical_v07.py
- tests/test_pattern_motor_v06.py

First run the existing physical runner with:

--planning-mode gesture_v08
--chart "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai"
--output-dir artifacts/dmdogo-v08/bool-prefix-r1
--status-file logs/dmdogo-v08-bool-prefix-r1.status.json
--stop-after-s 10 --wall-timeout-s 180

Using uv run python scripts/run_dmdogo_v06_pattern_first.py as the executable.

Separate create-once outputs:
summary.json, telemetry-v08.json, replay-v08.json, replay-v08.html.

Offline-only audit:
uv run python scripts/audit_gesture_chunks_v08.py
with --input-replay artifacts/dmdogo-v06/bool-full-r1/replay-v06.json
and --output reports/dmdogo-gesture-v08-bool-r1.json

After a successful diagnostic prefix, a full Train bool episode can use
--output-dir artifacts/dmdogo-v08/bool-full-r1
and --status-file logs/dmdogo-v08-bool-full-r1.status.json
with --wall-timeout-s 1200, omitting --stop-after-s.

Do not run Validation20 or FINAL without explicit approval.

## Verification caveat

Eight focused planner tests passed locally with synthetic examples. The
full integrated repository tests were not executed in the assistant's
environment; the complete Train bool Windows physical simulation was later
executed by the user, with results recorded in the frozen reference. Genuine human-likeness evaluation will require aligned
human finger-key recordings from consenting participants, not only Hit rate
or reduced exact eight-period recurrence.
