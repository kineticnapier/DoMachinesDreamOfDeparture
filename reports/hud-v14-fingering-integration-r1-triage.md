# HUD-only Timing v1.4 × Fingering: first Windows r1 triage

Observed on Windows 2026-10-18, as reported by the user. This report
summarizes the *existing saved r1 data* and does not claim new local
simulation or completed fixes. The script outputs remain local; the
GitHub repository has the integration and diagnostic sources.

## Initial integration

- Ordinary 120/240 BPM: Oracle, HUD Network and HUD CenterFirst all
  48/48 Hit, 0 Miss, 0 TooEarly, no overload.
- Ordinary 480 BPM: Oracle 48/48, HUD Network 47/48, HUD CenterFirst
  48/48, no TooEarly or overload in any arm.
- SetSpeed speed_double: Oracle 32/32, HUD Network 29/32,
  HUD CenterFirst 29/32. No TooEarly/overload.
- Twirl speed fixture: 32/32 Hit for all three arms.
- Pause fixture: Oracle Network 31/32, HUD Network 31/32,
  HUD CenterFirst 32/32. No TooEarly/overload.
- Train bool first 5 seconds: Oracle 9/9, HUD Network 7/9,
  HUD CenterFirst 7/9, no TooEarly/overload.

The declared integration gates correctly FAIL overall. They must not
be weakened to hide these results.

## Saved r1 forecast/decision-only analysis

The r1 JSON did not include actual missed target ordinals, so the
following local indices describe forecast observations, NOT proven
missed events.

- Ordinary 480 HUD Network: zero unmatched target forecasts, zero
  insufficient lead and 48 assigned decisions. CenterFirst all Hit.
  Most informative next check: selected physical axis, recent actuation
  history, KeyDown delay, cancellation vs reset and timing difference.
- SetSpeed HUD Network and CenterFirst: 2 targets (ordinals 5 and 6)
  appear in insufficient lead, with 30 assigned decisions and 3 misses.
  A third Miss is not assigned to a cause by the saved aggregate. Check
  whether the third is cascading physical reservations or a distinct
  Timing/physical failure.
- Pause: both Network arms have one Miss, while CenterFirst gets all
  hits. Oracle's Miss prevents explaining this as solely a HUD Timing
  estimator defect; compare actual selected physical axes and timing.
- Train bool 5s: HUD arms each have 2 Miss. Target ordinal 0 has no
  forecast within 20ms; ordinal 1 has insufficient lead. Actual
  missed ordinals were not stored in r1 and must be confirmed.

## Differential follow-up (predeclared, no hyperparameter search)

Run exactly four cases (ordinary 480, speed double, pause, Train bool 5s)
through three frozen comparison arms. Keep current Network checkpoint,
style alpha, timing horizon, physical motor constants, and scorer fixed.
Only add *posthoc* physical timing and target-outcome diagnostics.

Use:

    .venv\Scripts\python.exe -m pytest tests/test_diagnose_hud_v14_failed_cases_v02.py -q

    .venv\Scripts\python.exe -u scripts/diagnose_hud_v14_failed_cases_v02.py --output-dir artifacts/fingering_network_v02/hud-v14-failure-crosscheck-r1

The script creates separate create-once arm JSON files and case-level
comparisons; interrupted runs preserve completed arm files. Comparison
of nearby KeyDown times is temporal association, not proof of target/
physical-finger causality.

Do not change frozen Timing v1.4, retrain Fingering, or use Validation20
to improve scores before examining the physical timeline. If a condition
is genuinely unobservable or the available lead is shorter than the
Motor's physical response time, report abstention/miss instead of
filling in Oracle true time.


## Target-level findings from first crosscheck (user-provided console output)

The targeted crosscheck finished all 4 frozen cases x 3 arms.
These are observations from the returned console output, not a proof
of evaluator attribution until the saved logical-event scorer replay agrees.

- Ordinary 480 HUD Network: target 33 Miss, even though selected left_1
  had a physical KeyDown at +2.500ms to the target. Oracle Network
  left_1 was -3.500ms and CenterFirst left_2 was -0.500ms; both Hit.
  All three had a correct +0.000ms Timing forecast for target 33.
- SetSpeed HUD Network and CenterFirst: actual Miss target ordinals
  **9, 30, and 31**. Forecast insufficiency ordinals 5 and 6
  were NOT themselves the reported missed target ordinals.
  Target 9: HUD Network left_2 at -4.113ms and CenterFirst right_1
  at +0.887ms, yet both target9 Miss.
  Target 30: HUD Network left_1 +3.806ms and CenterFirst right_2
  +9.806ms, both Miss.
  Target 31: the selected axis (HUD Net right_3, CenterFirst left_1)
  has no physical KeyDown within the reported 100ms window.
- Pause: target 13 Miss for both Network arms; both chose left_1
  with +2.774ms physical KeyDown. CenterFirst chose left_2 with
  -1.226ms and Hit. The frozen Timing due was exact for these events.
- Train bool first 5s: target 0 has no local forecast/physical KeyDown;
  target 1 has only ~20.163ms first-seen forecast lead and no physical
  KeyDown; both HUD arms Miss those ordinals. Oracle hits both.

### Why physical KeyDown near a missed due is not proof of a bad scorer

The existing RealChartMotorEnv._score_event uses _next_target_index
(the first still-unresolved target) rather than the intended local
Timing token. A physical KeyDown close to target i could therefore
have scored a previous unresolved target j. That hypothesis is especially
relevant for 480/SetSpeed/Pause where selected physical press times
are within a few ms of the missing target due.

New replay_hud_v14_scorer_routing_v02.py consumes the *saved logical*
switch streams, reruns only the scorer (no Motor/Network/Timing rollouts),
and compares every Hit/Miss flag and aggregate count against the
saved per-arm JSON. It reports target consumption only when scorer
reconstruction matches the saved original exactly.

Run from the repository root, after pulling the new code:

    .venv\Scripts\python.exe -m pytest tests/test_replay_hud_v14_scorer_routing_v02.py -q

    .venv\Scripts\python.exe -u scripts/replay_hud_v14_scorer_routing_v02.py --input-dir artifacts/fingering_network_v02/hud-v14-failure-crosscheck-r1

No physical experiment or new output artifact is created by the
scorer replay. Do not alter timing tolerance, student checkpoints,
Motor constants, or the evaluator to make these results PASS.


## Exact scorer replay: confirmed target-consumption chains

User-ran replay_hud_v14_scorer_routing_v02.py on saved raw switch traces.
**All twelve** case/arm scorer reconstructions matched the original
Hit/Miss/TooEarly counts and target flags exactly. This confirms the
following *immediate* consumption links:

| Case/arm | Missed ordinal | KeyDown intended near missed due actually scored |
| --- | ---: | --- |
| ordinary 480 / HUD Network | 33 | target 32: target33-adjacent +2.5ms KeyDown scores 32 at +65.0ms VeryLate |
| SetSpeed / HUD Network | 9 | target 7: -4.113ms vs target9, +56.371ms on target7 |
| SetSpeed / HUD CenterFirst | 9 | target 7: +0.887ms vs target9, +61.371ms on target7 |
| SetSpeed / HUD Network | 30 | target 29: +3.806ms vs target30, +34.048ms on target29 |
| SetSpeed / HUD CenterFirst | 30 | target 29: +9.806ms vs target30, +40.048ms on target29 |
| SetSpeed / HUD Network | 31 | no logical KeyDown within 25ms |
| SetSpeed / HUD CenterFirst | 31 | nearest target31-adjacent KeyDown scores 29 instead |
| Pause / Oracle and HUD Network | 13 | target 12: +2.774ms vs target13, +63.258ms on target12 |
| Train bool first 5s / both HUD arms | 0, 1 | no logical KeyDown within 25ms; 0 has no forecast, 1 has only 20.163ms lead |

This is **not a scorer bug**: the scorer intentionally scores the
first unresolved target and uses its Normal-mode pass window (which
can overlap the next event). Thus a KeyDown accurate for target i may
legally be a VeryLate hit for target i-1 or even i-2. A downstream
Miss is a symptom of an earlier unserved stroke; simply widening
HUD Timing tolerances or reassigning the scorer target would conceal
the underlying scheduling failure.

The immediate predecessor is not necessarily the *first* failed
actuation. The latest read-only diagnostic recursively inspects the
ordered-scorer ancestry and the chosen axis's nearest real KeyDowns,
comparing the same target across all three frozen arms. This does
**not** rerun physical simulation:

    .venv\Scripts\python.exe -m pytest tests/test_diagnose_hud_v14_predecessor_cascade_v02.py -q

    .venv\Scripts\python.exe -u scripts/diagnose_hud_v14_predecessor_cascade_v02.py --input-dir artifacts/fingering_network_v02/hud-v14-failure-crosscheck-r1

Use its earliest anomaly per chain as the next physical Motor/axis
command candidate. Do not modify the ADOFAI scorer, Network weights,
Timing v1.4 confidence or reservation limits to force a PASS.
