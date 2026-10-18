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


## Predecessor chain result (2026-10-18 user-provided output)

The read-only predecessor cascade isolates different root conditions:

- **Ordinary 480, HUD Network:** target33's scorer ancestry is
  33 -> 32 -> 31 -> 30 -> **29**. The first unobserved selected-axis
  physical KeyDown is target29: right_3 has no KeyDown within +/-100ms
  of due=1.937500s, even though the HUD Timing error is 0.000ms,
  first-seen lead is 77.5ms, and an assignment was made. The Oracle
  Network selected the *same* right_3 and pressed +5.5ms. The two
  arms differ in earlier Motor state and computed additional lead:
  HUD 5.274ms versus Oracle 18.622ms. This suggests physical
  activation/command readiness and requires per-frame verification,
  not timing tolerance adjustment.
- **SetSpeed early:** target5 is first predicted with 23.145ms lead,
  below the frozen 44ms normal physical lead, and intentionally
  unassigned. Target7's KeyDown then scores target5 VeryLate;
  target9's KeyDown scores target7 VeryLate. This is a genuine
  observation/physical-lead deficit for v1.4, not a bad target9
  forecast (which is numerically exact). Target6 is also marked
  insufficient lead by the earlier saved report.
- **SetSpeed late:** HUD Network and CenterFirst have score drift
  several targets before missed targets30/31. The full precursor
  trace shows the roots are earlier than target29. Compare physical
  chosen-axis timing rather than treating target30 as the first cause.
- **Pause:** Oracle Network and HUD Network share the same physical
  path. Target11 selects left_3 with extra lead 6.224ms but has
  no physical KeyDown within +/-100ms. Target12's right_2 KeyDown
  is instead consumed by target11 at +55.742ms; target13's
  left_1 KeyDown is consumed by target12 at +63.258ms. CenterFirst
  uses a different path and scores 32/32. This is a Motor/selection
  reachability candidate independent of HUD-only timing accuracy.
- **Train first 5s:** target0 is not forecast at all, target1 first
  appears with 20.163ms lead. Both have no logical/physical
  KeyDown and cannot be repaired by choosing different fingers
  for later already forecast events.

### Root-Motor diagnostic (frozen policy, no tuning)

The runner now supports an optional isolated per-10ms trace of one
physical axis, including pre/post position, velocity, activation,
KeyDown edges, issued command, reservation state and forecast status.

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --fixtures --case ordinary-480bpm --arm hud_network --trace-axis right_3 --trace-start-s 1.86 --trace-end-s 2.00 --output-dir artifacts/fingering_network_v02/hud-v14-right3-root-r1

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --special-fixtures --case special-pause_1 --arm oracle_network --trace-axis left_3 --trace-start-s .83 --trace-end-s .97 --output-dir artifacts/fingering_network_v02/hud-v14-left3-root-r1

Both are predeclared Train/synthetic diagnosis, with no Validation20
and no changes to model weights or Motor physical constants.
Read the AXIS_TRACE rows first to distinguish no positive Motor
command, command withdrawn early, and genuinely insufficient
actuator acceleration. No root performance fix is claimed yet.


## Confirmed Motor arm-boundary bug (2026-10-18)

Targeted +10ms-frame physical logs showed **neither a positive command
nor KeyDown** for the two first-failing selected actuators, and both
reservations disappeared *at exactly the planned arming step*:

- Ordinary 480 HUD Network target29 right_3 was reserved at 1.87/1.88s;
  canceled at 1.89s (due 1.9375s, nominal lead 44ms + extra 5.274ms,
  arming starts at 1.888226s). At 1.89s pos was -0.000997m,
  velocity +0.0008m/s, activation -0.0177, pressed false.
- Pause Oracle Network target11 left_3 was reserved at 0.84/0.85s;
  canceled at 0.86s (due 0.907258s, nominal lead 44ms + extra
  6.224ms, arming starts at 0.857034s). At 0.86s pos was
  -0.000999m, velocity +0.0002m/s, activation -0.0209,
  pressed false.

Code root: PersistentFingeringScheduler.arm() applied
MotorPhysicalSnapshot.ready() *again* at arming. That predicate
intentionally rejects ANY positive velocity for *fresh* allocation,
but it was reused for an axis that had already been exclusively
reserved for 20ms. While commanded neutral, the frozen physical
spring/hand model generated tiny positive passive velocity. arm()
raised ValueError; policy.command() then called cancel_unarmed()
and permanently marked that local event decided. This caused
zero Motor command, despite an exact Timing forecast and an
otherwise valid physical path.

Minimal fix: retain strict ready() when allocating a new axis, but
introduce ready_to_arm_reserved() for an ALREADY-EXCLUSIVELY-HELD
reservation. Require not pressed, below keyboard reset threshold,
and activation <=0; passive positive velocity alone no longer
deletes the reserved stroke. Existing true in-flight/pressed
release quarantine behavior is unchanged. Added independent scheduler
tests reproducing the observed right_3 and left_3 states plus a full
Timing-to-Motor command regression.

This fixes a proven source-code failure, **not a claimed new
Hit-rate result** until the user runs post-patch replay. All physical
constants, checkpoint, Timing v1.4 and scorer remain unchanged.

Minimal post-fix confirmation (create NEW output paths):

    .venv\Scripts\python.exe -m pytest tests/test_persistent_fingering_scheduler_v01.py tests/test_timing_fingering_motor_integration_v02.py -q

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --fixtures --case ordinary-480bpm --output-dir artifacts/fingering_network_v02/hud-v14-480-armfix-r1

    .venv\Scripts\python.exe -u scripts/run_hud_v14_fingering_motor_integration_v02.py --special-fixtures --case special-pause_1 --output-dir artifacts/fingering_network_v02/hud-v14-pause-armfix-r1

The SetSpeed early-forecast deficiency (target5 lead 23.145ms)
and Train first-forecast deficiency (target0 absent, target1
lead 20.163ms) are separate, **not fixed by this Motor change**.
If the two Motor fixtures still FAIL, examine changed first-failing
target and real KeyDown, do not loosen the fixed thresholds.


## Confirmed post-fix Windows reproduction (2026-10-18)

The user pulled the ready-to-arm-reserved fix and ran the two frozen
three-arm physical integrations, with the following console results:

| Case | Arm | Hits / Targets | Miss | TooEarly | Overload | X Accuracy | Gate |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| ordinary-480bpm | oracle_network | 48/48 | 0 | 0 | False | 100.0000% | PASS |
| ordinary-480bpm | hud_network | 48/48 | 0 | 0 | False | 100.0000% | PASS |
| ordinary-480bpm | hud_center_first | 48/48 | 0 | 0 | False | 100.0000% | PASS |
| special-pause_1 | oracle_network | 32/32 | 0 | 0 | False | 100.0000% | PASS |
| special-pause_1 | hud_network | 32/32 | 0 | 0 | False | 100.0000% | PASS |
| special-pause_1 | hud_center_first | 32/32 | 0 | 0 | False | 100.0000% | PASS |

Create-once Windows artifact folders:
- artifacts/fingering_network_v02/hud-v14-480-armfix-r1
- artifacts/fingering_network_v02/hud-v14-pause-armfix-r1

This is direct empirical support that the specific dropped-reservation
bug was responsible for the prior 480 and Pause Miss cascades.
It is not a claim that SetSpeed or Train startup now pass, and it is
not a Validation20 generalization result. No checkpoint, scorer, Motor
physical constants, or v1.4 Timing model were modified in the fix.

Unresolved from the pre-fix test: SetSpeed HUD 29/32 (initial
sufficient-lead shortage plus later event cascade), Train bool
5s HUD 7/9 (initial target unforecast and next with only ~20.163ms
lead). Those require visibility-boundary analysis, not reuse of the
physical reservation fix as an explanation.


## Post-reservation-fix comparison: remaining SetSpeed and Train cases

After the above 480 BPM and Pause passes, the user reran
the predeclared SetSpeed and Train five-second smoke with the fixed
physical reservation code.

| Case | Oracle Network | HUD Network | HUD CenterFirst | Gate |
| --- | --- | --- | --- | --- |
| SetSpeed speed_double | 32/32 Hit | 29/32 Hit | 29/32 Hit | FAIL |
| Train bool first 5s | 9/9 Hit | 7/9 Hit | 7/9 Hit | FAIL |

All six runs report zero TooEarly and no overload. SetSpeed
actual Miss ordinals remain 9, 30, 31, despite near-exact HUD
forecast due times for those three ordinals. The previous physical
reservation arming bug is **not** a complete explanation of these
failures. The earlier saved scorer replay identified lead insufficiency
at SetSpeed ordinals 5 and 6; missed ordinals can shift downstream
via ordered-scorer absorption.

Train target ordinal 0 at 3.999333624s has no predicted HUD
token and no selected physical actuation. Target 1 at
4.020163490s first appears only 20.163ms before due, shorter
than the frozen 44ms Motor normal lead. These are an observation
boundary and timing-lead issue, not an indication that the newly
fixed physical reservation predicate regressed.

### Next fixed HUD observation audit (no model or Motor policy changes)

The deterministic HUD v1.4 implementation intentionally blocks
future timing extrapolation across some SetSpeed/Pause boundaries.
The *reason* for each unforecast interval must be read from the
actual current-HUD stream rather than assumed.

New script diagnose_hud_v14_observability_boundaries_v02.py
runs only the idle Contact HUD renderer (the deterministic Timing
v1.4 tracker ignores all Motor features), and records each 10ms
frame's raw v1.4 status/reason, currently displayed Tile/Real BPM,
visible current and next floor markers, forecast due times and
first-confirmed event lead. The simulator's target truth is read
only AFTER completing the HUD stream for posthoc attribution.
An optional saved-HUD-arm comparison rejects a mismatch between
the idle-renderer first-seen times and the original run.

Run from the repository root (output paths are create-only):

    .venv\Scripts\python.exe -m pytest tests/test_diagnose_hud_v14_observability_boundaries_v02.py -q

    .venv\Scripts\python.exe -u scripts/diagnose_hud_v14_observability_boundaries_v02.py --case special-speed_double --targets 4,5,6,7,27,28,29,30,31 --compare-report artifacts/fingering_network_v02/hud-v14-failure-crosscheck-r1/special-speed_double/hud_network.json --output artifacts/fingering_network_v02/hud-v14-speed-visibility-r1.json

    .venv\Scripts\python.exe -u scripts/diagnose_hud_v14_observability_boundaries_v02.py --case train-bool-5s --targets 0,1 --compare-report artifacts/fingering_network_v02/hud-v14-failure-crosscheck-r1/train-bool-5s/hud_network.json --output artifacts/fingering_network_v02/hud-v14-train-visibility-r1.json

Do NOT weaken the frozen 44ms/69ms lead or invent the hidden new
BPM beyond a future SetSpeed boundary. First establish whether the
information is truly absent from the HUD or merely rejected by the
v1.4 estimator. Treat idle-to-Motor HUD parity failures as a blocker
to making that inference.


## HUD-only observability audit (2026-10-18, user Windows console)

The idle Contact HUD investigation passed parity against the previously
saved full-HUD Motor runs for BOTH fixed cases:

    speed: idle_to_motor_hud_parity.matched=True, no mismatched ordinals
    Train: idle_to_motor_hud_parity.matched=True, no mismatched ordinals

SetSpeed special-speed_double:
- Ordinal4 arrives at 0.362903s, forecast first appears at 0.290s,
  sufficient 72.903ms lead. Pre-arrival HUD shows tile BPM496,
  Real BPM992 and a *future SetSpeed marker* but no future numeric speed.
- When entering SetSpeed at ~0.37s, the visible BPM changes to Tile992,
  Real1984. Ordinal5 at 0.393145s then first appears with 23.145ms
  lead. Ordinal6 at 0.423387s first appears only at 0.400s,
  23.387ms lead, because the current SetSpeed floor is treated as
  the end of verified extrapolation. Frozen Motor normal lead is 44ms.
- Ordinals27..31 have 78.468..79.435ms lead; their numerical Timing
  estimates are near exact (1e-6..4e-6ms). Their previous Miss cascade
  cannot be attributed to lack of first-seen prediction coverage.

Train bool five-second smoke:
- Target0 at 3.999334s is never predicted; 3.92..3.99s are marked
  countdown_before_special_arrival with old displayed Tile/Real
  BPM60.01 and next-floor SetSpeed marker.
- At 4.00s the current HUD switches to Tile480.08,
  Real2880.479. Target1 at 4.020163s becomes predictable but has
  only 20.163ms lead. Its 44ms physical preparation target was
  already in the past.

### Information sufficiency question

The engine's early SetSpeed marker does not reveal the numeric new
BPM; the HUD263 layout has only current displayed Tile/Real BPM.
The floor-0 countdown orbit may repeat angular phases without exposing
the number of remaining countdown ticks. A guard removal alone could
thus lead to confident wrong-timing forecasts.

scripts/audit_hud263_nonidentifiability_v01.py and its tests construct
two causal counterfactual pairs:
1. Same geometry, same marker, different future SetSpeed numeric values.
2. Same geometry, old BPM and future SetSpeed marker, countdown tick
   counts differing by two rotations.

The program compares COMPLETE time-matched prefix HUD263 vectors,
then checks that future target deadlines differ by >44ms. It is an
offline counterfactual information test: true chart targets choose the
offline comparison horizon only, never enter a tracker or policy.

Passing pairs establish inability to recover all relevant future
deadlines *from the existing HUD263 observations alone*; not
necessarily an information limit for a richer genuinely player-visible
countdown or numeric preview. These new pairs and tests have not
yet been run by the user as of writing. No model/engine/policy code
or thresholds were changed.


## HUD263 non-identifiability: experimentally confirmed (2026-10-18)

The user ran scripts/audit_hud263_nonidentifiability_v01.py on
Windows with the two fixed counterfactual chart pairs. Both cases
reported INDISTINGUISHABLE; the combined counterfactual status
was PASS.

| Counterfactual intervention | HUD263 maximum absolute feature difference across matched prefix | Local probe time | First future differing due-time gap |
| --- | ---: | ---: | ---: |
| future SetSpeed numeric BPM, same visible floor/event-marker geometry | 0.0 | 0.33s | 90.726ms |
| countdown tick count differs by two while floor-0 angular phase repeats | 1.7484556000744134e-7 | 2.94s | 1999.667ms |

At the SetSpeed probe the unchanged v1.4 forecast was
eligible_partial; at the countdown probe it was
countdown_before_special_arrival. The numerical
HUD difference for the countdown pair is far below the
predeclared 5e-5 equality threshold; note that this is
numerical indistinguishability under the renderer's finite precision
rather than mathematical bit identity.

For these pairs, any deterministic causal Timing predictor receiving
only the same HUD263 prefix must issue the same prediction in both
worlds; their correct future deadlines differ. No larger model,
optimizer update, or different fingering alone can recover which
unseen world occurred.

Scope: This proves the existence of an observational alias in
the present HUD contract, NOT that every future SetSpeed or the user's
specific Train chart cannot be resolved if additional genuinely visible
signals are made available.

Non-solutions: Feed hidden future numeric BPM/countdownTicks/compiled
target times into the policy; use the Oracle comparator to rescue
missed HUD forecasts; lower safety thresholds or move physical
KeyDown blindly before confirming a due time; label abstained
unobservable targets as hits.

The next engineering question is observation provenance:
- Is an actual numeric future SetSpeed indicator displayed to the
  player before the event? The current HUD includes only current
  Tile/Real BPM and future SetSpeed marker.
- Is remaining initial countdown progression displayed or audibly
  announced? Floor-0 rotating phase alone cannot distinguish
  countdowns that differ by two revolutions.
- Is an audio stream available to the student as a sensory observation?
  If introduced, specify its causal latency, sample window, and whether
  playback truly contains disambiguating beats; do not hand it
  internal playback offset, chart time, or future note schedule.

Until a player-perceivable distinguishing signal is verified, a
calibrated ABSTAIN with explicit uncovered targets is the
correct output; an information-impossible target is not a v1.4
calibration bug.
