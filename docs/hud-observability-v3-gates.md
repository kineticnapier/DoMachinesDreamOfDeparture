# DMDOD — HUD-only observability barrier and v3 sensor gates

## Basis: frozen Windows counterfactual evidence

Run: scripts/audit_hud263_nonidentifiability_v01.py

| Hidden variable changed between two charts | Difference in all observed HUD263 frames through probe | Future event-time disagreement | Verdict |
| --- | ---: | ---: | --- |
| Future numeric SetSpeed value, unchanged visible geometry and marker | 0.0 | 90.726ms | INDISTINGUISHABLE |
| Countdown length, changed by two rotations | <=1.7484556000744134e-7 | 1999.667ms | INDISTINGUISHABLE |

The equality tolerance was 5e-5, fixed before running.
The SetSpeed probe was at local time 0.33s, status eligible_partial.
The countdown probe was at 2.94s, status countdown_before_special_arrival.

For those constructed worlds there is NO deterministic mapping from
the same causal HUD263 prefix to two different correct future event
times. It is inappropriate to fix this by introducing the hidden
chart values to the Timing model.

This is an existential identifiability result, not proof that every
possible chart fails. Even a stochastic model cannot guarantee the
correct unseen alternative in both indistinguishable worlds.

## Research goal

Distinguish genuinely player-perceivable information from evaluator
truth, and determine whether a *new sensor* can separate the two
counterfactual worlds early enough to issue a 44ms Motor command.

The requirement is NOT merely that a future target becomes observable
eventually. For a finger requiring 44ms ordinary lead, the
distinguishing observation must be received with enough time to
command the physical Motor; otherwise the target remains uncovered.

## Candidate observation channels (not yet approved)

1. Actual game-displayed countdown digit/tick or other visible
   remaining-countdown indicator, recorded on the same frame as the
   player's view. Do not use countdownTicks from the parsed chart.
2. Audio samples that the player actually hears at the current frame,
   with recording/playback latency and an explicit causal sampling
   boundary. Audio is NOT automatically sufficient: two different
   chart files may use identical sound recordings despite having
   different future SetSpeed actions. Only measured audio differences
   between the counterfactual worlds can disambiguate them. Do not
   infer unknown BPM from chart actions or future targets.
3. An actually displayed numerical future speed preview, if the
   game renderer exposes one. A SetSpeed event marker alone does not
   reveal its numeric speed; the current HUD's Tile/Real BPM are for
   the current floor, not the next floor.
4. A genuinely visible count of completed floor-0 countdown cycles,
   but ONLY if display state makes it measurable. A pure orbit-phase
   counter cannot recover different unseen countdown lengths by
   accumulating phase from the start, as both pairs have identical
   phase histories.

These are research questions, not a decision to add features. New
numeric channels require provenance that they are truly perceivable
before passing them to a HUD-only student.

## Versioned v3 boundary

Retain existing 263D as the frozen v1/v2 compatibility view. Any
approved observation must be a NEW versioned view, with a separate
dimensionality and a documented causal timestamp. Never write hidden
values into unused HUD263 fields or reuse the legacy first-24D motor
zeros as unversioned extra data.

The sensor boundary must state:
- When a player first sees/hears the signal, and sampling latency.
- Whether it is available on control frame t with access to frames
  <=t ONLY.
- Missing/unreadable signal representation (validity mask or abstain).
- Proof that data source is visible or audible, not a compiled-chart
  shortcut, a target ordinal, or a privileged playhead time.
- Reset and prefix-invariance behavior for different future hidden
  actions.
- How already-frozen Fingering 65D context can remain unchanged.

## Acceptance tests before any neural or Motor change

A. Re-run both original counterfactual pairs: they MUST remain
   indistinguishable under legacy HUD263 as a regression.
B. For a proposed real sensor, show recorded causal prefixes from
   each world and identify the FIRST frame at which the additional
   player-perceivable data differs. If it never differs, reject it.
C. At that first distinguishing frame, calculate posthoc lead to
   each distinct future target. Show whether the frozen 44ms Motor
   preparation budget can still be met. Do not call an insufficient
   lead a usable observation.
D. Compare counterfactual prefixes where hidden values change but
   player-visible/audio channels do not; new model must abstain.
E. Preserve same single-chart fixtures (120/240/480, SetSpeed,
   Pause, Twirl, Train first 5s) and Oracle/HUD/CenterFirst arms;
   no Validation20 threshold selection and no changed scorer.
F. Only after A-E are met, implement a new versioned observation
   adapter. Do not modify frozen Timing v1.4 or the old checkpoint.

## Known unresolved chart tests

- SetSpeed 32-event test: Oracle 32/32, HUD Network and
  HUD CenterFirst 29/32. The initial post-transition target5/6
  appears with 23.145/23.387ms lead (insufficient for 44ms),
  while many later events are exact but are affected by scorer
  target-consumption cascades.
- Train bool 5-second test: Oracle 9/9, HUD arms 7/9.
  Target0 never forecast; target1 appears only 20.163ms early.
- Frozen arm-reservation bug fix DID make 480 BPM all 48/48
  and Pause all 32/32 in three comparison arms.

## Interim behavior

Do not bluff target timing. Keep status ABSTAIN or report
insufficient_observable_lead, preserve full Hit/Miss coverage
denominators, and maintain physical-axis reservation correctness.
A partial success on eligible forecasts is not full-chart success.


## Real gameplay observation — bool recording (2026-10-18)

The user provided a full-screen, 60fps, 1920x1080 recording with
48kHz stereo audio (67.85 seconds). Approximate times below are
VIDEO-relative, not simulator episode times.

| Video time | Observed player-perceivable signal |
| --- | --- |
| 0.8–1.2s | A prominent on-screen 'Ready!' graphic |
| 1.44s, 2.44s, 3.44s, 4.44s | Four distinct audio pulses, one-second spacing |
| ~4.47s | On-screen 'Go!' is visible |
| ~5.44–5.45s | First input/hit advance and sustained song/audio start |

The audio pulses are grounded in the recorded track, not inferred
from future chart target data. On this one clip, the on-screen Go!
appeared almost a second before the first actual note inputs.

**Additional potentially decisive difference:** The recording's
top-right display before the first note visibly reads
'TBPM 60.01' and 'CBPM 15'. The current DMDOD implementation in
src/dmdod/features/hud.py::hud_bpms_for_floor explicitly returns
(tile_bpm, tile_bpm) for floor index zero, so its startup
Real BPM channel would show 60.01 instead of the distinct 15
readout seen in this video. This difference may matter for
identifying the initial countdown duration.

The user confirmed that the TBPM / CBPM overlay is supplied by a **Mod**.
Therefore the displayed numeric CBPM is **not eligible for the default
stock-game HUD-only observation**. It can only be considered under an
explicit Mod-equipped experimental observation contract, with verified
semantics and update latency. The apparent stock floor-0 renderer
mismatch is NOT evidence that DMDOD drops a stock-game display.

The user explicitly confirmed that Ready!, Go!, and the audible
countdown cues are present in unmodified stock ADOFAI. These are
therefore VALID STOCK observation *channels* for a new sensor
contract. Their exact timing semantics remain unverified: seeing Go!
in this video roughly one second before the first hit is one sample,
not a universal prediction rule. A chart-derived floor-0 duration
or parsed countdownTicks still counts as privileged truth.

These cues are preliminary evidence that stock HUD263 *omits*
real signals rather than proving that the full player-visible
observation is non-identifiable. The old paired 3-count/5-count
HUD263 proof remains correct for THAT input contract.

Next smallest disambiguating comparison: two short recordings
with the same notes and initial Tile BPM, but different
countdownTicks (3 versus 5). Compare visible Ready/Go onset times,
audio pulse pattern, and top-right CBPM as a function of
time since record/game-start. Determine whether each observable
difference first occurs >=44ms before the shorter-world
first target. No v1.4 or Motor changes before that check.


## Stock Countdown channel: next controlled validation

- Source: actual rendered Ready!/Go! pixels or recorded audible tick,
  not a synthetic cue generated from hidden compiled countdownTicks.
- Inputs: timestamped (frame-local) visible cue detections and/or
  audio-onset detections; include observed/unknown validity bit.
- Time origin: relative to *perceived cue arrival*, calibrated
  against actual captured video/audio latency, never chart time.
- Pair test: record two otherwise identical charts with countdown
  tick counts 3 and 5; extract the first perceptible divergence
  of their 263D-plus-cue prefixes. The legacy HUD263 prefixes
  must remain indistinguishable as previously verified.
- Physical gate: first distinguishing cue must leave at least
  44ms to the *shorter-world first playable note*; otherwise
  it is too late even if it distinguishes the worlds.
- Timing rule: cannot assume a fixed Go-to-first-hit delay from
  the one bool recording. Derive/validate the relationship only
  from causally available real cues and visible geometry. Preserve
  ABSTAIN where the cue is hidden, not yet observed, or ambiguous.
- Do not pass an alleged 'Go' timestamp to the policy from
  chart action metadata or audio file position.

SetSpeed remains independent. The confirmed unknown future
numeric SetSpeed counterexample cannot be resolved solely by
adding countdown cues.
