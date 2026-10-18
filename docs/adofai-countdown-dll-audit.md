# ADOFAI stock countdown: Assembly-CSharp DLL/IL audit

Source basis: user Library adofai.db indexed Assembly-CSharp.dll,
SHA-256 0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD.
The corresponding reverse-compiled C# text comes from the user Library
adofai-code.zip. Verified the actual indexed IL for scrCountdown.Update,
scrConductor.GetCountdownTime, scrConductor.adjustedCountdownTicks and
scrLevelMaker.CalculateFloorEntryTimes against the reverse-compiled sources.

This is **implementation inspection** of one DLL build, not proof all game
versions, settings or camera modes are identical. The gameplay HUD v3 must
read actual visible/audio cues, NEVER the code's internal countdownTicks.

## Exact source relationships

- ADOFAI/LevelData.cs: level songSettings["countdownTicks"] is
  exposed through LevelData.countdownTicks.
- scnGame.cs::RemakePath sets
  scrConductor.instance.countdownTicks = levelData.countdownTicks.
- scrConductor.cs:
  - crotchetAtStart = 60 / bpm.
  - adjustedCountdownTicks = countdownTicks / countdownSpeedMultiplier.
  - Default countdownTicks=4 and countdownSpeedMultiplier=1.
- scrCountdown.cs::Update:
  - Before GetCountdownTime(0), shows localized status.getReady,
    subject to speed-trial/practice text variants.
  - After first tick, shows integer count; as each subsequent
    GetCountdownTime(i) elapses, the value is countdownTicks - 1 - i.
  - After GetCountdownTime(countdownTicks-1), shows localized status.go,
    setting controller.goShown and a UI tween.
  - Uses DSP time corrected by calibration_i, updated on Unity frames;
    text changes may lag audio sample boundaries.
  - A separate fastTakeoff path can show Go immediately. Checkpoint,
    special game states and custom extra countdowns have other paths.
- scrConductor.cs::PlayHitTimes:
  - When playCountdownHihats and !fastTakeoff, loops i from zero to
    countdownTicks-1 and schedules sndHat at GetCountdownTime(i).
    Past sound times are skipped.
  - The audio is actual player-heard evidence; the conductor schedule
    itself is evaluator-side only.
- scrConductor.cs::GetCountdownTime:
  - For checkpointNum=0, let N=countdownTicks,
    B=crotchetAtStart, p=song.pitch,
    m=countdownSpeedMultiplier, o=addoffset,
    S=dspTimeSong. Then:
    
        T_tick(i) = S + i * B / (p * m) + o / p

  - At checkpoints the implementation branches to a DIFFERENT formula
    involving the next floor's entryTimePitchAdj, checkpoint floor speed
    and planet count; the expression above must not be generalized.
- scrLevelMaker.cs::CalculateFloorEntryTimes:
  - First playable floor's compiled unpitched entry time is

        entryTime[1] = (N/m - 1) * B
                     + GetTimeBetweenAngles(
                         floor0.entryangle, floor0.exitangle,
                         floor0.speed, bpm, !floor0.isCCW)

  - It then stores entryTimePitchAdj = entryTime[1] / p.
    Thus a Go! label is **not a universal first-note timestamp**.
    The Go-to-first-hit separation depends on floor-0 geometry and
    settings; for ordinary m=1 and conventional timing, it is the
    remaining floor-0 angle traversal, not always one beat.

## Normal-start examples

For the default countdown speed m=1:
- N=4 UI normally goes Ready -> 3 -> 2 -> 1 -> Go,
  accompanied by four hats at index 0..3 when enabled.
- Relative time from the first hat to Go is 3*B/p.
- N=3 vs N=5 with otherwise identical floor-0 geometry changes the
  first playable due time by 2*B/p; if bpm=60.01 and pitch=1,
  this is ~1.999667s (matching the previous counterfactual test).
- The observable digit differs right after the FIRST hat:
  N=3 displays 2; N=5 displays 4. Therefore unlike the old HUD263
  orbit/BPM-only observation, the *full stock screen* distinguishes
  these two counterfactual worlds well before the shorter-world hit
  in this slow-BPM example.
- The visible/audible Go cue provides an independent phase anchor.
  It is not necessary to assume there is a fixed one-second delay
  after Go, and doing so would fail at other BPM/angle layouts.

## Consequences for DMDOD

The previous HUD263 non-identifiability experiment is correct AS STATED,
because the current HUD263 excludes these stock cues. It does **not**
show that real-player observation during stock Countdown is
information-theoretically ambiguous.

Countdown: provide a **versioned stock observation** of actual screen
Ready/numeric/Go and optional causal audio tick onsets. Do NOT pass
LevelData.countdownTicks, scrConductor.countdownTicks, the computed
GetCountdownTime, any clock-based future event schedule or chart truth
to the student. The adaptation layer must consume only what it can
detect in current/past frames. Number recognition may directly convey
remaining ticks, but first-floor geometry and tick cadence must still
be estimated from observations and measurement latency.

For every detected cue, compare earliest OBSERVED distinguishing frame
to first playable due POSTHOC. The existing Motor needs ~44ms preparation.
A counter at very high BPM or a late-rendered Go cue might be too late,
so observability alone is not a promise of full coverage.

SetSpeed: Future SetSpeed action numeric BPM is not revealed merely
because a SetSpeed marker is visible. The original 90.726ms
SetSpeed-counterfactual ambiguity remains separate. The game countdown
digits and hat schedule do not solve hidden future SetSpeed numbers.

No production HUD/Timing/Motor/Network implementation was changed
as part of this audit.
