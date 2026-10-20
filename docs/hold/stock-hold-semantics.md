# Stock Hold semantics audit (Stage H0)

## Scope and evidence status

This audit uses the local decompiled source snapshot in `F:\adofai-code` and
the Train80 charts only. The snapshot files are dated 2026-09-15, but the
assembly declares version `0.0.0.0`; therefore their SHA256 fingerprints in
`reports/hold-h0-train-inventory-20261020-r1.json` identify the audited code,
but do **not** prove that another installed build has identical behavior.

Terms in this document:

- **CODE_CONFIRMED**: directly present in the audited control flow.
- **CHART_CONFIRMED**: present in the frozen Train80 data.
- **RUNTIME_UNVERIFIED**: not compared with a running stock game build.
- **UNSAFE_TO_IMPLEMENT**: evidence is ambiguous or contradictory.

## Parsed event and timing semantics

| Question | Audited behavior | Evidence | Status |
|---|---|---|---|
| Hold start tile | An active `Hold` action assigns `holdLength` to its floor when duration is non-negative. | `scnGame.cs` Hold cases | CODE_CONFIRMED |
| Hold requires a following tile | In the final event pass, `holdLength` is set only when `nextfloor` exists; otherwise `-1`. | `scnGame.cs:1170` | CODE_CONFIRMED |
| Invalid duration | Negative duration disables Hold (`holdLength=-1`). The loader casts the value to `int`. | `scnGame.cs:487-504`, `:1168-1171` | CODE_CONFIRMED |
| Base angle | Hold preserves the ordinary floor's stock angle, including Twirl/Midspin/MultiPlanet geometry. | `scrLevelMaker.CalculateFloorEntryTimes` | CODE_CONFIRMED |
| Additional angle | `duration > 0` adds `duration * 2 * pi`; duration 0 adds no loop but remains a Hold. | `scrLevelMaker.cs:591-594`, `:637-640` | CODE_CONFIRMED |
| Added time | Each full loop adds `2 * timeBetweenAngles(0, pi, floor.speed, conductor.bpm)`. | `scrLevelMaker.cs:637-640` | CODE_CONFIRMED |
| Pitch | Chart entry time is calculated first; `entryTimePitchAdj = entryTime / pitch`. | `scrLevelMaker.CalculateFloorEntryTimes` | CODE_CONFIRMED |
| Pause | Existing stock flow adds Pause separately to the floor timeline. Exact ordering with every Hold/SetSpeed offset needs parity fixtures. | source paths are separate | RUNTIME_UNVERIFIED |
| SetSpeed | Hold loops use the floor's resolved `speed`. Mid-floor SetSpeed behavior depends on the stock resolved floor speed and must be checked numerically. | `scrLevelMaker.CalculateFloorEntryTimes` | CODE_CONFIRMED formula; RUNTIME_UNVERIFIED combination |
| Twirl | Twirl changes direction; full added loops are direction invariant. | timing flow + added `2*pi` loops | CODE_CONFIRMED |
| Midspin | A duration-positive Midspin would receive zero base time plus full loops in the timing formula. Gameplay validity is not established. | `CalculateSingleFloorAngleLength` | RUNTIME_UNVERIFIED |
| MultiPlanet | Base offset uses planet count; added Hold loops remain full `2*pi`. A special visual tween exists for MultiPlanet Hold. | `scrLevelMaker`, `scrPlanet.cs:1055+` | CODE_CONFIRMED timing; RUNTIME_UNVERIFIED input behavior |
| Consecutive Hold | `nextTileIsHold` is an explicit branch in input handling. Exact required key sequence at the shared boundary needs runtime comparison. | `scrPlayer.UpdateHoldBehavior/UpdateHoldKeys` | CODE_CONFIRMED branch; RUNTIME_UNVERIFIED outcome |

Hold end time is the next floor entry time. It is not a second independent
ordinary target with the same scoring meaning as a tap. The ordinary press at
Hold start, Hold completion/release, and the next playable floor must therefore
be represented separately by DMDOD.

## Input state machine observed in stock code

`scrPlayer.Simulated_PlayerControl_Update` executes, in order:

1. remove released captured Hold keys;
2. check post-Hold fail;
3. process auto Hold hit;
4. enqueue newly pressed valid keys;
5. process release behavior;
6. process checkpoint-started Hold;
7. check missing Hold state;
8. consume queued presses through `UpdateHoldKeys`.

At a successful Hold start, `Hit()` advances onto the Hold floor. The code then
clears `holdKeys` and captures all valid keys pressed on that frame. On later
frames, releasing **any** captured key removes it and reports a release event.
An unrelated key is not automatically added to `holdKeys`; it enters the normal
keydown queue and its effect depends on the Hold behavior mode and timing.

### Mode table

| Setting | `strictHolds` | `requireHolding` | Early release | End action evidenced by code |
|---|---:|---:|---|---|
| Normal | true | true | fail outside allowed end margin | release is the intended completion input |
| CanHitEnd | false | true | fail outside allowed end margin | queued press may call `Hit`; release remains accepted |
| NoHoldNeeded | false | false | no required-holding death | queued press/release may advance at the end |

The flags are assigned in `scrController.SetupImportantVariables` from
`HoldBehavior.Normal`, `CanHitEnd`, and `NoHoldNeeded`.

The end threshold is derived from angle, not a fixed millisecond window:

```text
minAngleMargin = adjusted Counted boundary(BPM, pitch)
holdMargin = 1 - minAngleMargin * nextFloor.marginScale / currentFloor.angleLength
```

On release in gameworld, completion below `holdMargin` fails when
`requireHolding`; completion at/after it advances when the next tile is not
another Hold. The end `Hit()` path suppresses ordinary previous-Hold judgment
effects, so DMDOD must not award both an ordinary tap and a Hold completion for
the same transition.

### Ambiguous code requiring runtime evidence

`CheckPreHoldFail` tests
`holdCompletion > holdMargin && holdCompletion < 1 - holdMargin`. For ordinary
positive angle margins, `holdMargin` is near 1, making this interval appear
reversed. This may be unreachable, decompiler output may obscure intended
semantics, or another runtime value may alter it. DMDOD will not implement a
continuous “not holding” failure from this branch until a stock capture proves
the condition.

Likewise, Unity frame ordering for simultaneous multi-key down/up, the exact
outcome of re-press after partial release, and chained Hold boundary behavior
remain RUNTIME_UNVERIFIED.

## Visual evidence relevant to H3

The audited renderer exposes distinct start/release icons
(`HoldArrowLong/Short`, `HoldReleaseLong/Short`) and a `scrHoldRenderer` whose
total angular length includes `holdLength * 2pi`. This proves that Hold presence
and a loop-shaped path are renderer state, but not yet that the integer duration
can be recovered robustly from the pixels available to the Student. No chart
`duration` may be copied into HUD features.

## Frozen Train80 inventory

The H0 audit found 3 charts and 25 active Hold actions, with durations
`0, 1, 2, 3`:

| Chart | SHA256 | Holds |
|---|---|---:|
| `04349 - ヒアソビ (feat. 初音ミク) - Zedwards.adofai` | `baaf6bff3cd012d8b0e4f123f58868c1cd4f262a35394e32eb2ebcba898caab9` | 2 |
| `10907 - Remote Access Trojan - 0mni & Trippy.adofai` | `d7e5a1cb5abc8ea1104116ac154f5f1e0b9fd889fdf2650ed1fe1d27510f4b82` | 22 |
| `13131 - 『계획 이상 有!』 (feat. 重音テト) - Team Name.adofai` | `167e781dca406fcec3a0ee754a173d40a50b23f1714ee93758ca3eaf53564d64` | 1 |

The last chart has Hold duration 0 and Twirl on the same floor. Remote Access
Trojan supplies duration 0/1/2/3 and consecutive-Hold candidates. No evaluation
split outside Train80 was opened.

## Minimum stock comparison cases

H1 timing fixtures:

1. duration 0, 1, 2, 3 on identical geometry;
2. 90/180/270/360-degree base arcs;
3. BPM 120/240 and pitch 50/100/150;
4. SetSpeed before Hold, on Hold at offset 0, and after Hold;
5. Pause + Hold, Twirl + Hold, Midspin + Hold, MultiPlanet + Hold;
6. consecutive Hold, final-floor Hold, negative/fractional/missing duration;
7. no-Hold byte-identical timing regression.

H2 input fixtures for each verified mode:

1. correct start and release inside the end window;
2. early release; late release;
3. two keys pressed at start, then release one vs both;
4. unrelated keydown while holding;
5. release then re-press;
6. consecutive Hold boundary;
7. reset at every state.

Cases depending on exact live-game frames are tagged runtime-unverified rather
than declared PASS. H1 may implement the direct timing equation now. H2 may
implement a deterministic, explicitly versioned state machine for confirmed
transitions, while the contradictory continuous-hold branch remains disabled.
