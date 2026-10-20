# Hold Timing H1

## Implemented contract

`Hold.duration` is validated as a finite integer. Missing and fractional values
are rejected; negative values and Hold on the final floor disable the Hold in
the same way as the audited stock loader. Duration 0 remains a real Hold with
no additional loop.

For a non-initial Hold floor, the stock base angle is calculated first with the
existing Twirl/Midspin/MultiPlanet/SetSpeed rules. Then:

```text
additional_angle = duration * 2 * PI_STOCK
additional_time  = additional_angle / PI_STOCK * 60 / resolved_floor_bpm
```

Pause remains additive. Chart times are calculated before pitch; episode Hold
start/end times divide chart-relative time by the existing pitch ratio.

The special floor-0 path is unchanged because the audited stock
`CalculateFloorEntryTimes` does not add Hold loops while calculating the first
floor. This is represented but not claimed as a playable Hold start.

## Data separation

- `CompiledHoldTiming` is privileged chart/evaluator data.
- `PlayableHoldInterval` is separate from `PlayableChartTarget`.
- Existing ordinary target meaning and ordering are unchanged.
- No Hold duration or end time is exposed through HUD263.

## Verification

Synthetic parity fixtures cover duration 0/1/2/3, BPM, pitch, Pause,
SetSpeed-at-entry, Twirl, Midspin, MultiPlanet, consecutive Hold, final Hold,
negative duration, inactive Hold, missing duration, and fractional duration.
Hold-free chart tests and existing replay/HUD timing tests pass unchanged.

The formula is code-parity against the fingerprinted decompiled snapshot, not
yet a pixel/runtime comparison with the installed game. Real Train Hold chart
results are recorded separately in H5.
