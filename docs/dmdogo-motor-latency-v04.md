# DMDOGO v0.4 — Motor latency audit

## Contract

This audit keeps the AutoPlay due times, Fingering Network, physical constants,
scorer and existing selectors frozen. It analyzes the complete Free replay from
`artifacts/dmdogo-continuity/bool-v03-r2/replay-v03.json`.

All 1,122 events are one-contact Hits and the repository scorer consumes them
in order, so event/physical-KeyDown pairing is unique by FIFO order. Nearest
time matching is not used. Six rows have a selected-axis/physical-axis mismatch
because in-flight strokes can reorder physical contacts; these are retained
and flagged instead of silently reassigned.

For the frozen runner, selection and the first Motor command occur in the same
10 ms control frame. Thus:

- scheduled lead = Oracle due − decision time;
- physical response = KeyDown − decision time;
- timing error = physical response − scheduled lead.

## Result

| population | N | response mean | median | Q25–Q75 | timing error mean |
|---|---:|---:|---:|---:|---:|
| all | 1,122 | 55.755 ms | 56 ms | 50–61 ms | +16.670 ms |
| 480 travel BPM | 737 | 53.597 ms | 53 ms | 48–58 ms | +14.370 ms |
| 1920 travel BPM | 384 | 59.940 ms | 60 ms | 55–65 ms | +21.128 ms |
| selected-axis reuse ≤125 ms | 53 | 66.453 ms | 67 ms | 64–69 ms | +29.399 ms |

The scheduled lead itself is only 34.002–43.999 ms (mean 39.085 ms), because
44 ms is a visibility window sampled on a 10 ms grid, not 44 ms of guaranteed
actuation. The cold first stroke takes 39 ms and lands at −0.334 ms. The later
distribution reaches 75 ms under shared hand/bilateral dynamics and recovery.

This rejects a single constant-delay explanation. Control quantization creates
roughly a 10 ms scheduled-lead range, but density, hand coupling and short-axis
reuse move the physical response by substantially more. A blanket 16 ms
advance would make cold/rest cases early and still under-compensate the short
reuse tail.

## Implementation consequence

Motor compensation must be state-aware, start from an earlier planning window,
and preserve a legacy fallback. The v0.4 estimator therefore predicts the
earliest actuation reachable from the current Motor state using only the frozen
deterministic Motor equations and already-visible state. It never reads future
KeyDowns or scorer outcomes. The prediction is used as a safety lower bound:
if even the fastest isolated stroke would arrive no earlier than the due time,
the command may be latched; otherwise it waits one control tick.

This forward model is not a Teacher candidate rollout: it neither evaluates
judgements nor ranks fingering candidates. It forecasts one already-selected
physical command from current state. Cross-axis coupling makes the isolated
forecast optimistic, so actual arrival can be later but should not become
earlier solely because of other simultaneous commands.

## Reproduction

```powershell
.\.venv\Scripts\python.exe scripts\analyze_motor_latency_v04.py `
  --replay artifacts\dmdogo-continuity\bool-v03-r2\replay-v03.json `
  --policy free `
  --output reports\dmdogo-motor-latency-v04-bool-r1.json
```

The command is create-once. Full per-event rows and distributions are stored in
`reports/dmdogo-motor-latency-v04-bool-r1.json`.
