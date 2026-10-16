# Stage T — Coupled 16-contact KeyDown into the current ADOFAI judgement scorer

**Status:** read-only, evaluator-only physical-keydown diagnostic. No Student,
Stage A FAIL, original Geometry/Hybrid, fixed MaleCNS4096, model training,
Validation, FINAL, or Motor-physics changes.

## Question

If one edge-of-hand actuator causes **two same-time physical KeyDowns**,
and the chart currently needs only one, does the extra input become TooEarly
(TE), and when does the existing ordinary Overload gauge fail?

This experiment distinguishes a *physical contact* from an *ADOFAI judged
note*. One extra key is not unconditionally one TE: the **currently
implemented DMDOD scorer** offers each KeyDown to the **next unresolved
playable target**. It does not yet implement the real game's complete
Multipress/OverPress same-frame queue semantics.

## Protocol

1. Generate a deterministic straight 180° real-chart fixture at fixed
   600, 1,200 or 4,000 BPM, with 10 playable targets.
2. Run the existing `Coupled16ContactMotor` with a 12-axis physical body,
   10ms held commands and 1ms physics, and collect **real emitted 16-contact
   `TimedKeyEvent` KeyDowns**, not analytically fabricated key events.
3. Use the fixed *Motor-only* pre-roll of 60ms (600 BPM), 10ms (1,200 BPM),
   or 0ms (4,000 BPM), with **zero** action during pre-roll so that the
   first actual press occurs near the first chart target. These are
   evaluator-alignment fixtures, **not HUD-only policy decisions**.
4. Submit physical contact events **in their adapter-defined order** to
   the unchanged `DiagnosticHudNKeyRealChartMotorEnv` per-event judgement
   methods: advance ordinary overload decay, expire misses, and score
   event. The actual Motor has 12 axes and 16 logical contacts, so this
   diagnostic directly invokes scorer methods instead of running a full
   16-actuator gameplay `env.step()`.
5. Record exact KeyDown time/key, currently offered next target ordinal,
   Hit/TooEarly/FailOverload, and ordinary overload gauge after each event.
   No evaluator state affects contact commands.

## Results — seven fixed scenarios

| BPM | Physically pressed contacts | Actual judgement sequence | Hits | TE | Final gauge | Overload |
|---:|---|---|---:|---:|---:|---|
| 600 | Single ordinary finger (control) | Perfect | 1 | 0 | 0.0 | No |
| 600 | Left hand-edge 2K | Perfect, TooEarly | 1 | 1 | 0.5 | No |
| 600 | Flat thumb 2K | Perfect, TooEarly | 1 | 1 | 0.5 | No |
| 600 | Both hand-edge pairs (4K simultaneous) | Perfect, TooEarly, TooEarly, FailOverload | 1 | 3 | 1.5 | **Yes** |
| 1,200 | Left hand-edge 2K | Perfect, VeryEarly | 2 | 0 | 0.0 | No |
| 4,000 | Left hand-edge 2K | Perfect, Perfect | 2 | 0 | 0.0 | No |
| 4,000 | Both hand-edge pairs (4K simultaneous) | LatePerfect, Perfect, Perfect, Perfect | 4 | 0 | 0.0 | No |

At 600 BPM, normal pass windows reach about ±65ms, but the next normal
180° target is 100ms away. Hence two KeyDowns at ~99ms score the first
100ms target as Perfect; the remaining KeyDown against the 200ms target
is TooEarly. Ordinary TE gauge increments by 0.5 per TE, and its current
strict `>1.0` failure rule means two TEs leave gauge exactly 1.0 without
Overload; the **third** TE makes it 1.5 and produces `FailOverload`.

At 1,200 BPM consecutive notes are 50ms apart and at 4,000 BPM they are
15ms apart. The current scorer's **minimum time window** permits the
additional same-time press to claim the next floor. This should **not**
be interpreted as proof that actual ADOFAI would award those same-frame
multipresses as distinct Hits; the model explicitly does not yet emulate
that behavior fully.

## Reproduce

```powershell
uv run pytest -q tests/test_prestige1_autoplay_contact_judgement_audit.py
uv run python -u scripts/audit_prestige1_autoplay_contact_judgement.py --config configs/training/prestige1.toml
```

This writes a **new immutable** diagnostic; re-running refuses overwrite:

`diagnostics/prestige1/stage_t_contact_keydowns_adofai_judgement_readonly.json`

## Limits and next step

This is **not an integration test of full 16K gameplay**: contact events
are injected into the existing per-KeyDown scorer without a full `env.step`
and without subsequent Motor play or miss expiry. The current scorer lacks
complete Multipress/OverPress same-frame rules. Do not replace its
judgements with guesses, infer real-world playability, or train the fixed
8-Key Student from these labels yet.

If further game fidelity matters, the next separate task should be to
verify same-frame Multipress/OverPress queue behavior against the actual
ADOFAI game/DLL and only then implement corresponding evaluator rules,
with tests that protect this frozen diagnostic as a historical baseline.
