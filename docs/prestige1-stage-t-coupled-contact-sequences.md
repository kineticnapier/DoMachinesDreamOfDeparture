# Stage T — 12-axis / 16-contact consecutive-stroke and unwanted-key diagnostic

**Status:** isolated deterministic Motor-only contact experiment. The frozen
8-Key/263D Student, original Stage A FAIL, Geometry/Hybrid, 10ms policy clock,
1ms physical integrator and checkpoints are not changed. No training,
Validation or FINAL. The contact layer is **not** calibrated human anatomy.

## What this experiment answers

The current experimental hand model has, on each hand, four separate finger
axes, one thumb axis with tip-1K / flat-2K selectable contact modes, and one
hand-edge axis whose two keys are always pressed together. Two hands have
**12 simulated actuators / 16 logical contacts**.

Previous static mode tests established that 4K through 8K contacts can be
pressed and released. This follow-up probes *short consecutive strokes* and
compares desired contact sets to **actual physical KeyDown event sets**, not
to theoretical counts. There is no ADOFAI Hit/Miss evaluator in this test.

### Eleven fixed scripts

| Script | Desired per-stroke input | Actual physical result | Extra inputs |
|---|---|---|---:|
| Edge, key 7 only | `left_7` | `left_7 + left_8` | 1 |
| Edge, key 8 only | `left_8` | `left_7 + left_8` | 1 |
| Edge, 2-key chord | `left_7 + left_8` | Both simultaneously | 0 |
| Thumb tip | `left_5` | `left_5` | 0 |
| Thumb flat | `left_5 + left_6` | Both simultaneously | 0 |
| Thumb flat, key 6 only | `left_6` | `left_5 + left_6` | 1 |
| Edge 7 then edge 8 | Two distinct single-key strokes | Two 2-key chords | 2 |
| Thumb tip then flat | 1K, release, 2K | Intended contacts | 0 |
| Thumb flat then tip | 2K, release, 1K | Intended contacts | 0 |
| Finger + edge | `left_4 + left_7 + left_8` | All three | 0 |
| Both hand edges | `left_7 + left_8 + right_7 + right_8` | All four | 0 |

**Interpretation:** Isolating either edge contact or only the second
flat-thumb contact is **not representable by this particular contact model**.
The motor emits real simultaneous KeyDown events for both. An ADOFAI
simulator might regard such surplus contacts as unwanted presses or
Overload, but this experiment **does not** simulate its judgement rules.
Also, some expert real players can isolate a single edge key; their technique
is deliberately absent from the baseline model.

### Continuous physical recovery and mode transitions

Each scripted stroke holds its selected 12-axis Motor commands for **six
10ms control frames** (60ms total), then commands release until the axis
passes the original model's reset position and is no longer moving forward.
No Motor reset occurs between strokes. The next step changes the contact
mode only after that safe-release guard is satisfied. Press/release event
traces, recovery frames, safe-to-switch times, and surplus contacts are all
recorded. The test also rejects premature thumb/edge mode changes while the
physical axis is still depressed or moving into the key.

For these fixed 60ms strokes the first pulse returned to a switchable state
at ~130ms from the episode start (**70ms after release command started**).
The second pulse after continuous physical recovery could be started at that
same 130ms instant. This 70ms is **not** a human thumb-posture switching
time, nor the minimum release time of the simulator: it is a result for
this one fixed motor command waveform. At rest, the current contact model
switches posture **instantaneously** with no anatomical transition cost.

## Files and reproduction

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_coupled_contact_sequences_audit.py
uv run python -u scripts/audit_prestige1_autoplay_coupled_contact_sequences.py --config configs/training/prestige1.toml
~~~

The new immutable report (refuses overwrite) is:

    diagnostics/prestige1/stage_t_coupled_contact_sequences_readonly.json

No code is changed in the existing 12-axis contact adapter. This diagnostic
uses its real physical KeyDown/KeyUp events, not analytically invented presses.

## What remains untested

- No actual ADOFAI chart, BPM judgement/Overload, or sustained mixed-key
  pattern. These scripts fully release between strokes and are *not* evidence
  of high-frequency interleaved performance.
- No separate posture-transition dynamics, thumb joint geometry, motor-axis
  calibration against real fingering or isolated-edge expert technique.
- No policy able to select fingers and contacts based on HUD observation.
- No comparison between this 12-axis topology and independent 16-actuator
  model on the same full gameplay evaluation, and no trained Student.

**Next independent experiment:** add an isolated synthetic contact-event
judgement probe with explicit surplus-press/Overload accounting, before
linking the model to a real-chart evaluator or modifying learning pipelines.
