# Stage T — 16 keyboard contacts from 12 physically coupled motor axes

Status: **new, isolated contact topology feasibility diagnostic**. The frozen
8-Key / 263D Student, checkpoints, Stage A FAIL, Geometry/Hybrid and the
existing 8K/16-independent-actuator Motor capacity reports remain unchanged.
There is no model training, Validation or FINAL.

## Fingering specification (reported by the player)

Per hand (the left-hand description is mirrored for this test):

| Fingering | Four non-thumb fingers | Thumb contact | Hand edge contact | Total |
|---|---:|---|---|---:|
| 4K | 4 keys | unused | unused | 4 |
| 5K | 4 keys | tip (one key) | unused | 5 |
| 6K | 4 keys | laid flat (two keys) | unused | 6 |
| 7K | 4 keys | tip (one key) | 2-key simultaneous pair | 7 |
| 8K | 4 keys | laid flat (two keys) | 2-key simultaneous pair | 8 |

The hand edge normally presses both assigned keys together, rather than
pressing either independently. Rare expert edge-1K techniques are *not*
included. Thumb tip versus laid-flat selection is represented as a **shared
physical actuator with selectable contact geometry**, not as independent
thumb motors. The assumption that the two flat-thumb contacts share an
identical physical KeyDown instant is a **first-pass modeling choice**,
not a measured statement about actual human biomechanics.

## Motor structure

- One hand has **6 motor axes**: four independent non-thumb fingers, one
  thumb, one shared hand-edge. Two hands have 12 independently driven axes.
- One hand has **8 logical key contacts**: four finger contacts, two on the
  shared thumb, two on the shared edge. Two hands have 16 contacts.
- The existing `NKeyMotorEnv(12)` integrates the **unchanged generic
  NKeyBody physics**: 10ms action hold and 1ms physical steps. This is
  not a new anatomical calibration. Shared-hand coupling is inherited
  from the 12-axis NKeyBody and is not a validated thumb/edge muscle model.
- A key-contact adapter transforms each physical threshold event to one
  or two *same-time* logical KeyDown/KeyUp events, based on the chosen mode.
  No contact can issue a KeyDown without a physical actuation event.
- Thumb and hand-edge mode changes are forbidden during a physical stroke,
  including forward motion before reaching the KeyDown threshold; this
  avoids fabricating key events by toggling an already depressed axis.
- **Limit:** Thumb-tip/flat posture change while at rest is instantaneous
  and has no movement cost; lateral key locations, reach, true joints,
  physiological fatigue, palm contact force distribution and 16K-specific
  muscle coordination are not modeled.
- The modes are *action parameters*, not a Student output head. No 16K HUD,
  gameplay chart evaluator or neural network is integrated in this stage.

Logical contact labels on either hand:

- `<hand>_1..4`: ordinary individual fingers
- `<hand>_5`: thumb tip (also active in flat mode)
- `<hand>_6`: additional thumb flat contact (only in flat mode)
- `<hand>_7..8`: inseparable pair on the hand edge

Physical actuator labels `<hand>_1..6` map 1..4 to ordinary fingers, 5 to
thumb, 6 to the hand edge. It is deliberate that one **physical** channel
can generate two distinct **logical** key events.

## Deterministic smoke experiment

Each of 4K, 5K, 6K, 7K, 8K is tested on **both hands separately**, then
8K+8K together. Each episode uses one continuous physical Motor: press for
30 × 10ms frames, release for 40 × 10ms, then press again for 30 × 10ms,
without resetting between strokes. Check exact logical KeyDown and KeyUp
sets/counts, shared-pair event timestamps, model mode change guards, and
non-generation of events on disabled contacts.

| Mode | First KeyDown count | KeyUp count | Second KeyDown count |
|---|---:|---:|---:|
| Left or right 4K | 4 | 4 | 4 |
| Left or right 5K | 5 | 5 | 5 |
| Left or right 6K | 6 | 6 | 6 |
| Left or right 7K | 7 | 7 | 7 |
| Left or right 8K | 8 | 8 | 8 |
| Both hands 16K | 16 | 16 | 16 |

The 7K/8K modes generated their two edge events at exactly the same 1ms
physics timestamp; the two flat-thumb contacts were also simultaneous.
These fixed, static strokes **do not** establish any BPM clearance, timing
accuracy, release-retrigger capacity under realistic chart sequences, or
real-life 16K skill. The model has no ADOFAI evaluator in this audit.

## Reproduction

```powershell
uv run pytest -q tests/test_prestige1_autoplay_coupled_contact_audit.py
uv run python -u scripts/audit_prestige1_autoplay_coupled_contact.py --config configs/training/prestige1.toml
```

The immutable output report (refuses replacement) is:

    diagnostics/prestige1/stage_t_coupled_16contact_topology_readonly.json

## Next isolated experiment

Test short *sequences* rather than one simultaneous chord: can the same
thumb/edge actuator release and reuse key pairs under physically relevant
BPMs, especially when the chart needs only one of the two edge keys?
That will expose an important mapping constraint before adding chart
judgement, persistent planning or training. Keep it separate from the
frozen 8-Key Student and explicitly record unwanted simultaneous contacts.
