# Stage T — varied-angle 16-contact Motor, planned versus physical timing

**Status:** isolated, read-only **oracle Motor-only** diagnostic. The frozen 8-Key / 263D Student, MaleCNS4096, Stage A FAIL, original Geometry/Hybrid, checkpoints and existing 10ms control/1ms physics are unchanged. **No learning, Validation or FINAL.**

## Scope and fixture provenance

The repository does not include a user-supplied music chart for this test. This experiment compiles **three deterministic *generated* ADOFAI-format charts** with non-straight floor angles, then runs the existing `CoupledContactRealChartEnv` and the **same `ContactOracleTeacher`**. Unlike the earlier 180-degree straight-floor benchmark, the generated profiles include varied inter-note spacing and event modifiers. They are **not** actual commercial/community authored songs or holdouts.

| Generated fixture | Floors / targets | Distinct angles | Events | Shortest target gap |
|---|---:|---:|---|---:|
| `bends_1200` | 72 | 8 | none | 12.50ms |
| `events_1800` | 96 | 8 | two SetSpeed, one Twirl, one Pause | 13.04ms |
| `tight_bends_4000` | 108 | 8 | none | 7.50ms |

Each condition runs a single persistent 12-physical-axis Motor to chart completion. Modes **4K through 8K per hand** use static thumb/edge topology for that episode. The teacher receives **privileged true upcoming target times and ordinals**; the Motor and contact observation do not. The 44ms lead and reservation logic are unchanged (no tuning or hyperparameter sweep). The contacts of a flat thumb or hand edge are generated at the **same physical KeyDown timestamp**.

## Attribution: three genuinely different quantities

1. **Physical KeyDown time compared with the teacher's reserved target time** (planned timing MAE): includes the second target of paired contacts and counts physical onsets whether or not the evaluator rewards them.
2. **Physical KeyDown time compared with the evaluator's *actually claimed* target time** (scorer Hit MAE): shows the timing of Hits under the current normal judgement, which may not correspond to what the teacher intended.
3. **Reserved target ordinal versus actual scorer target ordinal** (mismatch count): detects target-cursor drift after misses or late/misaligned inputs. Separate counters track physical KeyDowns without an active reservation, judgement events and paired reservations.

These three measurements must **not** be conflated. A two-contact pair creates two same-time logical inputs for **two different target times**. This cannot demonstrate independent human strikes.

## Results (unchanged evaluator; motor-only oracle)

| Generated fixture | 4K H | 5K H | 6K H | 7K H | 8K H |
|---|---:|---:|---:|---:|---:|
| bends 1,200 BPM (72 targets) | 72 | 72 | 72 | 72 | 72 |
| events 1,800 BPM (96 targets) | 96 | 96 | 96 | 96 | 96 |
| tight bends 4,000 BPM (108 targets) | 95 | 108 | 108 | 108 | 108 |

At tight bends **4,000 BPM**, Hit count by itself hides substantial error:

| Keys/hand | Planned KeyDown MAE | Planned vs scorer ordinal mismatches | Paired axis reservations |
|---:|---:|---:|---:|
| 4K | 34.31ms | 49 | 0 |
| 5K | 35.88ms | 2 | 0 |
| 6K | 32.76ms | 7 | 9 (thumb) |
| 7K | 31.92ms | 2 | 8 (edge) |
| 8K | **29.44ms** | 2 | 17 (both types) |

For the 8K/hand condition, mean intended target spacing **within a paired reservation was about 13.83ms**, yet both logical KeyDowns occur **at the exact same physical instant**. The judgement still gives 108/108 Hits under current DMDOD rules. This is **not** proof of human 16K capability, accurate timing, or a trained HUD-only Student. The 8K/hand condition also had two physical contacts without an active teacher reservation.

At bends 1,200 BPM, 8K planned timing MAE was 13.50ms; at events 1,800 BPM, 15.82ms. Evaluator-perfect Hit totals can coexist with substantial timing deviation. The teacher is simplistic and not optimized for these patterns.

## Limits

- **Generated only:** not a chart uploaded by the user; no music/audio observation or real chart source.
- **Oracle teacher:** exact target timestamps/ordinals are read by the teacher from the evaluator; no HUD-only inference or training.
- **Static per-hand contact posture:** thumb tip/flat mode and edge enabled state do not switch during a chart; real joint dynamics not represented.
- **Current incomplete same-frame Multipress behavior:** two physically simultaneous coupled contacts can claim two different future notes and inflate Hit counts. The user elected to defer full Multipress parity; its limitation remains explicit.
- **Generic reused 12-axis Motor:** not anatomical measurement or proof of muscle endurance / fatigue.
- **Attribution counts** reflect physical contacts as provided by the existing contact layer; no independent per-logical-contact actuator is invented.

## Reproduce

```powershell
uv run pytest -q tests/test_prestige1_autoplay_coupled_chart_patterns_audit.py
uv run python -u scripts/audit_prestige1_autoplay_coupled_chart_patterns.py --config configs/training/prestige1.toml
```

New report path (refuses overwrites):

`diagnostics/prestige1/stage_t_coupled_16contact_varied_angles_intent_readonly.json`

**Next separate experiment:** run exactly this timing/attribution report on an actual `.adofai` supplied by the user (perhaps a selected short segment), and record the source chart identity, timing modifiers and any unsupported events. That is necessary before making conclusions about realistic charts or a future new 16-contact HUD-only Student.
