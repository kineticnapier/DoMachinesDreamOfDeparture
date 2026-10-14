# DMDOD Prestige I — Stage C v2.4 Actuator-aware Teacher Safety

## Frozen inputs and scope

- Base commit: `225088a197f39c716856f88ecbb26810b8b784a2`.
- Fixed Train20, same config, 263D visible HUD and unchanged 8-Key physics.
- Stage A **FAILED**, reason `gru_near_event_mae_not_20pct_better_than_constant_and_single`. Do not amend this gate.
- No Student inference changes, no Motor BC training, no Validation selection, no FINAL runs.
- Stage C v2.3 report is an immutable comparison baseline. All protected checkpoints, old Best, and prior audit files must retain their SHA256.

## Four failure mechanisms from the uploaded v2.3 report

| Train anchor | Last TooEarly KeyDown(s) | Interpretation |
| --- | --- | --- |
| #09 ALL MY FELLAS | 2.886s left_2 @ −207.8ms; 3.028s right_1 @ −65.8ms, both target #10 | Repeated TooEarly on a still-pending target; positive right_1 commands start at 2.980s |
| #10 碧空の約束 | 2.621s right_1 @ −91.8ms; 2.640s left_1 @ −72.8ms, both #2 | Previous #1 is pending when right_1 and left_1 commands overlap at 2.600–2.610s; target transitions between commands and KeyDowns |
| #13 My Piano Is Escaping | 1.477s left_1 @ −111.2ms; 1.514s right_1 @ −74.2ms, both #1 | The first note is missed; new candidate commands overlap near target transition |
| #16 Hyacinth | 6.742s left_1 @ −69.1ms; 6.745s right_1 @ −66.1ms, both #39 | 6.742s left_2 first resolves #38, then two keys press #39 outside its pass window, with only 3ms between the latter two |

These observations establish separate causes: scheduling overlap across a
resolution transition, actuator latency, and more than one physical KeyDown
before the next legal pass window. A `future_outside_pass` command alone
does **not** prove an error: legitimate pre-acceleration can be required.

## Fixed candidate — no parameter sweep

`ActuatorAwareTeacherSafety` wraps, but never changes,
`CenterFirstNKeyTeacher`:

1. At each 10ms control frame, take the **full motor state** (including
   acceleration-dependent state, key hysteresis, velocity and activation).
2. Copy it and step the **real 1ms motor** with the proposed command,
   holding that command for exactly `lead_s + control_dt_s`. There is
   no approximate constant-velocity arrival-time formula.
3. For each predicted physical KeyDown, apply the evaluator's actual
   unresolved-target, miss expiration, BPM/pitch and judgement windows.
   Multiple keys in one physics tick advance the target in event order.
4. If TooEarly occurs, try braking the corresponding single key (`-1`).
   Greedily choose a change only when the number of predicted TooEarly
   events decreases. Among changes with equal risk prefer fewer misses
   and more hits in the forecast; do not enable a new key.
5. Report unresolved projected risk instead of asserting safety.

**Limitation:** during the short forecast the command is held constant;
the future Teacher may release/reassign fingers. Consequently the forecast
can be pessimistic or miss a danger that depends on future decisions.
Predicted prevented events are **not** counted as observed successes.
Only actual simulator outcomes on fixed Train20 determine the result.

This is a teacher-side *training-data candidate*, not a production Student
gate. The next stage may build an offline BC dataset **only after** the
Train20 safety/Hit/Miss comparison has been reviewed.

## Actual comparison

The separate v2.4 runner verifies the v2.3 provenance, dataset manifest
and protected checksums; replays the unchanged charts with the candidate;
reports per-anchor and aggregate SAFE, Hits, Misses, TooEarly,
KeyDowns and unjudged notes, along with intervention counts and end traces.
No comparison runs on Validation20 or FINAL. The original v2.3 baseline
is read but not replayed or overwritten.

### Windows PowerShell

```powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_target_contract.py tests/test_prestige1_actuator_teacher.py
uv run python -u scripts/run_prestige1_actuator_comparison.py `
  --config configs/training/prestige1.toml
```

Output:
`diagnostics/prestige1/stage_c_v24_actuator_teacher_safety.json`

The runner refuses to replace an existing output. If local working tree
contains uncommitted edits, stop before `git pull` and integrate manually;
do **not** use reset, clean, stash, new branches or overwrite outputs.

## Decision gate for the next discussion

Assess the Train20 report *without* using Validation for tuning:

- First compare SAFE, Overload and actual TooEarly across all 20.
- Also compare Hits, Misses, KeyDowns and remaining unresolved targets.
- Inspect any new failure or lost-Hit anchors and whether the forecast
  is too conservative for dense note sequences.
- A better SAFE count by itself does **not** justify replacing the old
  Teacher or training a Student.
- Stage A remains FAIL regardless of these exploratory results.
