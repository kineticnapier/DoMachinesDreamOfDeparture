# DMDOD Prestige I — Stage C v2.3 Target Contract & Teacher Safety

## Context

The completed Train20 v2.2 audit found 3,389/13,681 control frames
with a chronological-vs-unresolved target mismatch, including 3,011
frames pointing to an already-judged note. Train20 Expert itself
terminated by Overload on anchors 09, 10, 13 and 16.

These are **observations**, not proof that one issue causes the other.
Stage A original gate remains FAILED. Original Stage A/B/C checkpoints
and any Validation/FINAL runs are untouched.

## Part I: Evaluator-only Target Contract

New pure module: `src/dmdod/training/prestige1_target_contract.py`.

The physical next target is the earliest `not used and not missed`
target, *not* simply the next chart time. For every individual target
at the current physical time the contract distinguishes:

| phase | condition | physical KeyDown |
|---|---|---|
| `hit` | already used | no target |
| `missed` | already missed | no target |
| `future_outside_pass` | earlier than passing window | TooEarly |
| `early_pass` | within early passing window | hit eligible |
| `on_time` | at scheduled time | hit eligible |
| `late_pass` | after scheduled time but within late pass window | hit eligible |
| `expired_pending` | after late pass window, but Miss not yet marked | TooLate / subsequent Miss processing |

Use `classify_timing` and `timing_windows` exactly as the
physical evaluator, including BPM/pitch/difficulty/scale. Remaining time
is **signed**. In particular, a late hit does **not** become an ordinary
zero-millisecond 200ms-future-event prediction. No new Timing regression
head has been trained and no model or legacy label function is modified.

All chart target indices and timing-window details stay evaluator-only;
they are **never** added to the 263D visible policy input.

## Part II: Train20 Teacher Safety Replay

Replays **the exact existing Expert/Teacher** on the frozen Train20 role,
logs sampled 10ms actuator commands, real 1ms physical KeyDown events,
TooEarly events including Overload gauge before/after and failure time,
and each episode's last twelve control frames with target contract,
motor position/velocity/pressed state and Teacher commands.

This is **read-only** and **Train20 only**, not another experiment with
new hyperparameters. It verifies exact Hit/Miss/TooEarly/Overload/KeyDown
agreement with the existing v2.2 Train report. Mismatches fail closed.
Commands and KeyDown events are separately timestamped; actuator inertia
means a KeyDown is not necessarily caused by the command in its same frame.

New report: `diagnostics/prestige1/stage_c_v23_teacher_safety.json`
(no replacement of existing file).

## Local Windows commands

```powershell
git pull --ff-only
uv run pytest -q tests/test_prestige1_target_contract.py
uv run python -u scripts/run_prestige1_teacher_safety.py `
  --config configs/training/prestige1.toml
```

The v2.2 Train-only report must still exist at
`diagnostics/prestige1/stage_c_v22_train_alignment.json`.

**No training**, no old-Student warmstart, no Stage A gate override,
no Validation selection, no FINAL evaluation and no modification to
original pretrained/Control/Prestige checkpoints.

After this audit, the next candidate is a **separate, Train-only
preregistered** comparison of explicit signed late-state supervision
and actuator-safe Motor BC. Neither correction is automatically accepted.
