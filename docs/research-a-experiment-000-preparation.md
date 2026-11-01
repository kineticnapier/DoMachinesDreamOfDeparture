# Research A — Experiment 000 preparation (NOT preregistration)

Status: infrastructure draft. No study execution, no RL training, no bodily
calibration, no parameter freeze. The user-provided preregistration remains
unapproved; this branch does not redefine success criteria.

## Reuse existing physical system

- `src/dmdod/motor/body.py`: `TwoFingerBody`
- `src/dmdod/motor/keyboard.py`: `TwoKeyKeyboard`
- `src/dmdod/motor/env.py`: `MotorEnv`
- `scripts/a000_two_finger_replay.py`: deterministic, teacher-free replay.
- `tests/test_research_a000_two_finger_determinism.py`: CPU consistency checks.

`MotorEnv` uses the existing personal blue-switch profile and physics
substeps. These are **provisional physical parameters**, not measurements
proving a human-equivalent body.

## Scope of current tests

- Same seed and 320 fixed action commands yield identical physical events,
  observations, and diagnostic floating-point traces in two fresh environments.
- Episode order and varying seed cannot contaminate initial state.
- Two separate Python processes with distinct `PYTHONHASHSEED` values
  reproduce the same CPU trace.
- Nonempty keydown/up events are observed so the hash comparison is not
  vacuously about an idle-only body.
- There is no chart, target schedule, policy training, reward, or teacher
  in the deterministic replay harness.

CPU exactness is the first *proposed* target; broader OS/CPU/CUDA consistency
criteria are not yet approved. Passing the above tests is a **smoke check**,
not Experiment 000's official completion.

## Next design decisions before experiment 001

1. Is the two-finger physical configuration same-hand or opposite-hand?
2. Which independent controls are permitted to measure idealized single- and
   two-finger speed limits without giving those controls to the student?
3. Define press judgment window and sustained-trial duration, repeated
   trial count, and statistical uncertainty.
4. Fix the relative next-target-time/body-state observations and disallow
   absolute time. This is a timing aid, not a fingering teacher.
5. Define separate untouched validation timing/start-delay splits.
6. Approve the preregistration and freeze a precise code revision BEFORE
   running experiment 002 or selecting favorable thresholds.

No modifications to Stage A, Prestige I checkpoints, motor constants, or
existing research-B PRs.

### Optional reproducibility command

```powershell
python -m pytest -q tests/test_research_a000_two_finger_determinism.py
python scripts/a000_two_finger_replay.py --seed 123 --actions 320
```
