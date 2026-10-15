# DMDOD Prestige I — Stage C v3.0 Motor-Finger Phase Vector

## Frozen evidence and motivation

The v2.9 Train20 results, uploaded after a Windows run, established:

| Teacher | Hit | Miss | MAE (ms) | Pure Perfect |
|---|---:|---:|---:|---:|
| v2.7 | 850/850 | 0 | 46.519 | 193 |
| v2.8 full-pulse SA | 850/850 | 0 | 28.376 | 462 |
| v2.9 global phase + SA | 850/850 | 0 | 13.120 | 780 |

v2.9 decomposition: the **phase-only** weighted MAE was
**13.684ms**; the additional full-pulse SA improved it by
**0.564ms**. Thus roughly 96% of the total v2.8->v2.9
reduction came from the chart-wide timing phase.

The final v2.9 signed mean error was +3.507ms;
left_1 mean +3.65ms (373 Hit), right_1 mean +3.74ms
(364 Hit). Residual errors differ by note/key.
The chart-level global phase does not adjust relative
timing between different motor fingers.

The v2.9 report and SHA256-bound 8-key command files
are immutable inputs. v3.0 neither overwrites them
nor uses Validation or FINAL.

## v3.0 search protocol: physically constrained per-finger phase

Optimize a single **8-component phase vector**, each
entry in 10ms integer control-frame units. For each
source-active finger, consider offsets of
-20ms, -10ms, +10ms, +20ms, and the current offset.

1. Start from the v2.9 physically verified SAFE850
   8-key motor command tape, independently replayed
   at 1ms to confirm exact per-note Hit timings and
   complete physical counters. No Student model is read.
2. Process active fingers in descending number of
   positive motor frames (ties by frozen finger order).
3. For one finger at a time, independently translate
   its *entire* original waveform column of
   +1/-1/0 10ms commands by the trial offset,
   retaining the other seven finger offsets. Preserve
   entire press durations, release/braking commands,
   and internal idle gaps. All translations are built
   from the immutable original tape, not resampled
   from a previous modified waveform. Reject any
   candidate that would trim an early nonzero command.
4. Independently replay each trial until true chart
   termination under 1ms physics. Keep a trial only
   if it exactly preserves the full-Hit, zero-Miss,
   zero-TooEarly, zero-Overload, zero-extra-KeyDown,
   zero-unjudged physical gate and strictly improves
   the complete-chart MAE.
5. Perform **one deterministic coordinate sweep**.
   No hyperparameter tuning, stochastic seed sweep,
   repeated coordinate passes, or premature use of
   heldout data is allowed.
6. From the feasible per-finger phase-selected plan,
   run **unchanged v2.8 whole-pulse SA with repair**
   for 96 main proposals/Train anchor. Final saved
   plan is independently re-evaluated from t=0
   without cached prefixes and cannot regress
   from v2.9 in Hit safety or MAE.

This explores relative per-finger timing, rather
than adjusting a global time offset. It is *not*
a guarantee of global optimum, and it cannot
repair individual late notes that require different
offsets on the same finger. Per-key waveform phase
is only one more controlled hypothesis about the
large v2.9 residual tail.

## Frozen hard gates

- Repo: \`kineticnapier/DoMachinesDreamOfDeparture\`,
  branch: \`develop/v0.1-simulator\`
- 8-Key, 263D visible HUD unchanged
- 10ms command frame, 1ms physics unchanged
- v2.9 SAFE20/H850 is mandatory fallback
- Train20 only; no Validation or FINAL evaluation
- Stage A FAIL and \`may_run_stage_c=false\` untouched
- no Student training, checkpoint writes, or adoption
- frozen source and old-best SHA256 checked before
  and after the exploratory Train-only experiment
- no overwrite of any v2.3–v3.0 diagnostic artifacts

Outputs (write-once under \`diagnostics/prestige1/\`):

- \`stage_c_v30_finger_phase_annealing.json\`
- \`stage_c_v30_safe_teacher_plans.json\`

Reports include all per-anchor phase vectors,
phase-feasibility breakdowns, final MAE, pulse
SA statistics, physical Hit counters, and a SHA256
of the final verified plan.

## Windows-side tests and Train-only run

\`\`\`powershell
git status --short
git pull --ff-only

uv run pytest -q \`
  tests/test_prestige1_global_annealing.py \`
  tests/test_prestige1_pulse_annealing.py \`
  tests/test_prestige1_phase_annealing.py \`
  tests/test_prestige1_finger_phase.py

# Run ONLY after CPU tests pass.
uv run python -u scripts/run_prestige1_finger_phase_annealing.py \`
  --config configs/training/prestige1.toml
\`\`\`

This is offline 1ms physics with 8-key controls.
The Windows run, performance evaluation, and
independent confirmation remain user-side.
Any failure must preserve the existing diagnostic
and checkpoint artifacts, not weaken the gates.
