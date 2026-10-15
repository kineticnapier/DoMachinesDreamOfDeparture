# DMDOD Prestige I — Stage C v2.9 Global Phase + Full-Pulse SA

## Motivation

The uploaded v2.8 Train20 run reported:

| Teacher | Hits | Misses | MAE |
|---|---:|---:|---:|
| v2.4 shield | 850/850 | 0 | 58.957ms |
| v2.6 short forecast | 841/850 | 9 | 5.135ms |
| v2.7 onset-delay SA | 850/850 | 0 | 46.519ms |
| v2.8 full-pulse SA + repair | 850/850 | 0 | 28.376ms |

In v2.8, 1,799 of 1,920 main proposals were physically
safe (93.7%); 302 repair trials produced 50 fully safe
repairs. Most residual signed errors were still early:
mean signed error **-26.573ms**. The actual saved Train20
action plan SHA256 reported by v2.8 is verified before v2.9
can start.

The residual common offset suggests a control-phase mismatch
in addition to individual pulse scheduling. The specific
causes are not proven by the Train data.

## Fixed hierarchical v2.9 experiment

v2.9 optimizes at two timescales while **always requiring
complete, physically verified episodes**:

1. Load the immutable, previously verified v2.8 Train20
   8-key, 10ms action tape. Re-simulate with 1ms physics
   and require bitwise-equal Hit records, summary, counters
   and termination against the previous saved report.
2. Optimize one per-chart *global phase variable*: shift
   **every** motor command by 0, 10, ..., 80ms. A delayed
   tape is exactly \`[neutral] * ticks + original_tape\`:
   no waveform is shortened, deleted, or altered.
3. Evaluate every phase candidate to episode termination
   under real 1ms motor physics. Miss/TooEarly/Overload/
   extra physical KeyDown candidates are ineligible.
   Choose the physically safe phase with minimum entire
   chart MAE. The 0ms candidate is always present as a
   guaranteed safe fallback.
4. Run the *unchanged* v2.8 local coupled, whole-pulse
   annealing + deterministic physical Miss repair from the
   newly phase-aligned SAFE tape (96 proposals per anchor).
5. Independently replay the final best physical episode
   from frame zero and verify exact event/timing/stat
   identity. Any inferior or physically unsafe best is
   rejected. The original v2.8 plan remains immutable.

This is a single declared search protocol, **not a
hyperparameter sweep**. The 0..80ms grid enumerates a
*control decision variable*. SA temperature, local
proposal budget, motor dynamics, physics, and chart
splits are unchanged.

## Limitations

A uniform phase shift could fail the physical gate
entirely; that is measured and reported, not ignored.
Shifting all actions does **not** make a segment-aware
motor schedule and cannot guarantee a global optimum.
v2.9 also retains all v2.8 restrictions on fixed key
identities and structural pulse movement.

Do not infer that v2.8's early residuals guarantee the
new phase will be safe or improve MAE. Only Train20
execution can answer that. The v2.8 saved plan is a
safe fallback if all nonzero phase candidates fail.

## Protection and outputs

Strict invariants:

- same branch \`develop/v0.1-simulator\`; no new branch
- old checkpoint SHA256 and Stage A evidence unchanged
- Stage A remains FAIL with \`may_run_stage_c=false\`
- Train20 only; no Validation or FINAL selection
- no Student model, training, optimizer, or checkpoint changes
- verified 8-key actions and true 1ms physical replay
- no overwrite of any prior diagnostics or saved tapes

Output files in \`diagnostics/prestige1/\` are write-once:

- \`stage_c_v29_phase_aligned_annealing.json\`
- \`stage_c_v29_safe_teacher_plans.json\`

Report includes per-anchor original/phase/final timing,
phase feasibility, pulse SA/repair counts, physical
safety checks and the SHA256 of the saved plan.

**Windows CPU tests:**

\`\`\`powershell
git status --short
git pull --ff-only
uv run pytest -q \`
  tests/test_prestige1_global_annealing.py \`
  tests/test_prestige1_pulse_annealing.py \`
  tests/test_prestige1_phase_annealing.py
\`\`\`

**Train20 experiment, only after tests succeed:**

\`\`\`powershell
uv run python -u scripts/run_prestige1_phase_aligned_annealing.py \`
  --config configs/training/prestige1.toml
\`\`\`

If the script fails, preserve reports, partial outputs
and uncommitted data. Do not reset, stash or clean.
Do not weaken any physical gate to make it pass.
