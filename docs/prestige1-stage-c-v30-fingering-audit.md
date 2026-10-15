# DMDOD Prestige I — Stage C v3.0 Offline Fingering Audit

## Purpose

This is a **read-only, Train20-only descriptive audit** of the 850 physically
verified v3.0 Teacher keydowns. It prepares an evidence baseline for the
separate \`FingeringPlanner\` interface. It does **not** optimize/learn
fingering, collect human demonstrations, replay alternative assignments,
or unlock Stage A / Student training.

Input (already produced by the frozen v3.0 Windows experiment):

- \`diagnostics/prestige1/stage_c_v30_finger_phase_annealing.json\`
- \`diagnostics/prestige1/stage_c_v30_safe_teacher_plans.json\`

The audit verifies the exact saved-plan SHA256 and report/plan
provenance, original Stage A FAIL, full Train20 safe850 physical
counters, per-anchor indices and chart IDs, event MAE and every
8-key command's finite ternary values. The output path is write-once:

- \`diagnostics/prestige1/stage_c_v30_fingering_audit.json\`

## What the supplied v3.0 data establishes

| Finger | Hits |
|---|---:|
| left_4 | 3 |
| left_3 | 10 |
| left_2 | 46 |
| left_1 | 373 |
| right_1 | 364 |
| right_2 | 42 |
| right_3 | 9 |
| right_4 | 3 |
| **Total** | **850** |

- 737/850 = 86.71% of all Hits came from the center pair
  \`left_1\`, \`right_1\`.
- Left hand claimed 432 events (50.82%); right hand claimed
  418 (49.18%).
- There are 830 adjacent within-anchor Hit pairs, **61 with
  physical KeyDown separation below 20ms**.
- Among these 61 near-adjacent pairs, **zero reuse the same finger**.
- Across all within-anchor same-finger reuses, the shortest
  observed Hit gap is **104ms** (left_1).
- All 20 Train anchors' source target onset times (reconstructed
  evaluator-side from physical \`time_s - error_ms / 1000\`) had **zero
  pairs of simultaneous target onsets** at the 1µs tolerance.
  Thus this dataset cannot demonstrate learning simultaneous
  chords, even though some physical KeyDown events are close.
- One piece of detailed source behavior: anchor #01 Inferno
  uses all 8 fingers, while many other anchors rely almost
  entirely on the center pair.

The counters describe the **optimized center-first privileged
Teacher's actual actions**, not human play. Observed minimum
same-finger Hit separation is *not* a physical recovery bound;
it may simply reflect the teacher's heuristic or the chart
distribution.

## Interpreting these data before IRL

The timing+motor physical score does not uniquely identify the
"correct" finger for a note. The v3.0 phase optimizer mostly
moves the commands of already chosen fingers; it does not
learn alternate note-to-finger assignments.

IRL on these 850 events alone is at risk of inferring an
arbitrary **center-first preference** as if it were human
optimal fingering. We need either:

1. genuine labelled human keydown demonstrations under
   sufficient note-pattern coverage; or
2. *physical counterfactuals*: from the same visible/current
   motor state, test alternate key allocations without
   leaking chart-truth into the eventual Student inputs.

Known simulator metrics (Hit, Miss, TooEarly, Overload,
KeyDown timing) stay as hard physical constraints; an
optional IRL/preference model would learn *additional*
costs and choice preferences. Evaluate separately:
physical feasibility, finger re-use/recovery, distribution
balance, and unseen-pattern generalization.

The existing \`src/dmdod/planning/causal_contract.py\`
provides the TimingPrediction -> FingeringProposal ->
JointIntent -> MotorController structural API.
The new audit stays solely in the evaluator/training
namespace and does not bridge oracle target time into
that API.

## Run

\`\`\`powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_fingering_audit.py
uv run python -u scripts/audit_prestige1_fingering_v30.py
\`\`\`

The script reads frozen JSON only. No CUDA or actual
1ms physics re-run is needed. It creates a new output
once and refuses to overwrite it. It makes no new
Validation/FINAL selection and does not run a Student.

If a v3.0 audit JSON already exists, preserve it;
do not delete/rewrite it merely to rerun the script.
