# DMDOD Prestige I — Stage C v2.7 Global Motor-Aware Teacher Annealing

## Motivation and frozen reference

Stage C v2.4 established Train20 SAFE20, Hit850/850, Miss0,
TooEarly0, Overload0, with Hit-only MAE 58.957ms. Stage C v2.6
reduced Hit-only MAE to 5.135ms but lost nine targets:
841/850 Hit, Miss9, TooEarly0, Overload0. **v2.6 failed its
physical safety gate and MUST NOT be copied as Student targets.**

The earlier v2.6 controller optimized 10ms counterfactuals using
a short constant-action forecast. The v2.7 controller instead
accepts/rejects a complete chart schedule using the *real full
1ms-physics replay result*. No 10ms local forecast decides validity.

## What exactly is optimized

The frozen, deterministic v2.4 Teacher (including the unchanged
v2.4 actuator safety shield) is replayed on each Train anchor and
its **executed 8-Key 10ms command trajectory** is captured.

That whole-episode command sequence becomes the **feasible initial
solution**. It must independently reproduce the exact frozen
v2.4 Hit events, timing summary and per-anchor physical counts or
the experiment fails before optimization.

The search space is deliberately bounded and reproducible:

- Every contiguous per-finger +1 command run is an independent
  timing variable, provided it is at least two frames long.
- Each run may be *left-trimmed* by 0..min(9, length-1) frames:
  removed early +1 frames become neutral 0 for the same finger.
  Original end frame and subsequent release/brake -1 commands
  are retained, so no new positive commands are introduced.
- Neighbor proposals adjust one run or a *contiguous group*
  of two or three runs in physical time order. Different
  fingers can be adjusted together, permitting coupled changes.
- Routing to different fingers is **not yet searched**.
  The scheduled fingers follow v2.4. This is a first constrained
  schedule optimization, not unconstrained global optimality.

The objective is **full-chart mean absolute signed Hit error in ms**,
but it is defined only for *physically valid* trajectories.
Candidates with any Miss, TooEarly, Overload, unjudged target, or
extra KeyDown are ineligible **at every temperature**.

The SA Markov chain can accept a slightly *worse feasible* MAE with
probability exp(-(candidate_mae-current_mae)/T), while a separate
best-so-far strictly improves and can never regress relative to
v2.4. The initial schedule is always a safe fallback.

One fixed, reproducible exploratory protocol (no hyperparameter
sweep): 96 candidate proposals per anchor, delays <=9 ticks,
T=2.0ms geometrically cooled to 0.15ms, one seeded pseudorandom
chain per anchor. All search settings are logged.

## Physical replay cost and correctness

All trial decisions use *true full-chart* 1ms physics and evaluator
judgements. For speed, snapshots of the complete environment
are saved at 100-control-frame boundaries along each accepted
trajectory. Each candidate starts from the latest checkpoint at
or before its earliest altered command, then simulates the entire
remaining episode until true termination. It does **not** reuse
an unchanged suffix based only on estimated timing.

Every anchor's final best schedule is then independently replayed
from frame zero without any parent checkpoints. Physical results
and full Hit-event records must match the cached evaluation exactly.
The protected v2.4/v2.5/v2.6 reports and all checkpoints must
retain their exact pre-run SHA256 values.

**Important limitation:** "global" here refers to the *objective
and acceptance criterion across the entire chart*, not to a
mathematical guarantee of finding the globally optimal schedule.
The feasible-only SA can still be trapped in isolated regions of
the safe scheduling landscape.

## Outputs and gates

The runner writes two **non-overwriting** files under
\`diagnostics/prestige1/\`:

- \`stage_c_v27_safe_teacher_plans.json\` — explicit verified
  8-Key action tape for all 20 Train anchors, plus delay schedule.
- \`stage_c_v27_global_annealing.json\` — baseline comparison,
  full physical and Hit timing metrics, per-anchor diagnostics,
  SA proposal counts, hash of the saved action-tape artifact
  and frozen input hashes.

It refuses to start if either output already exists. The first
write is the verified action tape, followed by its hash-bound
report. If interrupted between writes, preserve the surviving
file and diagnose rather than deleting or overwriting it.

Required physical gate: Train20 SAFE20, Hit850/850,
Miss0, TooEarly0, Overload0, KeyDown850, unjudged0.
A separate metric checks whether full-safe MAE improves over
v2.4. Even a pass is *not* automatic Student-training authorization.

The Stage A original gate remains FAIL, \`may_run_stage_c: false\`.
Stage C v2.7 is **Train-only exploratory Teacher research**,
not an override. No Validation/FINAL runs, no Student inference,
no Student model load, no optimizer update, and no checkpoint writes.

## Windows local execution

Use the existing branch \`develop/v0.1-simulator\` and preserve all
uncommitted changes (no reset/stash/clean). Confirm you have
the fixed v2.3-v2.6 diagnostics already generated.

\`\`\`powershell
git status --short
git pull --ff-only

uv run pytest -q \`
  tests/test_prestige1_actuator_teacher.py \`
  tests/test_prestige1_teacher_quality.py \`
  tests/test_prestige1_timing_teacher.py \`
  tests/test_prestige1_global_annealing.py

# After green CPU tests; Train20 only:
uv run python -u scripts/run_prestige1_global_annealing.py \`
  --config configs/training/prestige1.toml
\`\`\`

The trace and safe plan files are experimental **Teacher** outputs
and must not be presented to the Student as privileged future
target indices, schedules or target timestamps.
