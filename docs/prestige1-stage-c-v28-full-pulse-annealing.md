# DMDOD Prestige I — Stage C v2.8 Full-Pulse SA & Physical Feasibility Repair

## Purpose, scope, and protected evidence

v2.4: Train20 SAFE20, Hit850/850, Miss0, Overload0, MAE58.957ms.
v2.6: Train20 Hit841/850, Miss9, MAE5.135ms (unsafe).
v2.7: Train20 SAFE20, Hit850/850, Miss0, TooEarly0,
Overload0, MAE46.519ms. All twenty verified open-loop action tapes
are saved in the exact-hash-binded v2.7 plan artifact.

Stage C v2.8 is an **exploratory, Train20-only, offline Teacher
schedule experiment**, extending the search representation. It does
not change the frozen 263D visible HUD, the 8-key physical dynamics,
the 10ms controller, 1ms simulator, split, checkpoints or Stage A gate.

v2.7 only converted the *beginning* of a +1 pulse to neutral,
shortening the total physical press duration. The v2.8 search instead
**translates a complete +1 command run and its immediate -1 release
tail together**, preserving their durations. Only the contiguous
zero-command gap after that tail may be used to shift a pulse.
The saved schedule still starts from v2.7's independently verified
SAFE850 physical trajectory.

## Search and repair

1. Parse one 8-key action tape into separate per-finger full-pulse
   intervals; each interval is a positive +1 sequence followed by
   any immediately adjacent negative -1 release/brake sequence.
2. Treat right-shift of a whole interval by 0..8 x 10ms as a timing
   variable, capped at the actual empty per-finger gap. Mutations
   change one pulse or up to three successive pulses (possibly
   on different fingers), allowing coupled timing changes.
3. Reconstruct the *entire* 8-key command tape. A proposed schedule
   cannot overwrite another non-neutral finger command. Commands
   are checked as 8-key ternary actions.
4. Replay exact 1ms physics to normal chart termination. Accepted
   solutions must have every playable target Hit, zero Miss,
   zero TooEarly, zero Overload, zero unjudged and no extra KeyDown.
5. When a main proposal misses, identify the **first missed target**
   from the physical evaluator and attempt up to three deterministic
   neighboring full-pulse +/-10ms repairs around that target.
   Each repair is independently subjected to the complete physical
   safety gate. No unsafe state is ever used as an annealing parent.
6. For physically feasible candidates, ordinary SA Metropolis
   acceptance allows temporary deterioration in chart-wide Hit MAE.
   The best-so-far trajectory never regresses from v2.7.
7. Re-run the final best trajectory **from frame zero**, with no
   cached prefix, and require identical physical stats, hit records
   and MAE.

The fixed protocol is 96 main proposals per Train anchor,
temperatures geometrically cooling from 3ms to 0.15ms,
max 8 ticks per pulse, max 3 pulses per coupled mutation, and
max 3 repair attempts per unsafe main proposal. These are one
specified experimental setting, **not a hyperparameter sweep**.

The optimizer reuses v2.7's complete-environment prefix snapshots
and accurate tail replays to reduce cost. Every final saved best
is independently physically evaluated without prefix shortcuts.

"Global" means a *full-chart objective and end-of-episode constraint*,
**not** a guarantee of finding the globally optimal control tape.
Only per-finger pulse timing is searched; finger identity,
pulse amplitudes, motor parameters, and the v2.7 baseline chart
interpretation are frozen. In very dense sections with no idle gap,
full-pulse shifting cannot move the press: a later stage would need
a different action representation or joint rescheduling.

## Hard gates and data boundaries

No Validation or FINAL trajectories, no heldout selection.
No Student model/inference/training, no checkpoint writing.
No change to the original **Stage A FAIL** (\`may_run_stage_c=false\`).
The v2.8 Teacher is **not automatically adopted**, even if a
Train-only gate passes. Any later Student phase needs separate
explicit authorization and a valid Stage A path.

Frozen v2.3-v2.7 reports and v2.7 safe plans are checked against
the provenance, SHA256, Train20 chart identities, physical baseline
and Hit timing records. The v2.7 action tape is replayed under v2.8
from time zero and must reproduce exact prior Hit records.

Generated outputs in \`diagnostics/prestige1\`:

- \`stage_c_v28_safe_teacher_plans.json\` — 8-key time-indexed
  command tapes and moved full-pulse metadata, independently verified.
- \`stage_c_v28_full_pulse_annealing.json\` — per-anchor and aggregate
  physical/MAE statistics, repair rates and best-plan SHA256.

Both files are written once, **without overwrite**. If the
process stops after saving plans but before writing the report,
preserve the partial output and investigate; do not reset or
discard data automatically.

## Windows-side verification

Use existing branch \`develop/v0.1-simulator\`, preserving the local
worktree. This work performs only offline Python/CPU physics.
The existing v2.7 report and plans must already exist.

\`\`\`powershell
git status --short
git pull --ff-only

uv run pytest -q \`
  tests/test_prestige1_global_annealing.py \`
  tests/test_prestige1_pulse_annealing.py

# After passing tests, execute Train20 exploratory SA:
uv run python -u scripts/run_prestige1_full_pulse_annealing.py \`
  --config configs/training/prestige1.toml
\`\`\`

If an unchanged v2.7 action tape already occupies an output path,
the script rejects overwriting before any checkpoint reads.
To diagnose an exception, keep the error trace and the original
artifacts. Never weaken the Train20 provenance or safety checks
to make the experiment pass.
