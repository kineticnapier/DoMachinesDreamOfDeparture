# DMDOGO H1.4 — verified alternative fingering physical Motor pilot

## Goal and scope

H1.3 automatically compared the 22 Train counterfactuals, labeling
14 as P1 diagnostic review priority, zero as P2 and eight as P3.
That did NOT test whether the new axis assignments succeed on
the original Motor. H1.4 adds one **explicitly authorized isolated
single-candidate Motor pilot**, preceded by a strict read-only preflight.

Never edit the frozen v0.8 planner, physical runner, Motor, Timing,
candidate grammar, original three Train charts, checkpoints, or
saved reports. Do NOT access Validation20 or FINAL.

## Mechanism: no change to frozen source files

The original unchanged physical runner's `run_pattern_arm` invokes
`GestureChunkPlannerV08.plan` once to obtain an offline Oracle
full-chart axis schedule. In a new **child process only**, H1.4
temporarily replaces **that imported Planner symbol** using a
one-shot adapter. The adapter returns the **exact, previously
reconstructed H1.1 full-chart alternative plan** after verifying:

- selected original Train chart SHA256 against manifest;
- complete Git frozen-source digest/working tree via `preflight`;
- existing frozen full-chart physical summary and provenance match
  the originally verified report, without creating new original runs;
- exact H1.1 SHA256 chain and fresh reproduction of H1.3 metrics;
- same compiler/Oracle event indices, onset times, chord sizes,
  original v0.8 per-event assignments and complete Chunk metadata;
- exactly one selected candidate ID, its full event-by-event axes,
  different event indices, chunk segmentation and >=80ms reuse;
- original planner objective versus H1.1 saved original cost;
- new alternative's H0.6 source objective is finite and labeled
  separately from normal v0.8 Beam search output.

The `run_pattern_arm` Motor environment, live eligibility, controller,
lead compensator, local fallback policy and action loop run unchanged.
After exiting the context, the runner's imported symbol is restored.
The child process is bounded by an *outer* wall timeout, covering
chart loading, source checks, Planner and physical simulation.

**Important:** This uses runtime dependency injection, not a new
official v0.8 Planner policy. It must never be counted as an
unchanged v0.8 baseline episode. Its result format is
`dmdogo-h14-alternate-motor-pilot-v1`, NOT the v0.8 physical
format. It does not write a regular `replay-v08.json` or
pretend the original preselected plan was the one executed.

## What counts as evidence

The physical result includes real simulated-Motor scorer stats
(hits, misses, too-early, overloaded, MAE/X when available),
actual KeyDowns, all live decision rows, local replan/deferral
counts, selected-event count and adoption of the alternative
desired axes. **A success with changed physical axes is not
proof the original alternative assignment itself was executed.**

Original historical episode summary and provenance are read and
checked; **they are not re-executed**. An explicit boolean compares
the current environment fingerprint to the stored original.
If they differ, direct original-versus-alternative comparisons
are **not** controlled. The new execution still provides
descriptive alternate-Motor evidence, not human ability.

## Reproduction: two distinct runs, never overwrite

Use branch `develop/dmdogo`, preserve local files and all existing
artifacts. Do not reset, clean, stash, rebase or force push.

First run focused tests:

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h14_alternate_motor.py
~~~

### 1. Preflight (NO physical execution)

This is a lower changed-event-count candidate from ALL MY FELLAS,
chosen as a **first pilot**, not a representative sample.

~~~powershell
uv run python scripts/run_dmdogo_h14_alternate_motor.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h11-report reports/dmdogo-h11-fingering-pairs-r1.json `
  --h13-report reports/dmdogo-h13-automated-quality-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --candidate-id e142af91b178-1089-1090 `
  --output reports/dmdogo-h14-preflight-1089-r1.json
~~~

Inspect the preflight status, especially
`same_runtime_fingerprint_as_original`, candidate ID, chart SHA
and `changed_event_count`. This step never starts Motor.

### 2. Explicit independent alternate Motor simulation

Only after preflight PASS, use **unused output/status paths**.
The `--execute` switch is essential. This run does not overwrite
original Train physical outputs or H1.1/H1.3 reports.

~~~powershell
uv run python scripts/run_dmdogo_h14_alternate_motor.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h11-report reports/dmdogo-h11-fingering-pairs-r1.json `
  --h13-report reports/dmdogo-h13-automated-quality-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --candidate-id e142af91b178-1089-1090 `
  --output artifacts/dmdogo-h14/pilot-1089-r1.json `
  --status-file logs/dmdogo-h14-pilot-1089-r1.status.json `
  --wall-timeout-s 1200 --execute
~~~

The default run is a pure preflight. The explicit execution
uses an isolated subprocess. It writes one result JSON and a
distinct create-once status sidecar. Any failed or timed-out
run is a real negative/incomplete finding; do not silently
reuse the same filename, retune Motor, replace charts or
alter the old source to hide failure.

### 3. Interpret cautiously

Verify `termination_reason`, scorer Hits/Misses, TooEarly,
`physical.local_replans`, deferrals, planned-axis adoption
and which KeyDown axes actually occurred. A complete result
with misses is a complete but **unsuccessful** physical run;
an overload or timeout cannot be counted as playable success.

Even a full hit on the frozen physics is not evidence of
human-preferred fingering, human fatigue model calibration,
or HUD-only Student observability. All simulations use
privileged exact Oracle due times.

## Research gate

Run ONE pilot before committing to the cost of 22 full-chart
physical simulations. If it passes, compare one P1 large-change
case under the same predeclared constraints; only then plan
a broader Train-only census, with independently verified
status and no selective dropping of failures.

Do not use the H1.2 human reviewing tool unless genuinely
desired; no human manual grading is required by H1.4.
