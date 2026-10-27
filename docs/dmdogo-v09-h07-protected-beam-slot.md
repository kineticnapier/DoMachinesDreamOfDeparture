# DMDOGO v0.9 H0.7 — fixed-width protected hypothesis audit

## Question

H0.6 found seven independent non-quick forced-suffix plans cheaper
under the frozen heuristic than the original width-24 v0.8 choice.
All seven lineages were pruned in the original unrestricted beam.
H0.7 tests whether reserving one slot among those same 24
hypotheses allows an alternative to survive alongside the normal beam.

## Method

Use only the seven H0.6 preselected improving cases in three
provenance-verified Train charts. Recompile original unrounded
Oracle events and verify original v0.8 Planner output, evaluated
candidate count, exact chunk axes, and final score.

Before each target join, reproduce the original width-24 beam.
At the original selected predecessor, tag exactly the H0.6
non-quick candidate by its axes and gesture family. On each
subsequent chunk, use unmodified v0.8 candidates, cost terms,
80ms abstract reuse eligibility and stable score ordering.
Keep the usual top24, except when they contain no tagged
descendant: then replace the worst kept hypothesis with
the cheapest feasible tagged descendant. Never exceed 24.
If no tagged descendant is feasible, report that dead end.

Record the best terminal tagged path and the global winner;
compare objective costs, changed event axes, quick/rapid
flag locations and axis reuse. Individual interventions
cannot be combined into a valid simultaneous improvement.

**This is an independent experimental pruning rule, NOT
the unmodified v0.8 global beam.** Its one-slot reservation
changes competition, although source Planner and width remain frozen.
Human-like fingering and physical Motor eligibility are unverified.

## Windows reproduction

Run from branch develop/dmdogo. Preserve all local untracked
files and the modified prestige1.toml; never reset/rebase/stash/clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_v09_h07_shadow_lane.py

uv run python scripts/audit_dmdogo_v09_h07_shadow_lane.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h07-shadow-lane-r1.json
~~~

Use a fresh output name for a repeat run. No physical Motor,
Validation20, FINAL, tuning, training or human recordings.

Even a lower frozen objective does not establish preferred
human fingering or actual physical execution success.
