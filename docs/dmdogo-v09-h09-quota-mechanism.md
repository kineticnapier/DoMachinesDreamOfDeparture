# DMDOGO v0.9 — Phase H0.9: quota-8 internal mechanism audit

## Context

H0.8 showed that the same frozen-width-24 beam can recover all seven
preselected H0.6 lower-cost plans when at least eight descendants of
one non-quick target are retained, whereas quotas 1 and 2 recovered
none and quota 4 recovered three. Across the seven separate quota-8
experiments only 13 windows required rescue, displacing 56 slots.

These data do NOT establish that preserving eight paths makes the
planner human-like, or that eight is an optimal/default hyperparameter.

## Study (no implementation changes)

H0.9 replays exactly the already-saved H0.8 quota-8 experiments.
It does NOT edit frozen v0.8 Planner, Motor, Timing, runner,
hyperparameters, beam width, or H0.8's intervention.

For each Chunk at and after the target seam it records:

- Feasible tagged and untagged expansions before pruning.
- Tagged descendants naturally in the ordinary top 24, versus
  tagged descendants retained after the quota intervention.
- How many untagged regular-top-24 candidates are displaced.
- Best tagged candidate's rank in the full expanded list and
  best retained rank after pruning.
- Best tagged and untagged expanded costs and their difference.
- First Chunk where eight tagged descendants naturally survive;
  first Chunk from which this remains true until the chart end;
  and first Chunk where a tagged candidate ranks first.
- All rescue windows, underfilled-quota windows, and terminal
  tagged/global winner scores.

The word *natural* is relative to the H0.8 intervention-altered
beam population **before the current Chunk's quota adjustment**.
It does NOT mean survival in the unrestricted frozen v0.8 beam.

H0.9 also aggregates newly created and removed quick-depth-jump
and rapid-single-hand-run **locations**, not purported identities.
Such flags are unvalidated style heuristics, not biomechanical
constraints. The seven experiments are independent and overlapping;
their flag counts cannot be added to infer one corrected chart.

## Fail-closed protocol

Validate all original Train manifest hashes and verified physical
provenance without executing Motor. Recompile original unrounded
Oracle events. Require matching H0.1, H0.6, H0.7 and H0.8 source
identities and the saved H0.8 quota-1/H0.7 per-target comparison.
Before accepting H0.9 window results, independently re-run quota 8
using the unchanged H0.8 audit and require exact saved per-target
output match, then compare the independent window trace against
that H0.8 path: final cost/rank, rescue windows, displaced slots,
tagged survival and underfilled-quota count must be identical.
Mismatch aborts before creating the report.

## Windows PowerShell

On develop/dmdogo, preserve all dirty and untracked files including
prestige1.toml. Stop on pull conflicts; never reset/stash/rebase/clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_v09_h09_quota_mechanism.py

uv run python scripts/audit_dmdogo_v09_h09_quota_mechanism.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --h07-report reports/dmdogo-v09-h07-shadow-lane-r1.json `
  --h08-report reports/dmdogo-v09-h08-protected-quota-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h09-quota-mechanism-r1.json
~~~

The output is create-once; reruns require a new revision filename.
Per-window detail makes the JSON much larger than H0.8, and the
independent double reconstruction may be computationally expensive.

## Interpretation

If tagged descendants naturally dominate after early rescue, the
low subsequent rescue count is consistent with a self-sustaining
population in this experimental policy. If domination occurs much
later, inspect the costs, survivors and rescue windows instead of
attributing all benefit to early self-sustainability.

No physical Motor, Validation20, FINAL or human training occurs.
Conditional lower heuristic objective alone does not establish
physical feasibility or human-like fingering.
