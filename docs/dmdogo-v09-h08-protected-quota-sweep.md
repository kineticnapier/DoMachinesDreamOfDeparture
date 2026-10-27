# DMDOGO v0.9 — H0.8 Fixed-Width Protected-Lineage Quotas

## Objective

H0.7 kept one non-quick descendant alive to the end in all seven
preselected Train cases, but found zero improvements over the frozen
v0.8 objective. H0.8 asks whether preserving multiple descendants
of the same forced non-quick candidate changes that result.

## Frozen and changed factors

- The original v0.8 Planner and score weights, candidate grammar,
  chunk segmentation, minimum abstract axis reuse (80 ms), and
  original width 24 are never edited.
- H0.8 is a separate abstract counterfactual experiment, not
  a change to v0.8 source or a production planner.
- Only the tagged-lineage *minimum retained quota* changes:
  1, 2, 4 or 8 out of the same 24 beam slots.
- Keep normal score-ordered top 24 when they already contain
  at least quota tagged hypotheses. Otherwise replace the
  most expensive untagged survivors with the best tagged
  candidates that fell outside the normal top 24.
- Quota is an **at-least** rule, not an exact or maximum allocation.
  If normal top 24 naturally contains more tagged hypotheses,
  those remain; if fewer than quota are feasible, no paths
  are fabricated. All tie sorting is original stable score-only.
- On the target window there is exactly one tagged candidate,
  so quotas >1 can first matter on subsequent windows.

## Provenance and fail-closed controls

Only the seven cases already selected in the saved H0.6 report
are evaluated. The script SHA-verifies all three original Train
charts, checks physical-run provenance through H0.1, recompiles
exact unrounded Oracle events, and calls the unchanged v0.8
Planner to verify the original path, event count, candidate count
and final rounded cost for each independent sweep.

Quota 1 is compared against the **saved H0.7 file** at the
per-target level, including final costs, quick/rapid flags,
changed event assignments, rescue windows and original ranking.
Aggregate counts are also compared. If any H0.7 field differs,
the script fails and creates no new findings.

Each quota/target experiment is independent: seven cases ×
four predeclared quotas. Results cannot be added together
to infer a combined chart improvement.

## Reproduce (Windows PowerShell)

Use branch develop/dmdogo. Preserve all dirty/untracked
files, including local prestige1.toml. Stop on pull conflicts;
never reset, rebase, stash, clean, or force-push.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h08_protected_quota.py

uv run python scripts/audit_dmdogo_v09_h08_protected_quota.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --h07-report reports/dmdogo-v09-h07-shadow-lane-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h08-protected-quota-r1.json
~~~

Result is create-once. A repeat requires a new output filename.
Offline CPU abstract planning may be substantial: each of 28
counterfactual sweeps covers much of an original full chart.
No physical Motor, Validation20, FINAL, training or inference.

## Reading the outcome

The aggregate reports terminal survival, counts of cheaper
protected hypotheses, counts of cheaper globally selected
beam winners, and number of protected winners. It separately
counts *rescue windows* and total displaced non-protected
slots, since at quota >1 one window can replace several slots.
Per-target records include new quick/rapid locations,
axis reuse metrics and how many assigned events differ.

Even if quota 8 recovers one of H0.6's earlier improvements,
the cost is an **untrained six-term heuristic**, not human
comfort or physical success. Preserving eight tagged routes
necessarily reduces beam coverage for other lineages and
may regress on unrelated charts. The selected seven cases
are not a holdout cohort. A physical decision requires
independent approval and separately planned evaluation.
