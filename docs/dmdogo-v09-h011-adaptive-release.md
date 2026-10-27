# DMDOGO v0.9 — H0.11 Quota-8 diagnostics and exploratory early release

## Research question

H0.10 applied the same width-24, eight-slot protected beam to all 22
original H0.1 flagged seams. All 22 tagged lineages survived to the
end; only seven became cheaper than frozen v0.8 and were selected as
winners. Those seven needed **13 rescue windows / 56 displaced slots**
across independent runs; the other fifteen needed **10,295 windows /
81,970 slots** but did not alter any final event assignments.

H0.11 examines the six untrained heuristic terms and per-chunk tagged
versus untagged cost gaps, then performs an independent, online-only
counterfactual test of releasing protection when there is little early
evidence the tagged lineage is competitive.

## Method

1. Load the H0.10 results plus H0.6 and H0.8 provenance, the three
   verified Train charts, unrounded Oracle onsets, and H0.1 audit.
2. Independently reconstruct **all 22 original H0.10 quota-8 beams**
   by using the existing H0.9 per-window tracer; require exact final
   costs, winner lineage, best tagged rank, rescue-window IDs,
   displaced slots and quota underfill to match the saved H0.10
   result. Fail without writing if any discrepancy occurs.
3. For every chunk at and after the seam, observe best tagged-minus-
   best untagged cumulative objective. Record the first rank-1
   chunk, minimum/maximum gap, sampled gap checkpoints, and final
   difference in all **six** original objective terms.
4. Re-run all 22 independently using the *same width24, quota8*
   for the first five join-inclusive chunks. If no tagged candidate
   has ranked first in the expanded population by the end of the
   fifth, release the quota **starting with the next chunk**.
   After release, retain ordinary score-sorted top24; tagged paths
   are not forcibly removed and can survive naturally.
5. Recompute final winner, best tagged survivor, flags, changed
   event assignments, rescue windows and displaced slots. Compare
   each with the full quota8 H0.10 outcome.

### Important separation of evidence

The **five-chunk** horizon was selected after observing the seven
known successful cases in H0.9, where the tagged lineage first
ranked #1 two chunks after the join. It is an **exploratory,
in-sample heuristic**, not an externally validated safe stop.
The seven successes may fail to transfer to unrelated charts.

Quota-8 and adaptive runs use the original untrained v0.8
candidate grammar, cost, 80ms abstract reuse and total width24.
Only beam pruning/admission differs from the frozen v0.8 search.

Reported 'saved slots' means displaced non-tagged hypotheses
avoided relative to full quota8, **not** a CPU speedup or lower
number of candidate evaluations. Independent charts and targets
remain separate experiments; sums are accounting statistics.

### Failure and adverse outcomes

If a known winning case becomes worse, or a nonwinning case
changes the final winner after early release, the output reports
that divergence; it does not silently reject the run.
Unreleased cases must reproduce the exact H0.10 outcome.
Saved H0.10 baseline mismatches always abort before output.

Do not interpret lower cost, removed Quick Jump proxies or a
quota-release trigger as evidence of human ergonomics or physical
Motor feasibility. This experiment uses **only Train** artifacts;
no Validation20, FINAL, physical Motor or training.

## Reproduction — Windows PowerShell

Stay on develop/dmdogo. Do not modify or discard dirty
configs/training/prestige1.toml, untracked artifacts or prior
result JSONs. Do not reset, rebase, stash, clean or force push.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h011_adaptive_release.py

uv run python scripts/audit_dmdogo_v09_h011_adaptive_release.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --h08-report reports/dmdogo-v09-h08-protected-quota-r1.json `
  --h010-report reports/dmdogo-v09-h010-all-22-quota8-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h011-adaptive-release-r1.json
~~~

Create-once output: choose a new revision filename on repetition.
This requires two abstract full-chart searches per targeted case
(an independent baseline trace and an adaptive replay), and may
consume considerable local CPU. No external model API or Codex is
needed.

## Next step

If the in-sample experiment preserves the seven lower costs and
reduces displaced slots, the next question is what evidence could
justify a genuinely prospective rule in a separate Train-only
development cohort, rather than retrofitting this 22-case sample.
Do not change production or frozen v0.8 without authorization.
