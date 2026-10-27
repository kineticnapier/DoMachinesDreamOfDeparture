# DMDOGO v0.9 — Phase H0.10: all 22 flagged seams under one frozen-width quota-8 experiment

## Objective

H0.8 and H0.9 studied **seven** original quick-jump seams previously
selected for favorable results in the H0.6 forced-suffix experiment.
They cannot establish how the identical policy behaves on the remaining
15 original flagged seams. H0.10 applies that same one-target-at-a-time
quota-8 rule to **all 22 H0.6 seams** on the same three verified Train charts.

No extra learning, optimization, or hyperparameter search is performed.
The original Planner, Timing, Motor, runner, chunk segmentation, scoring,
beam width 24, and abstract minimum 80ms reuse remain unchanged.
H0.10 is a read-only counterfactual, NOT v0.8 behavior.

## Prespecified comparison

- Every saved H0.6 flagged pair is tested once, independent of its prior outcome.
- Use exactly H0.8 quota 8 among the same total width 24.
- Preserve source H0.6 labels in separate strata:
  seven previously lower-cost **forced suffix** cases and fifteen
  not previously lower-cost forced-suffix cases.
- Recompute entire original Oracle sequence and freeze-plan identity
  from saved chart+verified source before individual trials.
- Fail closed if any seven known-improved quota-8 result differs from
  the saved H0.8 output, or if H0.8 quota=1 differs from H0.7.
- The newly evaluated fifteen cases have NO required success rate.
  Worsening, tie, missing tagged survivor or new style flags
  are all legitimate outcomes and must be reported.

### Per-target measurements

- Final *beam winner* and *best protected descendant* reported separately.
- Cost difference versus the exact original v0.8 final heuristic.
- Terminal survival and protected/global winner identity.
- Number of rescued windows and displaced non-protected beam slots.
- Whether the original target quick flag was removed, as well as
  all added/removed quick/rapid flag **locations**.
- Changed event-axis assignments, abstract axis reuse under 80ms,
  minimum reuse and hand-use counts.

Summaries are provided for the seven familiar cases, the remaining
fifteen, and all twenty-two. The counts describe **separate experiments**.
Do not sum independent changes into one simultaneously improved chart.
Neither the 15 cases nor the complete 22 cases constitute a holdout:
they were all originally selected for H0.1 quick flags.

## Windows PowerShell

Stay on develop/dmdogo. Preserve local modified prestige1.toml,
untracked reports and experimental artifacts. Never reset, rebase,
stash, clean, or overwrite old revision outputs.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h010_all_quick_seams.py

uv run python scripts/audit_dmdogo_v09_h010_all_quick_seams.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --h07-report reports/dmdogo-v09-h07-shadow-lane-r1.json `
  --h08-report reports/dmdogo-v09-h08-protected-quota-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h010-all-22-quota8-r1.json
~~~

Output creation is exclusive; use a fresh revision path for a rerun.
This search includes 22 separate full-chart abstract beam replays
and may consume considerable local CPU and memory. The 7 familiar
runs are intentionally recomputed to verify exact H0.8 continuity.

## Interpretation

Even if the entire set improves under the heuristic, this is not
evidence that physical Motor, timing, or human-fingering behavior
improves. Extra quick/rapid flags are not necessarily physical
failures, and lower objective score is not a human ergonomic metric.
Further work must isolate regressions and define nonselected
holdout conditions before any production change.

**Physical Motor, Validation20, FINAL and any training are not run.**
