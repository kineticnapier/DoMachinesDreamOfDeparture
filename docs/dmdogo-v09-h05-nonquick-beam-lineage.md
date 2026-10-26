# DMDOGO v0.9 — Phase H0.5: non-quick suffix and beam-lineage audit

## Why this study

H0.4 reproduced the frozen original planner's chosen full-chart path and
found a feasible **non-quick** candidate from the same actual predecessor at
each of the 22 selected quick-jump seams. Sixteen such candidates were
within the original 24-wide beam at that step; in no case was the best
non-quick candidate cheaper than the original selected gesture *at that
step*. Six original selected paths were not the highest-ranked expansion
at their selected window. None of that establishes the final outcome of
the alternate paths.

H0.5 answers two distinct questions without changing v0.8 code:

1. **Unrestricted original beam:** Follow the *single cheapest immediate*
   non-quick candidate at each flagged seam. Tag its descendants as they
   pass through the original global width-24 beam with every original
   competitor present. Record if the original beam prunes the candidate
   immediately, first prunes its last descendant later, loses it because
   no legal abstract reuse continuation exists, or retains a descendant
   all the way to the end.
2. **Forced alternative suffix:** Starting from the exact original
   predecessor state and that same non-quick gesture, re-run the frozen
   candidate grammar, score and width-24 pruning for the remaining
   chunks, using only descendants of the forced alternative. Compare
   the conditional final heuristic cost and per-component totals against
   the original complete v0.8 chosen path. This is **not** the original
   unrestricted beam population; other prefixes are deliberately absent.

The script does not optimize new weights, alter the frozen v0.8
implementation, run physical Motor, or request human data. It tests only
one immediate alternative per seam, **not every** possible non-quick
candidate.

## Strict source checking

The script verifies the preselected Train manifest and completed
v0.8 physical result provenance, recompiles original (unrounded)
Oracle onsets from each hashed .adofai chart, and independently
reproduces the H0.4 audit result for each chart.

It refuses to produce output if frozen beam cost, total evaluated
candidate count, final chosen-prefix path, H0.4 event identity or the
original selected predecessor ranking differ. All new findings are
only about verified saved **Train** charts.

Original H0.4 experimental result file is read-only.
A single alternative intervention per selected seam can overlap
another targeted region; the 22 separate forced plans must **not** be
combined into one supposed executable chart.

## Run locally on Windows

Stay on develop/dmdogo. Do not overwrite, stage or discard local
prestige1.toml or any untracked artifacts.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git status --short
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h05_nonquick_suffix.py

uv run python scripts/audit_dmdogo_v09_h05_nonquick_suffix.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h04-report reports/dmdogo-v09-h04-frozen-beam-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h05-nonquick-suffix-r1.json
~~~

The output is created with exclusive create mode. A second run must
use a new output filename instead of overwriting r1.

## Interpretation guide

- **Original-beam immediate pruning:** The best non-quick gesture was
  already outside the width-24 global beam at the join.
- **Original-beam later pruning:** It initially survived among all
  competing global plans but its last descendant later fell outside
  the beam.
- **Original-beam no legal descendant:** All its descendants became
  abstract reuse-infeasible at some later chunk.
- **Original-beam terminal survivor:** At least one descendant survived
  all later original beam cuts; its final cost is measured against the
  actual frozen winner.
- **Forced suffix reached terminal:** A separate limited-beam search
  of descendants of the alternative found a complete abstract
  assignment; compare final cost with the original frozen global
  winner, acknowledging the different search populations.

A forced suffix that scores less than the v0.8 result demonstrates a
**width-limited search discrepancy** for this constrained experiment,
not a proof of global optimality. A larger cost does not establish that
the alternative would feel worse to a human, and no result here
establishes physical Motor feasibility or successful Hits.

All reported costs are from untrained heuristic objective terms.
No body-based fatigue or ergonomic measurement is available.
