# DMDOGO v0.9 — Phase H0.2: Chunk-seam counterfactual diagnostics

## Objective and inherited observations

H0.1 found 22 quick same-hand depth jumps, all at v0.8 gesture-chunk seams.
Of 109 rapid single-hand runs, 48 were within chunks and 61 crossed boundaries;
10 quick jumps occurred inside rapid runs. These are **review flags**, not
bad-fingering ground truth.

H0.2 separates two questions:

1. **Relabel only.** Keep the original 10,231 event axes and times fixed,
   but replace the chunk boundary labels within a two-chunk region. Does a
   quick-jump flag change? (It cannot: the H0 detector is independent of
   boundary labels.) How many rapid runs now cross the new labels?
2. **Independent local replan.** Keep all original events outside a pair of
   chunks pinned, generate new local gestures from the frozen **read-only**
   v0.8 candidate grammar and examine what alternative chunk partitions
   permit. Enforce the abstract 80ms same-axis-reuse constraint across the
   entire chart. This is a **surrogate local optimizer**, not the original
   v0.8 global beam and not the physical Motor.

The study tests all legal 2- or 3-chunk repartitions, with chunks of 1–4
notes, **excluding partitions that retain the original flagged seam**.
For 4+4 windows, a shifted single seam would violate the four-note limit;
3-chunk repartitions (e.g. 2+4+2) are therefore considered. Those also
change the **number of chunks**, a separate confound. All viable variants
are reported; the lowest surrogate-cost variant is merely an exploratory
summary, not a statistically unbiased treatment estimate.

This is **targeted conditional analysis of the 22 selected quick flags**,
not a population estimate or proof of causation. There is no selection of
BPM, thresholds, or neural model weights to improve a score.

## Contents

- `scripts/analyze_dmdogo_v09_h02_boundary_cf.py` — independent read-only
  analysis, reuse constraint audit and create-once JSON output
- `tests/test_dmdogo_v09_h02_boundary_cf.py` — focused synthetic tests

## Preconditions and command

Use the existing local Train 3/3 results, verified r4 report and H0/H0.1
reports. The H0.1 report is checked against **recomputed** H0.1 from
verified-source provenance before any counterfactuals are generated.

```powershell
uv run pytest -q tests/test_dmdogo_v09_h02_boundary_cf.py

uv run python scripts/analyze_dmdogo_v09_h02_boundary_cf.py `
  --manifest "reports/dmdogo-v08-train-selection-r2.json" `
  --verified-report "reports/dmdogo-v08-train-report-r4.json" `
  --h0-report "reports/dmdogo-v09-h0-train-style-r1.json" `
  --h01-report "reports/dmdogo-v09-h01-train-conditions-r2.json" `
  --results-root "artifacts/dmdogo-v08/train-generalization-r2" `
  --output "reports/dmdogo-v09-h02-boundary-cf-r1.json"
```

No v0.8 Planner/Timing/Motor/runner edit, physical replay, training,
Validation20 or FINAL run is authorized. New output uses file creation mode
`x` and **will refuse to overwrite** an existing result. Original repository
branch and uncommitted changes including `prestige1.toml` must be preserved.

## Report fields and interpretation

Per flagged seam, the JSON includes original boundary sizes, number of
candidate partitions, same-boundary surrogate feasibility, original quick
flag, fixed-axis rapid runs crossing labels, and results for all feasible
local replans. It records candidate counts, chosen axes, abstract reuse,
number of differing assignments, whether a local quick flag remains and
rapid-run count change. Feasibility here means **abstract 80ms reuse legal**
under pinned outside decisions, not physically verified KeyDown/Hit.

**Comparison cautions:** the surrogate beam has different exposure context
from the original global v0.8 beam and cannot be assumed to reproduce the
frozen assignment; only compare alternative and baseline runs of **the
same surrogate**. Proposal search introduces selection bias. Grammar changes
when chunk lengths change. Passing the abstract reuse test does not ensure
Motor eligibility, no Miss, or human-like style. If no feasible partition
exists, report this explicitly rather than silently treating it as no effect.
