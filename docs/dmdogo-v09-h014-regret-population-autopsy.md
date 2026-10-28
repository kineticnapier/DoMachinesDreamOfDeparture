# DMDOGO v0.9 — H0.14: Exact autopsy of H0.13 release-regret cases

## Research question

H0.13 found two **real abstract-search counterexamples** to
the fixed five-Chunk release rule in the Discovery procedural cohort:
trial IDs **4115** and **4127**. The full protected quota-8 beam
found a lower final frozen-heuristic cost than the original v0.8
schedule, but early release lost that improvement. Evaluation with
a separate procedural seed cohort found no regret in its 64 targets.

These discoveries invalidate the universal claim that five-Chunk
release always preserves the full-quota8 final objective; they do
not establish failures of physical Motor or human fingering.

H0.14 is an **explanatory** replay of exactly those two known
failures. It is NOT tuning the rule, finding a new threshold,
or estimating an unbiased failure rate.

## Reproducibility

- Require H0.12's complete 32-trial source report with exact
  format, trial IDs, generator SHA256 and recomputed aggregate.
- Require H0.13's source SHA256, both complete 32-trial
  Discovery/Evaluation cohorts, generated-input SHA256 and
  recomputed aggregate/falsification records.
- Require the regret trial IDs to be exactly 4115 and 4127,
  with no unreported additional Discovery regrets and no
  Evaluation regret in the saved H0.13 experiment.
- Re-run each recorded trial with H0.12's unchanged generator
  and both original quota8 and five-Chunk early-release audits.
- Independently trace the exact, cost-stable expanded and
  retained beam populations for **both** policies, verifying
  terminal winner, best tagged survivor, rescue windows,
  displaced slots and target admission rank against H0.8/H0.11.
  Any mismatch aborts before writing H0.14 findings.

## Diagnostic fields

Every Chunk at and after the selected seam records the:

- Best expanded tagged/non-tagged cumulative score and the
  original six-term score decomposition.
- Best expanded tagged rank and ordinary top-24 cutoff score.
- Number of feasible tagged candidates, number naturally in
  top24, number retained, and slots replaced by quota protection.
- Best retained tagged and non-tagged scores and term totals.
- Complete surviving candidate-index paths for exact checks
  of when the two beam populations first diverged.

The case summary highlights:

1. Grace-period decision Chunk and first subsequent divergent beam.
2. First Chunk where tagged candidates disappear from the
   **adaptive** beam, distinguishing pruning from having
   no feasible tagged successor.
3. First expanded rank1 of the tagged lineage under **full**
   quota8; this does **not** imply it was alive in adaptive.
4. Relevant beam/cost snapshots at the release decision,
   lineage loss and rank1.
5. Final original v0.8, full quota8, and adaptive costs.
6. All six component differences **adaptive minus full** and
   Quick/Rapid changes. Cost differences are associations
   between two terminal histories, NOT causal attribution to
   an isolated term or window.

## Run on Windows PowerShell

Use branch develop/dmdogo, preserve dirty
`configs/training/prestige1.toml` and untracked reports.
No reset, rebase, stash, clean or force push; use ff-only pull.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_v09_h014_regret_autopsy.py

uv run python scripts/audit_dmdogo_v09_h014_regret_autopsy.py `
  --h012-report reports/dmdogo-v09-h012-synthetic-r1.json `
  --h013-report reports/dmdogo-v09-h013-independent-seeds-r1.json `
  --output reports/dmdogo-v09-h014-regret-autopsy-r1.json
~~~

Output creation is exclusive. To rerun, use a new revision filename.
Per-Chunk path keys and scores make the JSON larger than aggregate
reports. All source inputs are deterministic synthetic 64–96-event
sequences, and no physical Motor is run.

## Interpretation

If tagged successors were feasible but all fell below ordinary
top24 and were pruned when protection ended, the immediate
mechanism is explicit **loss of search diversity**. If tagged
successors became infeasible while still retained, the cause
requires separate abstract feasibility analysis. The final
score terms describe why a *particular resulting path* scored
higher, not why the original pruning event is universally harmful.

**H0.11's five-Chunk rule remains falsified regardless of the
mechanism here; no production default change is authorized.**
Validation20, FINAL, real ADOFAI runs, Motor and training remain
untouched.
