# DMDOGO v0.9 — H0.13: late-reversal autopsy and independent synthetic seed sets

## Motivation

H0.12 examined 32 synthetic sequences (64 independently targeted
counterfactuals). Its fixed five-Chunk early-release rule recorded
three cases where the **full quota-8 counterfactual** first ranked the
tagged lineage #1 *after* the five-Chunk grace period. They did not
produce a higher final cost under early release. This is evidence of
**late rank reversal**, not of missed terminal heuristic improvement.

The stopping rule was originally selected based on the successful
H0.9 Train cases. Do not tune it on the H0.12 reversals.

## Predeclared H0.13 protocol

Use the same deterministic H0.12 generator, alternative candidate
preselection, frozen v0.8 score, axis reuse and total beam width 24.
No new pattern filter, fitness function or threshold search.

### 1. Historical diagnostic (description only)

- Load H0.12 `reports/dmdogo-v09-h012-synthetic-r1.json`.
- Verify all 32 original trial indices, generator SHA256 inputs,
  configuration, original per-target counts and aggregate statistics.
- Re-run only the original late-rank-1 trials to require exact
  deterministic equality with the saved source report.
- For each original late-rank-1 target, independently replay full
  quota8, per-Chunk ranks, and five-Chunk adaptive release.
- Record the cumulative tagged-minus-untagged cost and retained
  hypothesis populations around the release decision (Chunk +4)
  and the first later full-quota8 tagged rank1.
- Clearly distinguish: a late rank1 in the **full quota8 population**
  need not remain alive in the **adaptive population**. The reported
  late window is *not* a claim that the released search reversed.

### 2. Discovery (new synthetic seeds)

- Trial IDs 4096..4127 by default; per-trial seed remains
  `20261028 + 7919 * trial_id`.
- These IDs cannot overlap H0.12 trial IDs 0..31.
- Run the **original H0.12 audit_one** without modifying candidate
  choice, quota, stop rule, or success threshold.
- Report every attempted, skipped, late reversal, regret and
  missed-v0.8-improvement trial, including full input hashes.

### 3. Independent evaluation (separate synthetic seeds)

- Trial IDs 8192..8223 by default, disjoint from discovery.
- Exactly the same generator and fixed protocol; no learned
  parameter, runtime tuning or switch based on discovery outcomes.
- Report its counterexamples and null results separately.
- The evaluation is seed-disjoint, but **not** a new distribution:
  both cohorts use the same procedural generator. This is not
  independent real-chart validation.

### Counterexample taxonomy

- *Late rank1:* best tagged candidate's first score rank1 in full
  quota8 occurs at offset >=5 after admission.
- *Stop-rule regret:* final early-release beam winner costs
  strictly more than full quota8 by >1e-7.
- *Missed improvement:* full quota8 beats original v0.8, but
  early release does not.
- A single late rank1 **without regret** falsifies the stronger
  informal 'never catches up after five' assertion, but **not**
  the claim of equal terminal cost in that specific case.

Do not select seeds after outcomes. A found regret falsifies a
universal guarantee for the abstract synthetic evaluation; a null
result on a finite cohort does **not** prove safety.

## Reproduce on Windows PowerShell

Use `develop/dmdogo`. Preserve all untracked outputs, dirty
`configs/training/prestige1.toml` and local artifacts. Only
`git pull --ff-only`; never reset, rebase, stash or clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_v09_h013_late_reversal.py

# Small disjoint-source smoke (still includes historical autopsies):
uv run python scripts/audit_dmdogo_v09_h013_late_reversal.py `
  --h012-report reports/dmdogo-v09-h012-synthetic-r1.json `
  --discovery-trials 2 `
  --evaluation-trials 2 `
  --output reports/dmdogo-v09-h013-smoke-r1.json

# Full predeclared 32+32 trial cohorts:
uv run python scripts/audit_dmdogo_v09_h013_late_reversal.py `
  --h012-report reports/dmdogo-v09-h012-synthetic-r1.json `
  --discovery-trials 32 `
  --evaluation-trials 32 `
  --output reports/dmdogo-v09-h013-independent-seeds-r1.json
~~~

Both outputs use exclusive creation. For a rerun choose a
fresh revision filename; never overwrite original results.
Full execution includes independent full-chart abstract searches
and can be CPU-intensive. Codex, original Train/Validation charts,
physical Motor, learned Timing, Training and FINAL are not used.

## Reading results

Compare the `discovery` and `evaluation` aggregate sections
separately. The detailed `*_falsification` records give the
trial/target provenance of any counterexamples. Historical
autopsies are a third descriptive section and must **not** be
counted as newly evaluated evidence.

Keeping the final objective unchanged is only a statement about
the frozen six-term heuristic and the particular finite trials,
not human-like fingering or physical timing safety.
