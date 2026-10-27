# DMDOGO v0.9 — Phase H0.12: synthetic falsification of early release

## Why

H0.11's five-Chunk stop preserved all seven preselected improvements
in 22 original Train flagged seams and reduced forced-slot replacements
from 82,026 to 448. That is an **in-sample** finding: the stop horizon
was selected using H0.9's seven winners. Its safety cannot be claimed.

H0.12 investigates the possibility of **late reversal**: a protected
alternative fails to reach rank 1 in its first five Chunk windows
but becomes competitive later, after an early-release policy has
already stopped reserving its slots.

## Prespecified independent challenge

- No saved Train charts are loaded. No Validation20, FINAL, human
  demonstration, Motor or timing observation is processed.
- Generate a fixed, deterministic set of **32 synthetic event sequences**.
  Generation uses documented base seed `20261028`, trial-specific
  seeds `20261028 + 7919 * trial_index`, and four modes: constant,
  accelerate, alternating, regime_shift. Sequence lengths are
  64, 80 or 96 events; some inputs include two-key chord events.
- Two positions per feasible sequence are selected before examining
  future costs: Chunk 1, and Chunk max(2, chunk_count // 2 - 5),
  provided at least five further chunks exist.
- At each seam, use the **frozen selected predecessor** only. Among
  viable non-original alternatives, choose the first expanded-rank
  candidate **outside natural top 24**. If all are naturally inside,
  choose the best non-original candidate. The same candidate is
  compared under full quota8 and H0.11's fixed 5-Chunk release.
- This is a wider stopping-rule falsification, **not** a study of
  genuine quick-jump correction: the candidate need not be non-quick
  at a flagged original quick seam.
- Keep total width 24, unchanged cost terms, gesture grammar and
  minimum 80ms abstract-axis reuse.

Per selected candidate, independently recompute (a) full quota8
using H0.8 code; (b) H0.9 per-window rank trace with exact final-cost,
winner identity and rescued-slot parity; and (c) H0.11 online release.
If H0.11 does **not** release, its full plan and rescue history
must be identical to the full-quota8 plan.

### Report categories

A **late rank-1** candidate is one whose first rank-1 occurrence is
at least five Chunk windows after joining. It may not be a better
terminal winner. An actual **stop-rule counterexample** is a case
where the adaptive final winner's cost is greater than the
full-quota8 final winner's cost by >1e-7. A stricter case
**missed improvement versus v0.8** requires full-quota8 to beat
v0.8 while the adaptive result fails to do so.

Counts of infeasible frozen synthetic sequences, missing alternatives,
insufficient windows and non-surviving tagged lineages are reported
explicitly. Those are **not** positive evidence of safety.

All events (including due_s and chord_size), source generation
metadata, candidate ranks, outcome deltas and first/last five
window observations are deterministic; per-trial input SHA256
is also retained. No candidate is chosen after inspecting its
downstream performance.

## Windows PowerShell

Stay on develop/dmdogo, leave dirty local
`configs/training/prestige1.toml` and all existing artifacts alone.
Use only ff-only pulls, never reset/rebase/stash/clean or overwrites.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h012_synthetic_counterexamples.py

# Optional four-trial smoke; its output path differs from the full run.
uv run python scripts/audit_dmdogo_v09_h012_synthetic_counterexamples.py `
  --trials 4 `
  --output reports/dmdogo-v09-h012-synthetic-smoke-r1.json

# Predeclared full 32-trial experiment.
uv run python scripts/audit_dmdogo_v09_h012_synthetic_counterexamples.py `
  --trials 32 `
  --output reports/dmdogo-v09-h012-synthetic-r1.json
~~~

Each output path is **create-once**, reruns must use an unused revision.
A full run can require dozens of independent full-chart beam searches;
no physical Motor or GPU model inference is used.

## Interpretation

A discovered counterexample falsifies the universal five-Chunk
release claim **for the abstract simulator** and should be
treated as a reason to keep v0.8 frozen, not to immediately
retune the threshold on those same synthetic trials.
Absence of counterexamples in finite synthetic trials establishes
only the results of that search, not general correctness.

Neither synthetic lower-cost fingering nor the absence of
new quick/rapid proxies implies human-likeness or physical validity.
