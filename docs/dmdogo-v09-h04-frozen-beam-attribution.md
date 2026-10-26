# DMDOGO v0.9 — Phase H0.4: Frozen beam choice audit

## Purpose

H0.3 found that independent local replanning with unchanged chunk boundaries
already eliminated all 22 selected quick-jump flags. H0.4 traces the *actual*
frozen v0.8 beam objective and pruning history at those 22 joins.

The audit never modifies Planner, Motor, Timing, physical runner or checkpoints.
This is not a human-likeness or causal-effect study.

## Exact replay contract

The script loads the three preselected Train charts from the original
SHA-verified manifest, recompiles *unrounded* Oracle event times, and
instrumentally replays the original width-24 beam and candidate grammar.
It checks its complete chosen chunk/axis path, number of candidate evaluations
and rounded final beam cost against the original saved v0.8 artifact; it also
independently calls the frozen Planner to cross-check the result.

A discrepancy is an **error**, not permission to continue with an approximate
reconstruction or to claim a result.

For each flagged seam it records the chosen final-path predecessor rank,
chosen candidate expanded rank, current beam survival rank, feasible/rejected
candidate counts, full cost breakdown and the cheapest *non-quick* gesture
from that **same predecessor state**. This prevents silently comparing
unrelated histories from the H0.2 local surrogate.

Six cost components: gesture preference; hand-exposure prior; axis-reuse
margin; hand switches inside one chunk; intra-chunk depth motion proxy;
cross-chunk join transition.

## Reproduction on Windows

Use branch develop/dmdogo; preserve all existing untracked files and local
prestige1.toml changes. If the branch has diverged, do not force a reset,
stash, rebase or merge.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git status --short
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h04_beam_decisions.py

uv run python scripts/audit_dmdogo_v09_h04_beam_decisions.py `
  --manifest "reports/dmdogo-v08-train-selection-r2.json" `
  --verified-report "reports/dmdogo-v08-train-report-r4.json" `
  --h0-report "reports/dmdogo-v09-h0-train-style-r1.json" `
  --h01-report "reports/dmdogo-v09-h01-train-conditions-r2.json" `
  --results-root "artifacts/dmdogo-v08/train-generalization-r2" `
  --output "reports/dmdogo-v09-h04-frozen-beam-r1.json"
~~~

The new output is create-once. No original result is overwritten, no physical
Motor is run, and Validation20/FINAL are not touched.

## Cautions

An immediate cheaper alternative may lose during later beam search or be
infeasible under later reuse constraints. Surviving a beam step does not
prove the alternative would produce a valid full-chart plan. The reported
terms are **heuristic costs**, not measured ergonomics or human preference.
There is still no human demonstration or verification of human-like playing.

Even a mathematically exact replay is a description of algorithm behavior,
not proof that one heuristic factor *caused* the original selected pattern.
