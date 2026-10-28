# DMDOGO v0.9 — H0.16 Cold-block admissible suffix bounds

## Motivation

H0.15 found **0 early certificates** among the known regret targets
and 32 previously observed fixed-survey targets. It nevertheless
certified 12/32 fixed-survey targets at later windows. The original
lower bound retained only preference, within-gesture hand switches
and within-gesture depth distance, relaxing everything else to zero.

H0.16 strengthens the lower bound **without weakening the proof
criterion, adjusting the five-Chunk horizon or editing beam pruning**.

## Mathematical construction

Let `C(P, w)` denote the exact frozen v0.8 incremental cost of
choosing candidate sequence `P` over window(s) `w`, starting
from the actual reachable incoming state. All six frozen
objective terms are nonnegative. Exposure load stays nonnegative,
decays multiplicatively by exp(-dt/0.75), and adds positive costs.
Pre-block axis-use history and the previous last axis can only
add reuse-margin or join penalties, or make candidates infeasible.

Define `cold_cost(i, i+k)`, for `k=1 or 2`, to be the *exact*
minimum v0.8 incremental cost over **every feasible combination**
of candidates in the indicated 1–2 chunks, starting with:

- Zero previous exposure in both hands;
- Empty axis-use history and no previous last axis;
- Previous-due unset for the first chunk of each block;
- Original within-block times, candidate grammar, reuse feasibility,
  hand-exposure decay and all six scoring terms.

There is **no beam truncation** while minimizing these local blocks.
Any feasible true continuation over that same block has a
corresponding cold-start sequence of no greater cost: resetting
history removes only nonnegative terms and constraints. Therefore
the cold-block minimum is an admissible lower bound.

The suffix lower bounds, indexed by first remaining chunk `i`, are:

    L1[i] = cold_cost(i, i+1) + L1[i+1]
    L2[i] = cold_cost(i, i+2) + L2[i+2]

with the last singleton handled by `cold_cost(i, i+1)` and
both suffix arrays zero at terminal. **Resetting cold state between
disjoint blocks** makes these sums optimistic; they cannot
overestimate a feasible actual suffix cost. Each block can
be minimized independently, without assuming the winning
candidate choices can be connected across blocks.

For each suffix i choose the maximum of three individually
admissible lower bounds:

    Lmax[i] = max(H0.15_original[i], L1[i], L2[i])

Because each is a lower bound, so is their maximum. H0.16
checks at every i that `L2 >= L1 >= H0.15` within 1e-7
numeric tolerance, and refuses a source or parity mismatch.

Certificate rule is **unchanged**:

    tagged_frontier_min_score + Lmax[next_chunk]
        > original_feasible_v08_complete_cost
          + (1e-6 + 1e-12 * abs(incumbent))

This can establish only that no extension of the **currently
retained tagged frontier** can strictly improve upon the known
incumbent. It **cannot** guarantee that removing protected slots
from a width-24 beam keeps the same final winning path. Changing
pruning changes competing lineages and the state population.

## Fixed experimental protocol

Reuse EXACTLY H0.15's source cohort:

- Historical abstract-search regret trials: 4115 and 4127.
- Discovery source trial IDs 4096..4103.
- Evaluation source trial IDs 8192..8199.

All prior H0.12/H0.13/H0.14/H0.15 report inputs and SHA256s
are checked. The frozen scorer v0.8 and H0.4 code Git blob hashes
are pinned and fail-closed on change.

For every H0.15 target, independently reconstruct the original
full quota8 search and compare its terminal winner, best tagged
survivor, source candidate, rescued windows, displaced slots and
all H0.15 original per-window certificates. Report each of:

- `intrinsic_h015` — original lower bound unchanged.
- `cold_one` — exact independently reset single-Chunk lower bound.
- `cold_two` — exact independently reset two-Chunk lower bound.
- `maximum` — pointwise best of all the admissible lower bounds.

Measure certificate windows, first certificate position relative
to the seam, and certificates during the first five Chunks.
Any certificate in a known-regret case that has a better tagged
terminal plan than the incumbent is an audit failure, not a
positive result.

Do not use these fixed, previously studied synthetic IDs as a
fresh holdout. This is an explanatory assessment of bound strength.

## Windows PowerShell

Use branch `develop/dmdogo`, preserve local edits and all prior
artifact JSONs. Pull ff-only; no reset/rebase/stash/clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h016_cold_block_lower_bounds.py

uv run python scripts/audit_dmdogo_v09_h016_cold_block_lower_bounds.py `
  --h012-report reports/dmdogo-v09-h012-synthetic-r1.json `
  --h013-report reports/dmdogo-v09-h013-independent-seeds-r1.json `
  --h014-report reports/dmdogo-v09-h014-regret-autopsy-r1.json `
  --h015-report reports/dmdogo-v09-h015-admissible-suffix-r1.json `
  --output reports/dmdogo-v09-h016-cold-block-r1.json
~~~

Output is exclusive-create. If you repeat, change the revision
filename instead of overwriting a source. Cold two-chunk
enumeration increases CPU work; it is an **audit**, not a
demonstrated production speedup.

## Interpreting results

Finding earlier certificates would establish that the original
nonnegative frozen objective provides more optimistic bound
information than H0.15 used. It would **not** certify that a
new width-24 beam quota-release policy is lossless. No real
charts, Validation20, FINAL, physical Motor, learned Timing,
or HUD-only policy is used or modified.

If still no useful early certificates occur, do not weaken
admissibility or relabel an unproved stopping policy as safe.
