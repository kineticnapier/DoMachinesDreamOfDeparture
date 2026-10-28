# DMDOGO v0.9 — H0.15: admissible suffix-bound certificate

## Status and claim

H0.14 isolated two synthetic counterexamples to H0.11's fixed
five-Chunk release: **4115** and **4127**. Releasing quota protection
pruned still-feasible tagged descendants; continued protection found
later lower-cost final plans. Therefore fixed-horizon release is
**not generally safe**, even for the frozen abstract heuristic.

H0.15 investigates a deliberately narrower mathematically
justifiable criterion. It DOES NOT implement a new pruning policy
or modify v0.8, H0.11, Motor or the frozen score.

At a chosen window `i` in a full quota-8 execution, let:

- `S_i` be the **minimum cost** of a *currently retained* tagged
  hypothesis after window i.
- `L_{i+1}` be a proven optimistic lower bound on all future
  per-window costs from windows i+1 to terminal.
- `U` be the final cost of an **already known complete, feasible**
  path, here the exact original frozen v0.8 plan.

Every feasible descendant of the currently retained tagged
frontier must cost at least `S_i + L_{i+1}`. Therefore, when

    S_i + L_{i+1} > U + 0.000001 + 1e-12 * abs(U)

it is certified that **none of these tagged descendants can
strictly beat that particular incumbent**. The safety margin is
conservative with respect to floating point effects.

Failure to certify means **unknown**, not that the tagged path
will improve. If tagged descendants are already extinct, the
certificate is marked non-applicable rather than `certified`.

### Frozen nonnegative objective proof assumptions

The original unmodified heuristic has six additive terms:

1. `gesture_preference`: generated values are nonnegative.
2. `hand_exposure`: `.45 * nonnegative exponentially decayed load`.
3. `axis_reuse_margin`: `2 * max(0, .150-duration)`.
4. `intra_gesture_hand_changes`: `.35 * number of hand transitions`.
5. `intra_gesture_depth_distance`: `.10 * depth jump excess`.
6. `chunk_join_transition`: `.12` or `.05 * excess distance`,
   or zero.

The suffix lower bound sums the smallest context-independent
gesture preference + intra-gesture hand/depth penalties over
**all generated candidates in each remaining Chunk**. Other
terms are relaxed to zero. Abstract reuse constraints can only
exclude candidates and thus cannot make this bound too high.

Source Git blob checksums for the exact v0.8 planner and H0.4
replay scorer are pinned and checked before producing results.
If either frozen source changes, H0.15 fails closed.

## What this does **not** prove

This certificate only compares tagged continuations **reachable
from the currently retained tagged frontier** against an already
known complete feasible incumbent. It does **not** guarantee that:

- A changed **width-24 beam** preserves the same winner as the
  full quota-8 search; removing one lineage changes which other
  hypotheses obtain slots, and later beam pruning is nonmonotonic.
- An individual protected candidate can safely be removed while
  preserving Quick/Rapid flags or the original assignment.
- A global optimum has been discovered (v0.8 itself uses a bounded beam).
- An offline incumbent is available to a live HUD-only controller.
- Physical Motor, Timing or human-fingering safety follows.

Consequently, there is **no new automatic early-release rule**
introduced in this phase.

## Experimental protocol

1. Verify H0.12, H0.13 and H0.14 source report provenance
   (including the two known counterexamples).
2. Reproduce full quota-8 source behavior on 4115 and 4127;
   independently verify exact terminal winner, tagged survivor,
   rescue windows and displaced slots.
3. Calculate a suffix lower bound for every retained tagged
   population on those known-regret trajectories. Any attempted
   positive certificate while an improved tagged terminal plan
   is known makes the audit fail.
4. Measure how often certificates are possible on a **fixed**,
   previously observed diagnostic survey of Discovery IDs
   `4096..4103` and Evaluation IDs `8192..8199`. Both selected
   before analysis, independent of whether individual targets
   are favorable. Recompute saved H0.13 trials exactly.
5. Count certifiable windows and targets, including any early
   certificate available during the first five join-inclusive
   chunks. Do **not** retune a coefficient or release threshold.

These prior artificial cohorts are **not a fresh holdout**;
they merely measure whether the safe certificate has enough
coverage to be worth further research.

## Windows PowerShell

Use `develop/dmdogo`; preserve locally modified
`configs/training/prestige1.toml`, reports, and all artifacts.
Never reset/stash/rebase/clean or overwrite saved results.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h015_admissible_suffix_bound.py

uv run python scripts/audit_dmdogo_v09_h015_admissible_suffix_bound.py `
  --h012-report reports/dmdogo-v09-h012-synthetic-r1.json `
  --h013-report reports/dmdogo-v09-h013-independent-seeds-r1.json `
  --h014-report reports/dmdogo-v09-h014-regret-autopsy-r1.json `
  --output reports/dmdogo-v09-h015-admissible-suffix-r1.json
~~~

Output is created exclusively; repeated runs need new revision
paths. This audit reruns full-chart abstract searches, possibly
several times per target. No optimizer or physical Motor runs.

## Reading results

- `known_regret_cases`: whether the provable test correctly
  abstains while the protected lineage is known to win.
- `fixed_survey`: numbers of certifiable windows and cases among
  the prespecified 16 synthetic source trials.
- `certifiable_within_first_five_count`: potential early pruning
  coverage **relative to the completed v0.8 incumbent only**.
  It does NOT represent a guaranteed improvement in runtime
  or final search performance under altered beam pruning.

If few frontiers can be certified, this is a useful negative
finding: one should not weaken the condition and call it safe.
