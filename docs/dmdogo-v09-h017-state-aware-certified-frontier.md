# DMDOGO v0.9 — H0.17 State-aware lower-bound audit

## Objective

H0.15 established an admissible but late suffix certificate; H0.16
added exact cold-start one/two-Chunk blocks without improving certificate
coverage (0/34 targets certifiable in first five join-inclusive Chunks).
H0.13's release-regret trials **4115** and **4127** remain decisive
counterexamples to an unconditional five-Chunk early-release rule.

H0.17 is the final **state-aware early-certificate feasibility test**
using those same frozen cohorts and scoring system. It does not
change beam selection, implement pruning, train a model or claim
human-like movements.

## Proof-limited construction

At each of the first five post-admission windows in an unchanged
width24/quota8 trace, collect **all** retained protected Branch
states `T`: each includes the full cumulative score, hand exposures,
last-axis identifier and last-used times for all axes.

For every `s in T`, enumerate every reuse-feasible continuation
for `k=1` and `k=2` next Chunks **without beam pruning**. Use
the exact frozen H0.4/v0.8 six-term scoring and original times.
Let `score(s → p)` be the total cumulative score after extension
`p` of `k` Chunks, and let `L16[i+k]` be H0.16's previously
verified state-independent admissible future suffix. Define:

    A_k(T, i) = min_{s∈T, feasible p of length k}
                (score(s → p) + L16[i+k])

where `i` is the first unprocessed Chunk. Because **every**
feasible descendant of every `s` is represented by the exact
enumeration before `L16` is appended, each `A_k` is an
admissible lower bound on the best possible tagged completion.

Also keep H0.16's existing retained-frontier bound:

    B16(T,i) = min_{s∈T} score(s) + L16[i].

Use `max(B16, A1, A2)` when available, since the maximum of
admissible lower bounds is itself admissible. If no tagged
states are retained, mark non-applicable; do not turn it into
an unsupported numerical certificate. If no feasible child
remains in a lookahead, explicitly record that result without
silently treating it as a finite bound.

Against the **offline completed, feasible** frozen v0.8
incumbent of total cost `U`, certify only when:

    max(B16, A1, A2) > U + 1e-6 + 1e-12*abs(U).

This statement means **none of the descendants of the CURRENT
retained tagged frontier can strictly improve on U**. It does
NOT establish global optimality, preserve exact final results
after modifying the width24 beam, or prove real Motor safety.

## Reproduction and audit controls

- Use the unchanged original H0.12 artificial generator.
- Reproduce the same 2 regret targets 4115/4127 and H0.16's
  prior 8 Discovery + 8 Evaluation trials (34 targeted seams).
- Verify H0.12/H0.13/H0.14/H0.15/H0.16 report identities,
  SHA256 chain and pinned frozen v0.8/H0.4 scorer Git blobs.
- Independently replay the first five join-inclusive quota8
  windows with **full state objects**; verify exact retained
  candidate-index paths, counts, and best tagged cumulative
  scores against H0.14 full-trajectory snapshots.
- Recompute H0.16's cold suffix, and require exact old
  H0.16 first-five certificate parity. New proofs may only
  supplement, not revoke, earlier proofs.
- If a tagged terminal path beats the incumbent, any positive
  early certificate is a hard failure.
- Count locally evaluated lookahead candidates so that extra
  computational expense is visible rather than claimed as
  a performance optimization.

No generation of fresh holdout data or retuning of 5-Chunk
thresholds is permitted. The experiment measures bound strength
on already-observed synthetic source cohorts.

## Windows PowerShell

Continue on `develop/dmdogo`, preserving the dirty
`configs/training/prestige1.toml`, existing local reports and
any untracked artifacts. Use ff-only pull, no resets or cleanup.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only

uv run pytest -q tests/test_dmdogo_v09_h017_state_aware_lower_bound.py

uv run python scripts/audit_dmdogo_v09_h017_state_aware_lower_bound.py `
  --h012-report reports/dmdogo-v09-h012-synthetic-r1.json `
  --h013-report reports/dmdogo-v09-h013-independent-seeds-r1.json `
  --h014-report reports/dmdogo-v09-h014-regret-autopsy-r1.json `
  --h015-report reports/dmdogo-v09-h015-admissible-suffix-r1.json `
  --h016-report reports/dmdogo-v09-h016-cold-block-r1.json `
  --output reports/dmdogo-v09-h017-state-aware-r1.json
~~~

Output uses exclusive file creation. A rerun requires a fresh
output revision name, never overwriting previously saved evidence.

## Interpretation / planned stopping decision

If there are new early certificates, inspect their exact
frontier state and independent parity; a follow-on phase may
study whether this narrow proof is worth the computational
expense. **Do not** immediately enable a pruning rule.

If there are still zero useful early certificates, H0.17
provides evidence to stop locally refining lower bounds
on this frozen cohort. The project can then document a
cost-versus-diversity trade-off and move toward independent
offline Motor/human-likeness validation, **without** claiming
an error-free safe release policy.

A failure of H0.17 to certify is not a proof that no possible
state-aware/DP bound can work. It is a scoped negative result
for exact one-/two-Chunk lookahead plus the known H0.16 suffix.
