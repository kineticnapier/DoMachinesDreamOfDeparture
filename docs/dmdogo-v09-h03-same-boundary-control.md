# DMDOGO v0.9 — Phase H0.3: Same-boundary control

## Why H0.2 is not yet a chunk-boundary causal result

H0.2 reported 22/22 original quick depth-jump flags at seams. It
produced 106 abstract-reuse-legal alternative repartitions, all 106 of
which removed the original flag. It **did not** report whether replanning
with the **unchanged original chunk boundary**, using the same local
surrogate, would already remove the flag.

Additionally, the v0.8 gesture candidate grammar itself strongly
constrains consecutive same-hand movements *within one chunk*. Once a
repartition removes the old seam, the originally flagged pair belongs
to one local gesture. A flag-free candidate there may be guaranteed by
the grammar rather than reflecting any empirically learned preference
or ergonomic improvement. An alternative 3-chunk partition also changes
chunk count, not solely the position of the original seam.

## Design

H0.3 analyzes the 22 same original flagged seam pairs, in each of the
three provenance-verified Train charts. It **does not modify** the
frozen v0.8 run or simulate physical KeyDowns.

Three settings are compared:

- **Frozen axes, frozen boundary**: observed v0.8 result; flag present
  by study selection.
- **Replanned axes, same original boundary**: H0.2's
  `same_boundary_surrogate`, recomputed in H0.3 with the **same
  independent local optimizer**, pinned outside-chart assignments and
  80ms abstract same-axis reuse.
- **Replanned axes, changed boundary**: all H0.2 alternatives, verified
  against the original chart assignments, abstract reuse constraint and
  candidate membership in the frozen **read-only** grammar.

H0.3 reports, per flag, whether the same-boundary replan already removes
it, how many alternative partitions work, and whether a repartition
also changes chunk count (2 vs 3 chunks). It enumerates the candidate
grammar at the original pair *inside* each new chunk, independently
counting possible quick jumps. This directly tests whether the H0.2
106/106 flag-free result is a **structural artifact** of the
within-chunk grammar.

H0.3 deliberately makes **no claim** that one run over only 22
conditioned-on-flag examples identifies a causal effect. Its labels
`replanning_alone_suffices_for_this_flag` and
`partition_associated_under_same_surrogate` are descriptive and
do not indicate human-likeness.

## Run on Windows

Read-only inputs and no new physical simulation, model learning,
Validation20 or FINAL:

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture

uv run pytest -q tests/test_dmdogo_v09_h03_boundary_controls.py

uv run python scripts/analyze_dmdogo_v09_h03_boundary_controls.py `
  --manifest "reports/dmdogo-v08-train-selection-r2.json" `
  --verified-report "reports/dmdogo-v08-train-report-r4.json" `
  --h0-report "reports/dmdogo-v09-h0-train-style-r1.json" `
  --h01-report "reports/dmdogo-v09-h01-train-conditions-r2.json" `
  --h02-report "reports/dmdogo-v09-h02-boundary-cf-r1.json" `
  --results-root "artifacts/dmdogo-v08/train-generalization-r2" `
  --output "reports/dmdogo-v09-h03-boundary-controls-r1.json"
~~~

The script revalidates H0.1 against frozen v0.8 provenance and checks
cohort identity, time/event coverage, original flags, same-boundary
baseline determinism, candidate grammar membership, and reuse
constraints for each feasible moved-seam variant.

The output is **create-once**. Existing JSON files are never
overwritten. If a test or source assertion fails, preserve the
original files and report the failure.

## Interpretation protocol

### If the same-boundary control removes most of the 22 flags

It shows that *local replanning alone* can change the axes enough to
remove a selected flag. The 106/106 alternative result then provides
**no independent support** that moving seams was necessary.

### If the same-boundary control retains most flags

Alternatives are associated with their removal under the **same**
surrogate optimizer, but candidate grammar and potentially the number
of chunks changed. This does not isolate a pure seam-placement effect.

### If the grammar never permits a quick jump within the newly combined chunk

The 106/106 result is structurally expected from the grammar. It is
not a learned human behavior, a performance gain, or a verified
physical intervention.

### If the local surrogate fails to reproduce v0.8 assignments

Expected in principle: the surrogate starts with truncated exposure
history, and the frozen v0.8 planner uses a chart-wide beam. Only
surrogate-vs-surrogate costs can be compared.

## Non-claims

- No human demonstration or independent anatomical observation.
- No preferred-fingering score, reward model or NN training.
- No claims of causal identification, Motor feasibility, or real-time performance.
- No changes to the Planner, Motor, Timing, runner, existing outputs or
  the locally modified `prestige1.toml`.
