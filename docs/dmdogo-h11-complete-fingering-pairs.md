# DMDOGO H1.1 — Reconstruct complete paired fingering assignments

## Research aim

Finish the *offline evidence construction* prerequisite to human
or physical validation. Given three previously verified frozen
Train charts and the H0.4–H0.6 evidence lineage, reconstruct
**all 22 independent H0.6 single-seam counterfactual worlds**.

H1.0 confirmed 7/22 improve the frozen six-term heuristic.
That is NOT 7 human-preferred fingerings or 7 physically
successful alternate Motor plans. H1.1 supplies complete
event-by-event pairings to permit fair future review.

## Scope and invariant

- Branch: `develop/dmdogo`. Do not create additional branches/PRs.
- Never alter v0.8 Planner, Motor, Timing, runner, beam width,
  six heuristic terms, Train/Validation/FINAL source or baseline.
- Use only the previously selected **3 Train charts** (10,231
  unique original events); no chart substitution or new datasets.
- Only abstract, privileged Oracle time reconstruction; **NO**
  physical Motor execution, human label inference or training.
- Maintain one *separate* full-length alternate for **each** seam.
  A combination of the 22 candidates is untested and invalid.
- Exact input report H0.6 source Git blob is pinned:
  `67d8f63086f59c086c3724bc3284fe89178f85f2`.
- Reproduce the saved verified H0.1 report, saved H0.6
  case-by-case full audit and H1.0 quality register before
  accepting any reconstructed path.

## Reconstruction algorithm

1. Verify exact Train manifest SHA and original saved physical
   provenance via the existing verified H0/H0.1 reporter.
2. Independently recompile each exact chart to privileged
   Oracle `GestureEvent` times; match every event identity and
   onset timestamp against saved v0.8 telemetry/replay.
3. Reproduce original frozen v0.8 window segmentation and
   original event-to-axis path exactly. One copy of this
   full original assignment is stored per chart.
4. Run H0.5's `trace_unrestricted` without changing its
   width-24 beam, retaining the original frozen predecessor
   and tagged alternative-seam seed of each of 22 flags.
5. For each tagged seed, run the exact old H0.6
   `_force_suffix_detailed` with width 24. Form a full
   independent path by joining the unchanged frozen prefix
   with its independently replanned alternative suffix.
6. Recompute the six-term heuristic totals, final delta,
   original and forced abstract-axis metrics, 80ms reuse
   constraints, Quick/Rapid flag deltas, changed Chunk
   indices and event assignment counts. Require exact
   equality against the saved H0.6 report.
7. Store every complete alternative event-axis vector
   aligned to the chart's original full event order,
   its full alternative Chunk sequence, and precise
   event-by-event differences (event ID, due_s, before/after).

Any provenance disagreement, unexpected missing target,
incorrect event/chunk coverage, score discrepancy or
false human/physical claim aborts the audit **without**
writing the output.

## Output format

`reports/dmdogo-h11-fingering-pairs-r1.json` contains:

- `charts[]`: 3 independent source charts with exact SHA256,
  full ordered `original_events_by_index` and `original_chunks`.
- `charts[].reconstructions[]`: 10/10/2 alternative worlds,
  each containing all event axes in
  `alternative_axes_by_event_order`, independent full
  `alternative_chunks`, exact `changed_events`,
  `quick_flag_delta`, `rapid_flag_delta`, H0.6 score
  components and provenance.
- `source_sha256`: all preexisting input reports, plus the
  frozen H0.6 source blob identity.
- Evidence flags explicitly indicating that physical
  alternative Motor runs, human judgment and independent
  holdout validation are **not** part of H1.1.

Note: an absent difference at some event does **not** mean
that the candidate rejoined its original internal Beam
state. It only means that the observable axis assignment
at that event is identical.

## Windows PowerShell

Preserve local edits and all previous artifacts. In particular,
do not discard local `configs/training/prestige1.toml`.
Use `git pull --ff-only`; do not stash/reset/rebase/clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h11_fingering_pairs.py

uv run python scripts/reconstruct_dmdogo_h11_fingering_pairs.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h04-report reports/dmdogo-v09-h04-frozen-beam-r1.json `
  --h05-report reports/dmdogo-v09-h05-nonquick-suffix-r1.json `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --h1-report reports/dmdogo-h1-quality-evidence-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-h11-fingering-pairs-r1.json
~~~

This chart-level exact replay is heavier than H1.0's
simple evidence inventory. The output is **create-once**.
For reruns choose an unused revision filename; never
overwrite any previously saved report.

## H1.2 future blinded A/B review

After H1.1 completes and all 22 input identities match,
generate paired visualizations of matching chart spans,
counterbalance A/B ordering and hide cost direction.
Long changed suffixes (thousands of events) require
end-to-end context, not merely the immediate local join.
Human review must be actually observed and must remain
distinct from declared keyboard-finger mappings.

Even a human preference label is not a physical performance
result. Motor comparisons require separate preauthorized
verified alternate physical runs on the same exact charts;
no Validation20/FINAL is authorized by this phase.
