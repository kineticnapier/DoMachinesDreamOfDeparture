# DMDOD Phase 11 — Frozen Unseen Train16 Timing Audit

## Implementation status

**PHASE11 AUDIT_COMPLETED / FULL EVENT-LEVEL DIAGNOSIS COMPLETE**

## Phase 11 measured outcome (user-run, 2026-10-19)

A full three-arm HUD278 audit was run against the frozen, verified 16 Train
charts on the user's Windows environment. The user supplied its standard-output
aggregate reporting `AUDIT_COMPLETED`. The event-level JSON was subsequently uploaded and independently inspected;
console aggregate provenance remains in
`reports/hud-timing-phase11-console-summary-20261019-r1.json`.

| Metric (16 unseen, 1,068 events) | A v1.4 alias OFF | B v1.5 alias OFF | C v1.5 alias ON |
|---|---:|---:|---:|
| Clean lifecycle | 1,039 (97.28%) | 1,052 (98.50%) | 1,052 (98.50%) |
| Category 2: timing estimation failure | 0 | 0 | 0 |
| Category 3: justified abstention | 14 | 14 | 14 |
| Category 4: observability unverified | 2 | 2 | 2 |
| Category 5: lifecycle / ID failure | 13 | 0 | 0 |
| False-positive IDs | 0 | 0 | 0 |
| Conditional Timing MAE | 0.020121 ms | 0.019872 ms | 0.019872 ms |
| Mean first forecast lead | 70.7869 ms | 70.7772 ms | 70.7772 ms |

The known reference five were reported as 199/200 clean (99.5%), one
abstention, zero false-positive IDs. Lifecycle v1.5 recovered 13 events in
previously unseen material; identity ON/OFF had identical aggregate outcomes.
No real Motor Hit/Miss inference follows from these event classification rates.

Remaining 16 events are 14 category-3 abstentions and 2 category-4
observability-unverified cases. Their causes, chart clustering, and safety
interpretation require the **full result JSON**. An absence of category-2
events is a statement about the present evaluator classification only.

The MAE numbers have different clean-event populations for A/B/C and must
not be described as estimator-precision gains. No acceptance threshold was
used to select or filter the 16 Train charts.

## Phase 11 full event-level diagnosis completed

The full 2.2 MB user-run result JSON was uploaded to the analysis chat and
verified (SHA256 `e9e7788797671146571bc734ece294d096478aef86d6e232c1419366bbc79350`).
**All 16 unsuccessful events were classified and checked against compiled
chart-floor structure**. See
[`docs/hud-timing-phase11-event-diagnostics.md`](hud-timing-phase11-event-diagnostics.md)
and `reports/hud-timing-phase11-event-diagnosis-20261019-r1.json`.

Findings: 11 of 16 unseen charts are completely clean. Of the 16 remaining
events, 13 follow Midspin (conservative `special_future_floor` gate),
one is on a SetSpeed+Checkpoint floor (Checkpoint flag is the direct
gate), and two occur 7.353ms after Twirl in a 10ms control loop and remain
observability-unverified. All 13 v1.4 category-5 lifecycle failures recovered
by v1.5 are on Parallel Universe Shifter. No learner, physical Motor, or
estimators were changed.

The raw 2.2MB event JSON was not duplicated into Git history; the exact
source checksum, compact failure cases, and review are committed.

## Earlier preparatory status (superseded by the Windows run)

## Uploaded Train80 verified (2026-10-19)

The user supplied `Train.zip` directly to the assistant. Its 80 ADOFAI charts were
read in the container, and all five previously recorded reference SHA256 hashes
matched. There are 75 unseen candidate filenames; deterministic selection
froze 16 entries and their SHA256 checksums without looking at inference
results. These were committed in
`reports/hud-timing-phase11-selection-20261019-r1.json`.

The uploaded archive checksum is
`99852c736487017796c7ed27cc94f120f30459acdf8189b6573ecd9e5554b071`.
All 16 selected hashes, candidate-pool fingerprint and selection order passed
independent validation. Dataset structural counts across **entire charts**
(not 12-second prefixes) are in
`reports/hud-timing-phase11-uploaded-dataset-preflight-20261019-r1.json`.

The DMDOD implementation is available through the GitHub connector, but the
separate Python execution container cannot access GitHub over the network.
Thus the complete imported HUD278 pipeline cannot currently be executed in
that container. The actual 16-chart Timing/Lifecycle measurements and project
pytest execution remain **NOT RUN**. No accuracy, event coverage, or regression
claim may be inferred from the structural counts.

## Earlier infrastructure-only note

The audit code is committed to the existing branch. The local Train80 data directory
`data/DMDOD-v140-tuf-p11-p20/Train` is ignored by `.gitignore` via `data/*`;
it is not published in this GitHub repository. The assistant's execution
environment does not contain that exact Train80 directory. Therefore no
16-chart manifest, measured coverage, category counts, or PASS verdict has
been fabricated. This is a data-availability boundary, not a Timing failure.

### Frozen baselines

- Starting commit: `818613f7efea17885ddfa9038f6e39deb989b85d`
- 5 known Train reference cases, 200 events: 199 clean / 1 abstention
  (Phase 10.1, prior Windows results; **not re-run here**)
- No Motor Hit-rate claim is made from clean lifecycle coverage.
- Phase 6 stock icon capture stays `UNVERIFIED`.

## Code

`scripts/audit_hud_timing_phase11_generalization_v01.py`

There are two distinct operations. **Prepare always runs before inference**:

1. `prepare`: inspect only top-level Train `.adofai` filenames, excluding the
   known five. Sort the candidates by SHA256 of the fixed salt, a NUL separator,
   and UTF-8 filename. Select the first 16. Freeze names, chart hashes,
   filename-pool fingerprint, and 12-second scope to a new manifest.
2. `audit`: verify the exact same pool, selected names, all 16 SHA256 values
   and all five known-reference hashes. Refuse to continue on missing or
   changed data. Replay identical idle-HUD/countdown input streams into three
   independent Forecast Sources, collect every prediction, *then* read target
   truth for evaluator-only association.

Neither command changes charts or overwrites an output; files use exclusive
creation. There is no selection based on observed accuracy, difficulty or
special-event occurrence.

### Arms

| Arm | Lifecycle | Countdown/v1.4 identity reconciliation |
| --- | --- | --- |
| A | v1.4 | OFF |
| B | v1.5 | OFF |
| C | v1.5 | ON |

All share 80 ms forecast horizon, HUD278, 10 ms control ticks,
same Countdown renderer and Mod-visible BPM observation contract.

No Fingering network, learned policy, physical KeyDown, validation/FINAL,
checkpoint updates or hyperparameter search. The chart environment still
advances idle physics to produce the current visible HUD; no Motor *policy*
is evaluated.

### Measures

For each of the 16 previously unused Train charts, 0–12 seconds:

- five-way posthoc event classification (as in Phase 7/8);
- clean lifecycle coverage and per-chart/event-weighted aggregates;
- false-positive IDs; lifecycle new/reused/canceled/elapsed;
- conditional Timing MAE and first forecast lead, **not** interpreted as
  better timing prediction when the evaluated populations differ;
- A/B forecast interval comparison; B/C source identity differences;
- existing A→C and B→C successful-event regressions;
- failed-event statuses, special-event constructs (SetSpeed, Pause, Twirl,
  Midspin, full turns, MultiPlanet), and a priority list of observed statuses;
- shared-success subset MAE for an apples-to-apples arm comparison.

The original Phase 7 `_evaluate` classification is reused for comparability.
A status classified as a "justified abstention" is justified relative to
the current observation contract; **it is not a proof that all possible
observation sources are insufficient**. When evaluating failures, retain
`observability unverified` separately.

Known reference cases are also rerun with arm C and checked against
199/200 clean, 0 false-positive IDs. The unknown 16 have **no coverage
acceptance threshold**. Bad results must be reported unchanged.

A full output with structural regressions reports
`AUDIT_COMPLETED_WITH_FINDINGS` rather than concealing or silently
dropping charts. Dataset omissions or hash changes abort before evaluation.

## Commands for an environment with the exact local Train80

~~~powershell
.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase11_generalization_v01.py prepare `
  --manifest reports/hud-timing-phase11-selection-YYYYMMDD-r1.json

.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase11_generalization_v01.py audit `
  --manifest reports/hud-timing-phase11-selection-YYYYMMDD-r1.json `
  --output reports/hud-timing-phase11-generalization-YYYYMMDD-r1.json

.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_hud_timing_phase11_generalization.py
~~~

These commands are for reproducibility, not a request that the user consume
Codex's remaining five-hour quota. The published GitHub-only state is the
completed **audit implementation**, not a fabricated completed 16-chart run.

## What remains

The existing HUD-only timing estimator has **not** been modified or revalidated
on new observations. Follow-up is a narrow observability study, not a request
to rerun the 16-chart audit: Midspin-to-next-playable-floor, a short
Twirl-to-next-target gap, and the Checkpoint conservative blocker should be
investigated separately before any policy relaxation.

The full event-level JSON is preserved in the original user upload; its
SHA256 is recorded above, and an event-level subset is in the compact
GitHub report.