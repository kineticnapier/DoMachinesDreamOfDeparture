# DMDOD Phase 11 — Frozen Unseen Train16 Timing Audit

## Implementation status

**AUDIT_INFRASTRUCTURE_COMMITTED; TRAIN80_EVALUATION_NOT_RUN**

The audit code is committed to the existing branch. The local Train80 data directory
\`data/DMDOD-v140-tuf-p11-p20/Train\` is ignored by \`.gitignore\` via \`data/*\`;
it is not published in this GitHub repository. The assistant's execution
environment does not contain that exact Train80 directory. Therefore no
16-chart manifest, measured coverage, category counts, or PASS verdict has
been fabricated. This is a data-availability boundary, not a Timing failure.

### Frozen baselines

- Starting commit: \`818613f7efea17885ddfa9038f6e39deb989b85d\`
- 5 known Train reference cases, 200 events: 199 clean / 1 abstention
  (Phase 10.1, prior Windows results; **not re-run here**)
- No Motor Hit-rate claim is made from clean lifecycle coverage.
- Phase 6 stock icon capture stays \`UNVERIFIED\`.

## Code

\`scripts/audit_hud_timing_phase11_generalization_v01.py\`

There are two distinct operations. **Prepare always runs before inference**:

1. \`prepare\`: inspect only top-level Train \`.adofai\` filenames, excluding the
   known five. Sort the candidates by SHA256 of the fixed salt, a NUL separator,
   and UTF-8 filename. Select the first 16. Freeze names, chart hashes,
   filename-pool fingerprint, and 12-second scope to a new manifest.
2. \`audit\`: verify the exact same pool, selected names, all 16 SHA256 values
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

The original Phase 7 \`_evaluate\` classification is reused for comparability.
A status classified as a "justified abstention" is justified relative to
the current observation contract; **it is not a proof that all possible
observation sources are insufficient**. When evaluating failures, retain
\`observability unverified\` separately.

Known reference cases are also rerun with arm C and checked against
199/200 clean, 0 false-positive IDs. The unknown 16 have **no coverage
acceptance threshold**. Bad results must be reported unchanged.

A full output with structural regressions reports
\`AUDIT_COMPLETED_WITH_FINDINGS\` rather than concealing or silently
dropping charts. Dataset omissions or hash changes abort before evaluation.

## Commands for an environment with the exact local Train80

~~~powershell
.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase11_generalization_v01.py prepare \`
  --manifest reports/hud-timing-phase11-selection-YYYYMMDD-r1.json

.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase11_generalization_v01.py audit \`
  --manifest reports/hud-timing-phase11-selection-YYYYMMDD-r1.json \`
  --output reports/hud-timing-phase11-generalization-YYYYMMDD-r1.json

.\.venv\Scripts\python.exe -m pytest -q \`
  tests/test_hud_timing_phase11_generalization.py
~~~

These commands are for reproducibility, not a request that the user consume
Codex's remaining five-hour quota. The published GitHub-only state is the
completed **audit implementation**, not a fabricated completed 16-chart run.

## What remains

Execution requires the exact ignored Train80 directory (including the five
historically hashed reference charts), which is not present in the assistant
runtime or versioned repository. Until that data is available to an execution
environment, the Phase 11 generalization result remains **NOT RUN**.
