# DMDOGO H1.6 — Read-only Motor versus abstract-fingering evidence

## Question and scope

Does the six-term frozen v0.8 heuristic cost (7 lower-cost alternatives, 15 non-lower-cost alternatives) align with any **measurable** difference in simulated Motor execution or abstract axis/hand assignment indicators?

This is a descriptive, paired, complete 22-case analysis. It is NOT a new Motor run, statistical significance test, human comfort test, human-preference experiment, independent validation, or HUD-only Student result.

## Data sources / guards

- H1.1 22 alternative full-chart fingering vectors; H1.3 fully recomputed abstract/pressure metrics and six-term objective delta; baseline v0.8 full-chart Train report; H1.5 complete 22-case Motor census.
- Independently re-run H1.5's read-only provenance scan for all 22 output/status pairs; refuse missing/modified bytes, changed historical pinned H1.4 files, discrepancies with H1.1 desired axis choices, altered score or missing physical telemetry.
- Saved H1.5 census JSON must match a newly recomputed census *exactly*, including output hashes, input SHA256s, all 22 cases, full planned-axis adoption, and original runtime fingerprint matches.
- Require 3 verified original Train chart baselines, all 22 independent alternative episodes, zero new Motor runs, and no access to Validation20/FINAL.

## Paired candidate comparison

Each record is joined by candidate_id and original chart SHA256, not by row number across unrelated charts.

Physical Motor observations compare the alternative episode against the **existing** original v0.8 episode of that same chart: Timing MAE (signed delta in milliseconds); deferred_event count (signed delta); Hit/Miss, actual planned axis adoption and local replan. No re-execution of original or alternative is performed.

Original MAE/deferral values repeat for multiple candidates on the same chart. Runtime fingerprints match, but wall timing can differ. These differences are descriptive, not causal fingering effects. A conservative 0.1 ms MAE comparison band distinguishes larger signed differences from micro-variations; 0.1 ms is **not** a physiological or experimentally calibrated threshold.

Abstract assignment indicators reuse H1.3's full-chart original and alternative metrics: counts per physical axis and hand, left/right hand-contact redistribution, 120 ms quick same-hand depth jumps, short same-hand runs, same-axis reuse (<100 and <150 ms), peak half-second/one-second contacts and hand imbalance. These are deterministic characteristics of planned axis sequences. H1.5 shows all 22 plans were adopted; that does **not** turn these abstract proxies into directly measured physical strain.

The report separates (1) objective improvement, (2) selected abstract metrics, and (3) simulated Motor outcomes. It shows 7-versus-15 descriptive groups and **per-chart** counts; neither 22 counterfactuals nor repeated baseline rows are independent chart samples. No correlation coefficient, p-value or claim of human quality is generated.

## Windows PowerShell reproduction

On develop/dmdogo, preserve all working files; no reset/clean/stash/rebase/force-push. First:

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h16_motor_proxy_relationships.py
~~~

Then execute the **read-only** physical evidence scan and create-once H1.6 analysis:

~~~powershell
uv run python scripts/audit_dmdogo_h16_motor_proxy_relationships.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h11-report reports/dmdogo-h11-fingering-pairs-r1.json `
  --h13-report reports/dmdogo-h13-automated-quality-r1.json `
  --h15-report reports/dmdogo-h15-motor-22-census-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-h16-motor-proxy-relationships-r1.json
~~~

Do NOT supply --execute: H1.6 has no execution mode. Input path flags may be overridden only when referring to the same source-verified cohort; the defaults for H1.5 existing episode files are artifacts/dmdogo-h15/motor-r1 and logs/dmdogo-h15/motor-r1. The original two successful H1.4 pilots remain at their recorded locations.

Failure to verify any existing record, any changed input SHA256, or an existing H1.6 output file causes an error. No existing data is overwritten.

## Interpretation limits

- 22/22 success means current Oracle-timed physical simulation is saturated for these Train alternatives, **not** that fingering quality is known or identical.
- The frozen heuristic and H1.3 probes are both computed from the same axis schedule; they are not independent labels.
- MAE differences may reflect execution variance; deferrals mean control state conflicts rather than physical fatigue.
- No independent physical human finger data, measured force, ergonomics, labels or HUD-only Student evidence.
- Before asserting correlation or human superiority, additional independent observable evidence and independent unseen chart designs would be necessary. This stage does not access them.

## Status

The implementation is committed to develop/dmdogo as a new research-only script, tests and this document. It does not edit frozen Planner/Motor/Timing, checkpoints, historic chart selection, original saved reports or previously collected H1.4/H1.5 evidence. Local Windows test and H1.6 analysis results are **not yet known**.
