# DMDOGO v0.8 — Codex review remediation

## Frozen baseline remains unchanged

Physical bool: 1,122/1,122 Hit, X100%, MAE 4.500752ms,
0 local replans and 0 deferred events.

The frozen source commit is 45f30bcaf968281400e2c2913a96dd95515222d2.
Planner, Runner, Motor, Timing, checkpoints and chart were not edited.
Validation20 and FINAL remain prohibited without user permission.

## High #1: Run provenance

New scripts/dmdogo_v08_provenance.py compares all tracked frozen
Python/TOML files under src/dmdod and scripts plus pyproject.toml/uv.lock
to frozen Git object hashes. It records actual HEAD, code tree digest,
dependency versions, Python/platform, selected environment variables,
Motor control/physics dt, lead defaults, physical mapping, planning mode
and minimum reuse limit.

New scripts/run_dmdogo_v08_verified_train_case.py accepts ONLY cases
preselected from Train, checks chart SHA and the complete source tree,
forces gesture_v08 / 80ms reuse / no diagnostic prefix, and launches
the unchanged physical runner in an externally timed subprocess.

Per-case sidecars are saved under the results root in _provenance,
with exit code, completion state, source/environment/settings digests,
selection manifest SHA and completed summary SHA.

The outer timeout covers compilation, Planner AND Motor. The historical
bool run's operating environment was not comprehensively recorded,
so bit-for-bit equivalence to that historic run cannot be claimed.

## High #2: Strict reporter

scripts/report_dmdogo_v08_train_generalization.py now rejects:
- missing provenance sidecar or failed/incomplete sidecar;
- mismatched source digest, chart SHA, chart name, policy or manifest SHA;
- non-80ms reuse, diagnostic prefixes or changed Motor settings;
- changed Python package/environment fingerprint across Train cases;
- non-finite/negative/implausible Hit, Miss, MAE, X or Motor values;
- Hits + Misses != Targets, invalid total event count, or edited summary bytes.

It reports pending / invalid_identity / invalid_metrics /
unverified_provenance / different_runtime_environment /
did_not_complete / chart_complete separately. Chart completion can include
misses, so completion is not all-Hit success.

Old Runner summaries without proof cannot count as verified generalization
results. Do not manufacture retroactive sidecars.

## Medium #3/#4: Actual gesture adoption and unique KeyDown pairing

scripts/audit_dmdogo_v08_execution.py reads existing complete summary,
telemetry and replay files. It measures:
- exact planned vs chosen physical axes;
- preserved entire gesture chunks;
- planned due, chosen due and actual physical KeyDown axis reuse;
- one-to-one diagnostic matching with ±35ms and ambiguity counters.

One KeyDown is never reused for multiple events in this diagnostic.
This is still heuristic; authoritative Scorer Hit/X/MAE is unchanged.

Independently checked on the saved complete bool files:
- chosen axes exactly match planned axes: 1,122/1,122;
- complete gesture chunks preserved: 392/392;
- minimum chosen due-time reuse: ~156.224ms;
- minimum actual same-axis KeyDown interval: 156ms;
- violations against 80ms: 0.

## Medium #5: Timeout

Verified launcher applies a subprocess timeout across compilation,
Planner and physical Motor, without editing the frozen Runner timer.

## Low: Chord coverage

Codex found no chords (chord_size > 1) in 80 Train charts. This
Train-only study cannot validate chord handling, regardless of Hit/X.

## Selection cases from Codex — NOT physical results

ALL MY FELLAS: e142af91b178, 1,540 events, 138.19s.
ADOFAI Regular Dan - 2nd Dan: 2ada1d0a06b5, 3,637 events, 495.25s.
Megamix2.7: e6997b348c82, 5,054 events, 606.91s.

All three had feasible abstract plans. These are NOT physical outcomes.

## Windows commands

Remain on develop/dmdogo. Do not reset/rebase/stash/clean, replace files,
change branches or discard modified prestige1.toml. Inspect git status
before any fast-forward pull.

Run tests:

    uv run pytest -q tests/test_dmdogo_v08_train_generalization.py tests/test_dmdogo_v08_train_generalization_report.py tests/test_dmdogo_v08_execution_audit.py

Create a NEW r2 selection, because older r1 manifests lack the full
code digest (do not overwrite r1):

    uv run python scripts/select_dmdogo_v08_train_generalization.py --train-root "data/DMDOD-v140-tuf-p11-p20/Train" --output "reports/dmdogo-v08-train-selection-r2.json"

Run ONLY the first case at first (PowerShell):

    $m = Get-Content "reports/dmdogo-v08-train-selection-r2.json" -Raw | ConvertFrom-Json
    $first = $m.cases[0]
    uv run python scripts/run_dmdogo_v08_verified_train_case.py --manifest "reports/dmdogo-v08-train-selection-r2.json" --case-sha $first.sha256 --results-root "artifacts/dmdogo-v08/train-generalization-r2" --status-root "logs/dmdogo-v08-train-generalization-r2" --wall-timeout-s 1200

Diagnose saved physical execution without rerunning:

    $id = $first.sha256.Substring(0,12)
    $root = "artifacts/dmdogo-v08/train-generalization-r2/$id"
    uv run python scripts/audit_dmdogo_v08_execution.py --summary "$root/summary.json" --telemetry "$root/telemetry-v08.json" --replay "$root/replay-v08.json" --output "$root/gesture-audit-r1.json"

Collate the chosen Train cases, including pending ones:

    uv run python scripts/report_dmdogo_v08_train_generalization.py --manifest "reports/dmdogo-v08-train-selection-r2.json" --results-root "artifacts/dmdogo-v08/train-generalization-r2" --output "reports/dmdogo-v08-train-report-r2.json"

Never overwrite existing result folders or existing reports. Do not
tune v0.8 on these cases, and do not run Validation20/FINAL.

## Verification status

The previous Codex focused 27 PASS applied to the original 7097ecd
review snapshot, not these newer evaluation-only changes. New focused
tests and the verified launch workflow are added but still require
execution on the user's Windows environment.
