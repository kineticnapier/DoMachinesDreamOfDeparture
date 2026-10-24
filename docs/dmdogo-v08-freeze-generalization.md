# DMDOGO v0.8 — Frozen baseline and Train generalization

**2026-10-24 REVIEW UPDATE:** The direct physical-run and collate
commands later in this document are now **historical, not verified
generalization procedures**. They omit mandatory execution-provenance
sidecars and the old reporter could accept corrupt metrics. Use
[Codex review remediation](dmdogo-v08-codex-review-remediation.md) for
the revised create-once r2 manifest, verified outer runner,
one-to-one gesture audit and strict reporter. The frozen v0.8 Planner
and Motor remain unmodified.


## Frozen reference

Full physical Train bool, supplied by the user:
- Source chart: 04042 - bool - vo[ICE].adofai
- Chart SHA256: 04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1
- v0.8 source commit: 45f30bcaf968281400e2c2913a96dd95515222d2
- Hit 1122/1122, Miss 0, TooEarly 0, Overload false
- X Accuracy 100%; Timing MAE 4.500751665ms; signed error +4.436558357ms
- Physical replan 0, deferred events 0, unassigned events 0
- Gesture chunks 392; abstract same-axis reuse minimum 156.22396ms
- Maximum observed Motor controller call 11.1626ms

The authoritative numeric snapshot is docs/dmdogo-v08-frozen-baseline.json.

Frozen code Git blobs, checked before chart selection:
- Planner: 4ce02c2e4450dc7a5181ab74cbaa0c1accea94f5
- Physical runner: 5fddfc84419722de7bbd0556419511069fd3200d

No Timing or Motor changes; no per-chart training, weight changes or
hyperparameter experiments. Never run Validation20 or FINAL without
explicit permission. Remain on develop/dmdogo and preserve all dirty or
untracked local files; never reset, rebase, clean or stash to force a pull.

## 1. Select a few different Train charts — no physical execution

Run focused tests:

~~~powershell
uv run pytest -q tests/test_dmdogo_v08_train_generalization.py tests/test_dmdogo_v08_train_generalization_report.py
~~~

Select at most three different Train charts, excluding the exact bool
baseline by SHA256:

~~~powershell
uv run python scripts/select_dmdogo_v08_train_generalization.py --train-root "data/DMDOD-v140-tuf-p11-p20/Train" --output "reports/dmdogo-v08-train-selection-r1.json"
~~~

The script must be pointed to a directory named Train. It explicitly
excludes Validation/FINAL/holdout subtrees and records parse failures.
Selection dimensions are chord prevalence, small inter-onset intervals
and large gap variability. It falls back to highest event count when a
category has no eligible candidate. This is purposive exploratory
screening, not a statistically representative sample. Nothing is
trained or physically simulated at this stage.

The selected chart paths and SHA256 hashes are saved create-once in
reports/dmdogo-v08-train-selection-r1.json. The selector refuses to
claim frozen identity if either checked code blob differs.

## 2. Run separate physical Train episodes

Inspect the manifest first. Then in PowerShell:

~~~powershell
$m = Get-Content "reports/dmdogo-v08-train-selection-r1.json" -Raw | ConvertFrom-Json
foreach ($case in $m.cases) {
    $id = $case.sha256.Substring(0, 12)
    $out = "artifacts/dmdogo-v08/train-generalization-r1/$id"
    $status = "logs/dmdogo-v08-train-$id.status.json"
    if ((Test-Path $out) -or (Test-Path $status)) {
        Write-Host "Skipping existing output: $id"
        continue
    }
    uv run python scripts/run_dmdogo_v06_pattern_first.py --planning-mode gesture_v08 --chart $case.path --output-dir $out --status-file $status --wall-timeout-s 1200
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed: $($case.chart) exit=$LASTEXITCODE"
    }
}
~~~

Each result is independent. No baseline outputs are overwritten.
A failed chart is evidence; never hide it with fallback to a different
chart, modified Motor settings or a tuned policy.

## 3. Collate results — no physical execution

~~~powershell
uv run python scripts/report_dmdogo_v08_train_generalization.py --manifest "reports/dmdogo-v08-train-selection-r1.json" --results-root "artifacts/dmdogo-v08/train-generalization-r1" --output "reports/dmdogo-v08-train-generalization-r1.json"
~~~

The reporter distinguishes pending, invalid_identity, invalid_metrics,
did_not_complete and chart_complete. It reports per-chart Hits, X,
MAE, local replans, deferrals and overload. Missing/incomplete charts
are never credited with success, and the aggregate only includes
confirmed complete physical runs.

We do **not** require 100% X. The research aim is to locate which timing
motifs create plan infeasibility, late replan, physical deferral or motor
issues, without moving the benchmark.

## Human-likeness boundary

The current v0.8 gesture vocabulary and hand-exposure preference are
heuristic, not learned from human demonstration. Its full-chart Hit
cannot validate human-likeness or HUD-only observability because the
planner uses privileged exact event timestamps. Real human fingering
evaluation needs aligned human key/finger traces and an independent
out-of-sample split. This phase gathers generalization evidence only.

If a chart fails, record where and why. Any changed algorithm must
be separately versioned; never silently edit the frozen v0.8 baseline.
