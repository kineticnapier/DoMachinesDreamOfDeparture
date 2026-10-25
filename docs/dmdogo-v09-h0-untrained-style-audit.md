# DMDOGO v0.9 — H0: untrained fingering-style diagnostics

## Goal

Continue development **without asking the user to record or play**.
v0.8 remains frozen. H0 uses saved physical results to describe gesture
selection in an auditable way, not to score human-like fingering.

The previously verified Train test completed 3/3 selected charts with
10,231/10,231 Hits, 4,635/4,635 gesture chunks preserved,
zero local replans and 15 deferred events. This is Motor feasibility
evidence, NOT demonstration that the same axes are human-preferred.

The base physical runner, gesture planner, lead compensation, Timing,
dataset split, checkpoint and all existing outputs are unchanged.

## Scope and interpretation

New source files:

- scripts/analyze_dmdogo_v09_style_proxies.py
- scripts/report_dmdogo_v09_style_proxies.py
- scripts/prepare_dmdogo_v09_human_input.py

All are read-only with respect to source physical results. Outputs are
create-once. No model training and no human recordings are required.

Proxies computed over scheduled due times and selected physical axes:

- Hand contact share and per-axis utilization.
- Number of short-gap same-hand vs cross-hand adjacent event pairs.
- Fraction of event axes matching the assignment eight events earlier.
- Peak per-hand contacts inside a sliding one-second interval.
- Gesture family distribution and planned-axis adoption fraction.
- Concrete review locations for fast same-hand runs, nearby large
  same-hand depth jumps and high short-window hand concentration.

Flags are **review candidate regions**, not claims of discomfort,
impossibility, actual finger motion or inhuman technique. Thresholds
are fixed exploratory descriptive parameters, not optimized or learned
human values. Physical axis 1 is only an ordinary-finger proxy; it is
NOT an observed human finger. No synthetic reference is relabeled as
a human demonstration.

## One-command analysis of the existing three Train runs

Keep the current `develop/dmdogo` branch and preserve every local change.
If local changes would block fast-forward, stop; do not reset/rebase/stash/
clean, or overwrite `prestige1.toml`.

First run focused tests after fetching safely:

~~~powershell
uv run pytest -q tests/test_dmdogo_v09_style_proxies.py tests/test_dmdogo_v09_style_report.py tests/test_dmdogo_v09_human_input.py
~~~

Then analyze the 3 previously saved, independently verified Train runs
(no new Motor executions, no Validation20, no FINAL):

~~~powershell
uv run python scripts/report_dmdogo_v09_style_proxies.py `
  --manifest "reports/dmdogo-v08-train-selection-r2.json" `
  --verified-report "reports/dmdogo-v08-train-report-r4.json" `
  --results-root "artifacts/dmdogo-v08/train-generalization-r2" `
  --output "reports/dmdogo-v09-h0-train-style-r1.json"
~~~

The aggregator rejects incompletely verified cases and rechecks the
per-case physical summary SHA, selection manifest SHA and execution
provenance. The report contains chart-level metrics, candidate regions,
totals and explicit caveats. The existing `r4` report is not modified.

To produce the full style report for only one chart, bypass the batch
aggregator and use the standalone analyzer with separate create-once
output:

~~~powershell
$r = "artifacts/dmdogo-v08/train-generalization-r2/e142af91b178"
uv run python scripts/analyze_dmdogo_v09_style_proxies.py `
  --summary "$r/summary.json" `
  --telemetry "$r/telemetry-v08.json" `
  --replay "$r/replay-v08.json" `
  --output "$r/style-proxies-v09-r1.json"
~~~

The standalone analyzer verifies matching and complete v0.8 artifacts,
but unlike the batch aggregator does **not** verify a sidecar. It is
suitable for diagnostic exploration; use the batch report for frozen
cross-chart comparison.

## Future human input (optional, NOT required for H0)

The repository already has
`scripts/compare_fingering_human_log_v011.py` with a parser for previous
FingeringRecorder key logs. Reuse that data instead of asking for a new
capture tool.

`scripts/prepare_dmdogo_v09_human_input.py` normalizes the previous
human key-down log format to a versioned JSON container. For an
already-existing capture with a known corresponding chart SHA:

~~~powershell
uv run python scripts/prepare_dmdogo_v09_human_input.py `
  --legacy-log "path/to/existing/fingering-log.txt" `
  --chart-sha256 "EXACT_64_HEX_SHA256" `
  --output "reports/dmdogo-v09-human-input-new.json"
~~~

This command is **only for a future existing real recording**. Do not
fabricate or synthetically generate a human demonstration to fill a gap.

Every imported record is labeled `declared_key_mapping_not_observed`.
Although legacy K01..K16 key labels are associated with finger names,
the recorder's key assignment does **not** prove the physical finger
pressed the key. The resulting clock is explicitly unaligned, and
the script refuses to declare an event-to-oracle correspondence.

Axes 5/6 in the old recorder belong to the earlier 12-axis proxy,
not to frozen v0.8's eight ordinary physical axes. Do not silently
convert these into v0.8 axes.

Later H1 needs verified clock alignment, false/missed press handling and
(optional) finger observation before use as supervised demonstration
labels. H0 does not train on or compare against this JSON.

## What H0 can and cannot decide

H0 may reveal patterns worth inspecting, including long same-hand runs
or repeated switching; those patterns may be **normal** for a person.
It cannot declare them wrong or infer real anatomy without human
reference data. Any learned preference model belongs to a separate
version and must be evaluated independently of the frozen v0.8 Motor.

No Validation20/FINAL, hyperparameter search or hidden human
performance claims are authorized here.
