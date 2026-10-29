# DMDOGO H1.3 — Automated fingering quality-proxy audit

## Background and explicit stopping point

H1.0 inventoried all 22 previously observed H0.6 single-seam alternatives.
H1.1 recovered every full event-to-axis assignment on three Train charts
(10,231 events), with 7 lower scores under the frozen, *untrained*
six-term objective. H1.2 built a blinded human A/B viewer, but
requiring a manual rating of 22 cases (some with >2,900 changed events)
would be an undesirable bottleneck.

**H1.3 makes manual review optional.** It compares every recovered
source and alternative path automatically using deterministic,
predeclared descriptive proxies. It is intentionally not a
new physiological, human-preference, learned, or Motor quality model.

## Source and boundaries

Read only the preexisting locally saved:

- reports/dmdogo-h11-fingering-pairs-r1.json

Verify exactly 3 original Train charts, event counts 1,540 / 3,637 /
5,054, original candidate counts 10 / 10 / 2, and 7 strictly
lower heuristic alternatives. Require H1.1's declared frozen v0.8
commit and exact H0.6 source Git blob, the full event and Chunk
assignments, identical event IDs and due times, and zero unverified
Motor/human claims. Store the exact SHA256 of the H1.1 source.

No chart is recompiled, and no physical Motor, Timing controller,
HUD-only Student, training, Validation20 or FINAL is accessed.
Do not edit the frozen planner, runner, source charts, or earlier
report JSONs.

## Predeclared descriptive metrics

Every case independently compares original versus alternative
full-chart event-axis sequences with *the same* source due times.
Exploratory, uncalibrated thresholds:

| Metric | Calculation | Interpretation limit |
|---|---|---|
| Quick same-hand depth jump | One-axis adjacent same-hand events within 120ms and depth difference >=2 | Local abrupt axis displacement proxy, not hand geometry |
| Rapid same-hand run | At least four consecutive single-axis same-hand events, each gap <=120ms | Repeated hand use, not finger performance |
| Longest rapid run | Maximum event count in such a run | Descriptive, no calibrated fatigue threshold |
| Short same-axis reuse | Per-axis due-time interval <100ms, <150ms | Abstract press-gap proxy; independent of hardware |
| Minimum reuse interval | Smallest gap on a repeated axis | Numeric descriptor, not a performance prediction |
| Peak one-second hand contacts | Largest sliding 1,000ms hand contact count | Not fatigue calibrated |
| Peak half-second axis contacts | Largest sliding 500ms same-axis contact count | Not physical key frequency guarantee |
| Hand-usage imbalance | Absolute left-right contact count difference | Whole-chart proxy; no ergonomic target known |

The existing >=80ms same-axis reuse is checked as a **strict
abstract feasibility invariant**, not claimed physical viability.
All metrics are recomputed on both complete sequences; no scores are
derived from eyeballing H1.2 HTML.

## Triage — not a new optimization objective

For every descriptor where a larger value represents more of that
pattern, report the alternative-minus-original delta. A positive
delta is named 'increased pressure proxy', but **does not mean
proved human discomfort**. Negative means the proxy decreased.

Deterministic optional follow-up priority:

- P1: frozen heuristic score improves but one or more pressure
  proxies increase, or some proxies increase while others decrease.
- P2: >=500 changed event assignments, or some proxies increase
  without a higher-priority conflict.
- P3: all other descriptive cases.

The complete P1/P2/P3 census and optional top five are saved.
These priority tiers cannot be used as human-quality labels or
automatic safer-action guarantees. The *top five* is a starting
place for **optional** review, not a new required grading task.
Original and alternative paths remain 22 separate, potentially
overlapping counterfactual worlds; do not merge them or treat
their observations as statistically independent human samples.

## Result and failures

`scripts/audit_dmdogo_h13_automated_fingering_quality.py`
writes a create-once JSON report containing chart-wide source
metrics, all 22 alternative metrics, numerical deltas,
diagnostic disagreement, and ranked triage. It fails closed if
a candidate changes any frozen event identity, timestamp, chord,
event-axis sequence, chunk segmentation, event difference list,
score decompositions, reuse safety or input source identity.

The report explicitly sets `motor_executed=false`,
`human_reviews_required=false`, `human_reviews_observed=false`,
`human_likeness_verified=false` and refuses synthetic labels.

## Windows reproduction

Branch: develop/dmdogo. Preserve all local modifications, especially
configs/training/prestige1.toml. Do not reset/stash/rebase/clean.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h13_automated_fingering_quality.py

uv run python scripts/audit_dmdogo_h13_automated_fingering_quality.py `
  --h11-report reports/dmdogo-h11-fingering-pairs-r1.json `
  --output reports/dmdogo-h13-automated-quality-r1.json
~~~

The source and all earlier reports remain untouched. The new
report uses exclusive-create and will not overwrite an existing
result; for reruns use a new revision filename.

## Follow-on decision

Use the H1.3 output to understand which objective improvements
carry trade-offs in observable axis-assignment patterns.
**Do not** automatically conclude that the best heuristic or
least-proxy path is a natural human fingering. Further validation
could compare alternative physical Motor execution on the existing
Train charts only under a separately approved, provenance-verified
protocol. Until then, feasibility and human preference remain unknown.
