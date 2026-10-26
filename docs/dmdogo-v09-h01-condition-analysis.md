# DMDOGO v0.9 Phase H0.1 condition analysis

## Scope and evidence

This audit explains the two frozen H0 review flags in the three saved v0.8
Train executions (10,231 events). It does not change the v0.8 planner,
runner, Timing, or Motor, and it does not label a flagged gesture as
unnatural or incorrect.

The analyzer reads the complete saved `gesture_chunks` and event assignments,
rather than the 25-item examples retained in the H0 summary. It verifies the
saved execution provenance and reproduces the H0 totals before producing the
enriched records. The thresholds remain exactly those declared by H0:

- `quick_same_hand_depth_jump`: adjacent, single-axis events on the same hand,
  IOI in `[0, 120]` ms, and absolute axis-depth change at least 2.
- `rapid_single_hand_sequence`: a maximal same-hand single-axis run, every IOI
  at most 120 ms, with at least 4 events.

`local_bpm_equivalent` is only `60000 / IOI_ms`. It is not chart BPM and does
not identify a musical beat subdivision.

## Results

| Flag | Detected | Like-for-like controls | Detected IOI median (IQR) | Control IOI median (IQR) |
|---|---:|---:|---:|---:|
| quick same-hand depth jump | 22 | 4,124 | 81.250 ms (44.118–81.250) | 83.333 ms (29.411–85.715) |
| rapid single-hand sequence | 109 | 3,494 runs | 46.875 ms (41.667–83.333) | 83.333 ms (28.572–88.235) |

For the rapid rows, the IOI distribution contains 327 intervals from the 109
runs. All detected runs contain exactly four events. The controls are the
otherwise matching maximal runs with two or three events. Consequently, the
detected/control comparison is descriptive and is not an independently
matched causal estimate.

The reciprocal-IOI medians were 738.462 versus 720.003 for quick flags and
1280.000 versus 720.003 for rapid runs. The detected depth-jump distance had
median 2 (range 2–3). The rapid runs had median total depth distance 3 (range
3–5).

## Gesture Chunk relationship

The central result is that the two flag types have different relationships to
chunking:

| Flag | Within one chunk | Crosses a chunk boundary | Cross-boundary share |
|---|---:|---:|---:|
| quick same-hand depth jump | 0 | 22 | 100.0% |
| rapid single-hand sequence | 48 | 61 | 56.0% |

Thus every quick depth jump is a join between Gesture Chunks. Rapid
single-hand action occurs in both places: 48 sequences are wholly inside a
chunk and 61 span at least one boundary. It is therefore inaccurate to
describe all rapid sequences as either an intra-chunk or a cross-chunk
phenomenon.

The relationship is chart-dependent:

| Saved Train chart | Events | Quick (within/cross) | Rapid (within/cross) |
|---|---:|---:|---:|
| ALL MY FELLAS | 1,540 | 0 / 10 | 23 / 0 |
| ADOFAI Regular Dan — 2nd Dan | 3,637 | 0 / 10 | 15 / 18 |
| Megamix2.7 | 5,054 | 0 / 2 | 10 / 43 |

Among detected rapid sequences, the touched chunk-family counts were
`same_hand_roll=91`, `paired_exchange=40`, and `single=39`. Counts can exceed
109 because one cross-boundary sequence can touch multiple families.

## Hand and axis tendencies

Quick flags comprise 6 left-hand and 16 right-hand transitions. The most
frequent transitions were `right_3 -> right_1` (7) and
`right_4 -> right_1` (7). Rapid runs were balanced more closely: 59 left and
50 right. Their principal four-event patterns were
`left_1 -> left_2 -> left_3 -> left_4` (56) and
`right_1 -> right_2 -> right_3 -> right_4` (43).

These are axis-order observations, not measurements of human finger travel.
The reported movement distance is the absolute change in the repository's
axis-depth index and is only a proxy.

## Overlap

Ten of the 22 quick flags occur inside a detected rapid sequence (45.5%). Ten
of the 109 rapid sequences contain a quick flag (9.2%). The overlap is real
but does not make the two detectors interchangeable: twelve quick flags are
outside rapid runs, while 99 rapid runs contain no quick depth jump.

## Candidate explanations, not causal conclusions

1. The 22/22 chunk-boundary result makes chunk joining or independent
   per-chunk candidate selection a plausible source of the quick depth
   changes. It does not establish that those joins are undesirable or that
   changing the planner would improve them.
2. Rapid flags are associated with denser intervals, but the detector also
   selects runs by length. The lower median IOI must not be interpreted as an
   isolated IOI effect.
3. Every detected rapid run has four events, matching the observed v0.8 chunk
   size ceiling. This is consistent with the chunk grammar and boundary joins
   shaping the flag population. The chart-specific within/cross split shows
   that no single boundary-only explanation covers all 109 cases.
4. The strong monotone `1 -> 2 -> 3 -> 4` patterns show that most rapid flags
   are regular axis sequences, not arbitrary switching. H0 is a review flag,
   not evidence that these gestures are physically or perceptually bad.

## Limitations

- No human demonstration or preference label is available, so human-likeness
  remains unverified.
- Controls share the same coarse eligibility conditions but are not matched
  for chart, local density, chunk family, or sequence length.
- Reciprocals of very short IOIs magnify numerical differences and must not be
  read as chart BPM.
- The saved output describes the deterministic v0.8 result on three Train
  charts only. It is not a Validation20/FINAL or population-level result.
- Axis-depth distance is a discrete proxy, not Cartesian finger travel,
  effort, fatigue, or injury risk.

## Reproduction

```powershell
uv run python scripts/analyze_dmdogo_v09_h01_conditions.py `
  --manifest "reports/dmdogo-v08-train-selection-r2.json" `
  --verified-report "reports/dmdogo-v08-train-report-r4.json" `
  --h0-report "reports/dmdogo-v09-h0-train-style-r1.json" `
  --results-root "artifacts/dmdogo-v08/train-generalization-r2" `
  --output "reports/dmdogo-v09-h01-train-conditions-r2.json"

uv run pytest -q `
  tests/test_dmdogo_v09_h01_conditions.py `
  tests/test_dmdogo_v09_style_proxies.py `
  tests/test_dmdogo_v09_style_report.py
```

The JSON includes every detected record and control opportunity with event
indices, due times, IOIs, reciprocal-IOI values, axes, movement proxy, and
the full touched-chunk descriptors.
