# DMDOGO v0.9 Phase H0.6 — improved forced-suffix attribution

## Scope

H0.6 replays the frozen v0.8 planner on the same three saved Train charts and
explains all 22 H0.5 non-quick interventions, with emphasis on the seven whose
conditional forced-suffix cost was lower than the v0.8 winner.

No planner, score, beam width, segmentation, Motor, Timing, runner, or
hyperparameter is changed. No physical Motor episode is executed. Each target
is an independent counterfactual; the 22 alternatives are never combined into
one supposed executable chart.

Before producing a result, the audit:

1. validates the manifest, chart SHA256 values and saved v0.8 provenance;
2. reconstructs all 10,231 unrounded Oracle events;
3. reproduces H0.1, H0.4 and H0.5 exactly;
4. repeats H0.5's width-24 forced suffix search while retaining its terminal
   chunk/axis path;
5. compares that path event-for-event with the exact saved v0.8 assignment.

## Result overview

| Saved Train chart | H0.5 targets | Lower forced final cost |
|---|---:|---:|
| ALL MY FELLAS | 10 | 3 |
| ADOFAI Regular Dan — 2nd Dan | 10 | 4 |
| Megamix2.7 | 2 | 0 |
| **Total** | **22** | **7** |

The seven improvements range from 0.293969 to 0.711016 objective units
(median 0.619279). This is improvement in the frozen, untrained six-term
heuristic only. It is not evidence of human preference or Motor success.

## Beam loss and propagation

| Chart / flag pair | Join rank | First loss | Survived after join | Loss | Final delta | Changed chunks / events |
|---|---:|---:|---:|---|---:|---:|
| ALL / 1089–1090 | 2 | 632 | 5 chunks | beam pruning | -0.618274 | 5 / 8 |
| ALL / 1112–1113 | 2 | 646 | 5 chunks | beam pruning | -0.619279 | 5 / 8 |
| ALL / 1135–1136 | 2 | 660 | 5 chunks | beam pruning | -0.701940 | 27 / 49 |
| 2nd Dan / 720–721 | 24 | 336 | 2 chunks | beam pruning | -0.693311 | 4 / 16 |
| 2nd Dan / 728–729 | 26 | 336 | 0 | initial pruning | -0.711016 | 1,264 / 2,906 |
| 2nd Dan / 1237–1238 | 24 | 583 | 2 chunks | beam pruning | -0.293969 | 1,013 / 2,385 |
| 2nd Dan / 1245–1246 | 26 | 583 | 0 | initial pruning | -0.313445 | 9 / 21 |

Five lineages entered the original beam and were pruned two or five chunks
later. Two ranked 26th and were outside width 24 immediately. The join rank
alone does not distinguish improved from non-improved cases: the 15
non-improvements span ranks 2–41 and have median rank 20, while the seven have
median rank 24.

The downstream effect is heterogeneous. Five cases reconverge with v0.8 after
4–27 changed chunks. The 728–729 and 1237–1238 interventions never permanently
rejoin before the end of 2nd Dan and therefore change thousands of event
assignments. Their final gain cannot be interpreted as a local seam fix.

## Why the seven final costs improve

The summed six-term deltas across the seven independent runs are:

| Frozen objective term | Sum of alternate − v0.8 |
|---|---:|
| gesture preference | -2.250000 |
| hand exposure | +0.578176 |
| axis reuse margin | +0.070588 |
| intra-gesture hand changes | -3.150000 |
| intra-gesture depth distance | 0.000000 |
| chunk-join transition | +0.800000 |

The improvement is therefore dominated by lower gesture-preference cost and
fewer intra-gesture hand changes downstream. Higher hand exposure, reuse
margin, and chunk-join cost oppose the improvement. The quick jump itself is
an inter-chunk relation, so the intra-gesture depth term does not change.

At the forced join, the three ALL alternatives tie v0.8 locally; the four 2nd
Dan alternatives are locally worse by 0.026711–0.037372. Local non-jump
selection alone does not explain the lower terminal cost.

### Fixed-original-suffix counterfactual

H0.6 adds a second counterfactual: force the same non-quick seam, then keep
every later v0.8 chunk unchanged instead of replanning the suffix. All seven
fixed suffixes remain abstract-reuse feasible, but all seven finish **more
expensively**, by +0.018750 to +0.160599. The lower costs appear only when the
downstream suffix is replanned; replanning contributes -0.439958 to -0.869520
relative to the fixed-suffix control.

This supports a bounded-search/path-dependence explanation. It does not prove
that a wider beam would select these exact paths or that any is globally
optimal, because the forced search excludes every prefix that does not descend
from the selected alternative.

## Flag displacement

Every one of the seven target quick jumps is removed. Across the seven
independent results:

- 15 quick flags are removed and 2 new quick flags appear;
- 20 rapid single-hand runs are removed and 12 new runs appear;
- the new quick flags are 724–725 and 1241–1242, four events after the
  corresponding forced seams.

Thus non-quick forcing can relocate the review condition rather than simply
eliminate it. The report deliberately calls these *added/removed locations*,
not movement of an identical gesture. The totals also must not be combined
across interventions because several cases affect overlapping regions.

The three ALL cases add no quick flag. They introduce later rapid runs at
1094–1097, 1117–1120, and, for the third case, 1140–1143 plus 1156–1159.
The four 2nd Dan cases substantially rearrange rapid-run boundaries around the
forced region.

## Axis reuse and hand-load proxies

All seven alternatives preserve the abstract 80 ms reuse constraint; no
under-80 ms reuse is introduced.

- Five keep the chart-wide minimum reuse interval unchanged.
- 720–721 and 1237–1238 reduce it from 160.000 ms to 132.353 ms and each add
  one under-150 ms reuse. This explains the +0.035294 reuse-margin component
  in those two cases.
- The three ALL cases preserve total left/right use, minimum reuse, maximum
  single-hand run, and add no under-150 ms reuse; they reduce pure-hand
  switches by two.
- 2nd Dan 728–729 shifts four axis uses from left to right; 1237–1238 shifts
  nine. The other two preserve aggregate left/right use.
- Pure-hand switch deltas for the four 2nd Dan improvements are +2, +3, +3,
  and +4. A lower heuristic cost is therefore not synonymous with fewer hand
  switches or a lighter hand load.

These are abstract axis-use and exponentially decayed exposure proxies. They
are not biomechanical force, fatigue, travel distance, or physical Motor
telemetry.

## Improved versus non-improved cohort

The seven improved cases and fifteen non-improved cases have similar local
join penalties: mean +0.018309 versus +0.019423. Their original-beam survival
is also not cleanly separated (means 2.71 versus 2.07 chunks). The decisive
observed difference appears in the forced downstream solution:

- improved cohort: mean final delta -0.564462 and mean downstream replanning
  gain -0.658242;
- non-improved cohort: mean final delta +0.021100 and mean downstream
  replanning gain -0.069357;
- improved paths gain gesture-preference and intra-gesture-hand-change cost;
  the non-improved cohort has zero aggregate change in both terms.

This is an observational contrast inside a selected set of 22 seams, not a
causal classifier. No threshold or weight was fitted from it.

## Limitations and unresolved questions

- The forced search population differs from the original global beam. A lower
  conditional result is a bounded-beam discrepancy, not global optimality.
- Two improvements alter nearly the entire remaining chart; their local cause
  cannot be isolated from suffix phase/tie behavior by this audit alone.
- The six costs are untrained priors. Human-likeness remains unverified.
- Only one H0.4-selected non-quick candidate is tested per seam; other
  alternatives are not searched.
- No physical Motor is run. Live eligibility, Hit/Miss, X accuracy and Timing
  error are unknown for every counterfactual.
- Because interventions overlap, the seven improvements cannot be applied
  simultaneously by summing their reported gains.

The next justified experiment, if desired, is an abstract wider-population
search or a predeclared physical pilot for a small local-reconvergence subset.
Neither is performed in H0.6 because beam-width/hyperparameter changes and
physical Motor execution are explicitly out of scope.

## Reproduction

```powershell
uv run pytest -q `
  tests/test_dmdogo_v09_h06_improved_suffix.py `
  tests/test_dmdogo_v09_h05_nonquick_suffix.py `
  tests/test_dmdogo_v09_h04_beam_decisions.py

uv run python scripts/audit_dmdogo_v09_h06_improved_suffix.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h0-report reports/dmdogo-v09-h0-train-style-r1.json `
  --h01-report reports/dmdogo-v09-h01-train-conditions-r2.json `
  --h04-report reports/dmdogo-v09-h04-frozen-beam-r1.json `
  --h05-report reports/dmdogo-v09-h05-nonquick-suffix-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2 `
  --output reports/dmdogo-v09-h06-improved-suffix-r2.json
```

The output is create-once. Use a new revision filename for a second run.
