# Phase 16 — First town 12s Motor A/B (user-run audit)

**Verdict: `MOTOR_AB_REGRESSION_OR_INCOMPLETE`**. This report records
statistics from the user-uploaded local file
`phase16-firsttown12-motor-ab-20261020-r1.json`. Its full JSON
has not been independently rerun here. Scope is one SHA-pinned Train chart
(first 12.0 seconds, 220 point targets), not Validation20, FINAL, Hold or
pixel recognition.

## Frozen result

| Statistic | v1.5 recovery OFF | Midspin+Twirl+Checkpoint ON |
| --- | ---: | ---: |
| Hit | 162/220 (73.64%) | 207/220 (94.09%) |
| Miss | 58 | 13 |
| X accuracy (separate scoring metric) | 48.00% | 66.09% |
| Physical KeyDown | 162 | 207 |
| Physical scorer timing MAE | 28.602361ms | 24.883172ms |
| Insufficient observable lead decisions | 42 | 13 |
| In-flight cancellations | 4 | 1 |
| TooEarly | 0 | 0 |
| Overload state/counter | false / 0 | false / 0 |
| Source conflicts | 0 | 0 |
| Lifecycle contract violations | 0 | 0 |
| Axis reservation contract violations | 0 | 0 |

The candidate **gained 46 individual Hits**, **lost 1 previously Hit
target**, and had **net +45 Hits** (+20.45 percentage points of Hit rate).
Only the `no_hit_regression` safety check failed; all eight other
checks passed. Do not enable candidate by default solely on the
aggregate improvement.

Candidate judgement counts: 102 perfect, 4 late_perfect, 101 very_late,
13 misses. The 101 very_late judgements indicate substantial residual
input-to-scorer alignment error even though Hit coverage improved.
Among 13 missed target due-times, nearest same-due decisions show 7
`insufficient_observable_lead` and 6 `assigned` cases.
A decision at a due-time is **not equivalent** to hitting that event,
because the scorer sequentially consumes its earliest unresolved target.

## The single lost Hit: Event ordinal 89

| Field | Recovery OFF | Recovery ON |
| --- | --- | --- |
| Target truth | 5.250000044s | 5.250000044s |
| Target 89 status | Hit (very_late) | Miss |
| Posthoc keydown attributed to target 89 | 5.292000s | none |
| Physical press at 5.253000s | targets old ordinal 88 | right_3, emitted to target 88 |

In the candidate, the policy scheduled local ID 89 (due 5.250000044s,
right_3) at step 519 (5.190s). It physically pressed right_3 at
5.253s. But the scorer attributed that press to **earliest unresolved
target 88**, due 5.205882397s, as very_late (+47.118ms).
The true ordinal 89 was still unresolved after that press; it was
subsequently marked missed. The next physical press is at 5.319s
(for local ID 90's due ~5.294118s), after target 89 had expired.

This is supported by the simulation's
`RealChartMotorEnv._next_target_index()` and `_score_event()`:
KeyDown acts on the first unresolved ordinal, not the nearest due
or the Timing lifecycle local ID. The interval around ~5.1–5.3s
has an existing backlog of previous targets being hit late.

**Do not patch the evaluator by reassigning the 5.253s press directly
to target 89**: that would change the scoring contract and artificially
erase the measured regression. It remains to determine whether
the scheduler can prevent the backlog, or must fail closed when a previous
target is unresolved. An observational/counterfactual trace with no policy
truth leakage is the next diagnostic, not arbitrary timing offsets.

## Next gates

1. Use frozen scorer to trace `judgement_trace`, scheduled local IDs,
   and physical keydowns around 5.08–5.34s, including target expiration.
2. Reproduce ordinal 89 loss as a **deterministic scorer association**
   test without modifying Stock parity or learner inputs.
3. If needed, compare scheduler-only changes behind opt-in, holding
   all Timing and Motor constants fixed.
4. Re-run First town and frozen bool safety smoke before any wider
   Train16 physical A/B. Continue to reject `no_hit_regression`
   until lost ordinal 89 is recovered or its trade-off is explicitly
   reviewed and accepted.

No code, policy weights, shader, image recognition, training,
checkpoint, Validation20, or FINAL was changed for this report.
