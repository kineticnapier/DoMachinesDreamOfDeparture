# Phase 21 — Stock-visible judgement feedback

## Observation boundary

`VisibleCounterRenderer` is a research renderer boundary.  A reference state
machine may send privileged judgement notifications into it, but the Student
receives only `JudgementCounterFrame`: capture/display timestamps and a fixed
set of cumulative visible integers.  Target ordinal, floor state, target time,
and scorer cursor are absent.  `JudgementObserver` then causally differences
those frames; it does not recover within-capture ordering.

The audited DLL establishes that `scrMarginTracker.AddHit` mutates internal
category counts synchronously, judgement paths invoke
`scrHitTextManager.ShowHitText`, reset clears counts, and checkpoint rollback
decrements counts.  Separately, the user has confirmed live cumulative
judgement counters and immediate visible category increments.  The exact
visible label set, whether a no-KeyDown automatic `FailMiss` appears in that
surface, pixel/render latency, and same-frame display ordering remain
unverified.  Internal counts are therefore not silently promoted to a Stock
HUD contract.

## Fixed replay

`reports/phase21-stock-visible-feedback-20261022-r1.json` evaluates two
contracts:

- an all-category fixture, explicitly marking `fail_miss` as fixture-only;
- a conservative surface that withholds automatic `FailMiss` because its live
  visibility has not been established.

Using a declared 10 ms capture latency, the event-36 automatic transition at
3.250 s would be observable at 3.260 s in the fixture, only about 1 ms before
ID 38's already-issued 3.261 s physical KeyDown.  If automatic `FailMiss` is
not visible, the first category anomaly is ID 38's `very_late` observation at
3.280 s, after that input.  The event-89 automatic transition at 5.320 s would
be observed at 5.330 s, after ID 90's 5.319 s physical KeyDown.

The tests also cover batched judgement increments, display/capture latency,
missing captures, counter reset, and exclusion of privileged identity fields.

## Result

Verdict: `STOCK_VISIBLE_FEEDBACK_PARTIAL`.

The causal counter pipeline is implemented and independently tested.  It can
report category multiplicity and observation intervals, but cannot name the
judged floor.  Even under the optimistic fixture, the relevant Phase 16
warning arrives far inside the frozen 44 ms Motor safety lead; it is not yet a
safe control intervention.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_judgement_observer_v01.py tests/test_stock_visible_judgement_feedback_v01.py tests/test_audit_phase21_stock_visible_feedback_v01.py
.\.venv\Scripts\python.exe scripts/audit_phase21_stock_visible_feedback_v01.py --output reports/phase21-stock-visible-feedback-20261022-r1.json
```
