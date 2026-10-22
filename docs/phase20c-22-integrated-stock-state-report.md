# Phases 20C–22 — Integrated Stock-state research report

## Executive result

The production scorer, Timing, Fingering, Motor, HUD contracts, and checkpoints
remain unchanged.  The research path now consists of:

```text
Frame-driven Stock reference
  -> explicitly evidenced visible-counter renderer
  -> JudgementObserver cumulative differencing
  -> set-valued Stock-floor state estimator
```

The fixed First town replay produces these verdicts:

- `FRAME_ACCURATE_REFERENCE_VALIDATED` for the declared 10 ms virtual-frame
  schedule;
- `STOCK_VISIBLE_FEEDBACK_PARTIAL` because automatic `FailMiss` visibility and
  live render latency are not established;
- `STATE_NONIDENTIFIABLE` under the conservative visible contract;
- `NO_SAFE_INTERVENTION_ESTABLISHED` under the frozen 44 ms Motor lead.

These verdicts do not claim parity with the currently installed Stock build.

## Phase 20C: scheduling and scorer comparison

The audited control-flow order is represented as time/angle update,
`CheckPostHoldFail`, automatic transition/fail, queued input, floor transition,
and notification.  A KeyDown-free frame can now fail or advance.

With NoFail, Phase 20B's 54/54 target/category parity is preserved.  The 50
KeyDown backlog remains: ID 38 is consumed by event 37, and ID 89 by event 88.
Event 89 automatically advances at the 5.320 s virtual update; ID 90 is then
judged against event 90.  In normal-fail mode event 36 crosses its strict fail
boundary at 3.313823567 s and stops on the next declared virtual update at
3.320 s.  The older 3.355 s result was an artifact of checking only on the
next KeyDown.

This is frame-accurate relative to the declared virtual grid, not a measurement
of Unity's actual update phase.

## Phase 21: what is actually visible

DLL evidence confirms synchronous internal `AddHit` count mutation, hit-text
calls, reset, and checkpoint rollback.  The live cumulative counter surface
and immediate category increments are user-confirmed evidence.  The precise
labels, automatic no-KeyDown `FailMiss` display behavior, display latency, and
same-frame render order are still unknown.

The renderer therefore supports an explicit evidence contract.  Unverified
categories can be withheld, and its output contains only cumulative labels and
capture/display times.  Batched events preserve counts but do not claim order.

Under an optimistic fixture, event 36's automatic transition at 3.250 s would
be observed at 3.260 s.  That is only about 1 ms before ID 38's 3.261 s
physical KeyDown.  Without visible automatic `FailMiss`, the first anomaly is
the `very_late` delta observed at 3.280 s, after the input.  Event 89's
automatic transition is observed at 5.330 s in the fixture, after ID 90's
5.319 s physical KeyDown.

## Phase 22: Stock-floor hypotheses

`StockPersistentJudgementStateEstimator` is independent of the Phase 19
repository FIFO estimator.  It retains possible sequential Stock floor
cursors, own-input attribution, visible automatic transitions, hidden
automatic-transition branches, and an explicit UNKNOWN search remainder.
Time alone never advances its floor.  `too_early` and `too_late` do not advance
the Stock cursor.  Counter reset or invalid observation widens uncertainty.

The optimistic fixture (automatic `FailMiss` asserted visible) contains a
hypothesis matching all 51 saved KeyDown-to-target pairs.  This is a
conditional model check, not live evidence.  At the simultaneous 5.320 s
`perfect + fail_miss` capture, within-frame order remains ambiguous even though
both transitions move beyond the fixed evaluation window.

The conservative replay first exposes alignment risk at 3.280 s.  It never
falsely reports a unique floor (`false_unique_identification_count = 0`).  The
hypothesis cap is reached, but an UNKNOWN remainder is retained, so truncation
cannot manufacture identification.  The exact saved attribution is therefore
not claimed to be recovered from counters alone.

## Event 89 intervention assessment

Visible feedback can establish that a mismatch is *possible*, but not safely
select a corrective action:

- optimistic warning margin before ID 38 physical KeyDown: about 1 ms;
- frozen Motor preparation lead: 44 ms;
- conservative first anomaly: after the input;
- a category delta has no target identity;
- ID 90 resynchronisation and event-89 automatic transition share a capture
  interval whose internal order is not visible.

Canceling, delaying, or adding a KeyDown on this evidence would either be too
late or risk destroying a valid hypothesis.  No production action was added.

## Evidence and remaining limitations

Online estimator inputs are limited to Timing local IDs/due intervals and
lifecycle, own KeyDown history, visible counter deltas, past hypotheses, and
causal time.  True target indices, scorer cursor, `_used`, `_missed`, true due
times, and future feedback are absent.  Oracle comparison happens only after
replay.

Still requiring live verification:

- installed-build identity relative to the audited DLL;
- Unity Update/capture phase and latency;
- exact visible counter schema;
- whether automatic `FailMiss`, invalid inputs, multipress, and checkpoint
  rollback are rendered in that surface;
- an independently visible game-stop signal for normal fail.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_stock_persistent_judgement_state_estimator_v03.py tests/test_audit_phase22_stock_estimator_v01.py
.\.venv\Scripts\python.exe scripts/audit_phase22_stock_estimator_v01.py --output reports/phase22-stock-estimator-20261022-r1.json
```

Primary results:

- `reports/phase20c-frame-stock-state-20261022-r1.json`
- `reports/phase21-stock-visible-feedback-20261022-r1.json`
- `reports/phase22-stock-estimator-20261022-r1.json`
