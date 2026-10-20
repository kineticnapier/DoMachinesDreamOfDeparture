# Hold H9 — deterministic end-to-end validation

## Scope and conclusion

H9 connects only an H8 `MOTOR_READY` release forecast to the H4 Hold Motor
overlay.  The adapter preserves the Hold Event ID, Fingering choice, start and
release due intervals, and refuses diagnostic proxy observations.  It also
refuses a non-point release-command interval: selecting one time from such an
interval is not proven safe.

The deterministic contract fixture completes this sequence with the frozen
12-axis/16-contact Motor and H2 state machine:

1. the preselected `left_1` finger produces one physical/logical KeyDown;
2. the contact stays pressed;
3. `right_1` performs an independent KeyDown/KeyUp while the Hold remains down;
4. the forecasted release command produces one KeyUp;
5. H2 reaches `COMPLETED` with one Hold completion credit and no duplicate
   ordinary-hit credit;
6. the overlay releases ownership and a later ordinary `left_2` strike works.

This fixture is a deterministic integration test, **not** evidence that a
live Stock frame exposes duration.  Its duration observation is explicitly a
synthetic sensor-contract fixture.  It cannot be substituted for the missing
live capture.

## Fixed Train80 / 25 Holds

H5's three fixed charts and 25 Hold actions are preserved.  H8 has zero
independently verified Stock-pixel observations, so H9 performs zero production
end-to-end releases and records 25 safe abstentions.  H1's 25/25 code-level
timing parity is a separate claim and does not establish visual observability.

No chart `holdLength`, target time, or future action is passed to the Student.
No Oracle fallback exists in the adapter.

## Remaining gates

- Live Stock pixels and capture timing are still required to validate the H7
  duration sensor and the 44 ms release-preparation gate.
- The audited shader was not available, so generated mesh metadata cannot be
  treated as a visual observation.
- Consecutive-Hold boundary semantics remain runtime-unverified and H2 rejects
  `continues_to_hold=True`.
- Stock judgement windows remain evaluator-owned; the synthetic H9 tolerance
  is a plumbing fixture, not a measured game margin.
- Unknown future SetSpeed/Pause values still force abstention; the
  `DO_NOT_RELAX_SETSPEED_GATE` rule is unchanged.

Therefore the H9 verdict is `HOLD_LIVE_VALIDATION_REQUIRED`.
