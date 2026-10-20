# Hold Motor integration (H4)

## Scope and safety boundary

`HoldMotorOverlay` connects a verified Hold interval and an already selected
`PhysicalFinger` to the frozen 12-axis/16-contact Motor.  It does not predict
duration, choose fingering, inspect chart truth, or score a Hit.  Existing
ordinary Fingering output remains the base `ContactAction`; the overlay changes
only an explicitly occupied Hold axis.

The overlay:

- preserves the start and release due intervals without narrowing them;
- separately records command onset times, so evaluator due and Motor command
  time are not conflated;
- maintains the selected axis after physical KeyDown even when the ordinary
  policy asks to release it;
- passes commands for all unoccupied axes through unchanged;
- supports ordinary finger, thumb-tip/flat, and hand-edge topology;
- rejects axis collisions and implicit local-ID reuse;
- derives completion only from physical reset, never evaluator Hit/Miss;
- clears all occupancy on reset.

The normal Fingering Network and its checkpoint are unchanged.  Integration is
a hand-off: the Network/scheduler first selects the ordinary start stroke, then
the same `PhysicalFinger` is registered as occupied for the Hold interval.

## Provenance gate

Every plan is labelled:

- `CAUSAL_VISIBLE_INTERVAL`: an upstream causal sensor supplied the interval;
- `ORACLE_DIAGNOSTIC`: evaluator/chart timing used only in an explicit test;
- `UNVERIFIED`: insufficient evidence.

Unverified plans always fail closed.  Oracle diagnostic plans require
`allow_oracle_diagnostic=True`; that opt-in does not relabel them as HUD-only.
This matters because H3 proved exact positive Hold duration non-identifiable
from the currently verified icon observation.

## Physical behavior verified

The deterministic Motor test runs a Hold on `left_1` while an unrelated normal
tap uses `right_1`.  The physical edge order is Hold KeyDown, ordinary KeyDown
and KeyUp, then Hold KeyUp.  The Hold axis emits exactly one Down and one Up,
and its occupancy is released only after physical reset.

The overlay issues release at the supplied `release_command_s`; it does not
invent anticipatory release compensation.  Whether physical KeyUp falls inside
a stock judgement window is an H5 evaluator measurement, not assumed here.

## Remaining limitations

- No HUD-only exact release plan currently satisfies the H3 evidence gate.
- Consecutive-Hold hand-off remains runtime-unverified and is not automated.
- Thumb posture switching at rest retains the existing Motor simplification.
- This stage proves deterministic command/physical-state behavior, not stock
  judgement parity or full-chart success.
