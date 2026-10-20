# Hold judgment H2

`HoldJudgementStateMachine` implements only the H0-confirmed transitions and
receives explicit start/release windows. It has no access to chart truth,
rendering, Motor state, or future events.

The start transition receives ordinary-hit credit once. Completion receives a
separate Hold-completion credit and never ordinary-hit credit. Captured contact
IDs preserve KeyDown/KeyUp identity; releasing any captured start contact
triggers the stock release path.

Supported behavior:

- Normal: holding is required; completion is by captured-key release.
- CanHitEnd: holding is required; an additional KeyDown in the end window may
  complete the Hold.
- NoHoldNeeded: early release enters `WAITING_FOR_RELEASE` without automatic
  success; an end-window input is still required.
- missed start, early required release, and missed end are distinct failures.
- reset clears clock, keys, outcome, and event identity.

An unrelated KeyDown outside the end window is reported but does not fabricate
a Hold success. Damage/overload remains the existing scorer's responsibility.
Consecutive-Hold shared-boundary behavior raises `NotImplementedError` because
its exact stock key sequence is runtime-unverified. The contradictory
`CheckPreHoldFail` branch is not reproduced.
