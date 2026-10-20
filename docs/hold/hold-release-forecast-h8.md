# Causal Hold release forecast (H8)

## Contract

`hold_release_forecast_v01.py` computes a release interval from:

- a persistent Hold-start local ID;
- the start due interval;
- a pixel sensor's duration candidate set and observation timestamp;
- visible base-angle and Real-BPM intervals;
- an explicit assertion that no unobserved SetSpeed/Pause boundary is crossed;
- a frozen-Motor release-latency interval.

It accepts no chart object, Hold action, `holdLength`, floor ordinal, target
time, true release time, or future BPM.  Real BPM is treated as already
including pitch.  All uncertainty is propagated by interval arithmetic.

For duration `d`, visible base angle `a`, and Real BPM `b`:

```text
release delta = (a + d*2*pi) / pi * 60 / b
release command = release due - physical release latency
```

The conservative forecast lead is measured from the lower bound of the release
command interval to the latest required causal observation.  Motor-ready
requires at least 44ms lead and a command interval no wider than one 10ms
control step.

## Safety gates

- Multiple duration candidates: abstain.
- Missing/stale/invalid pixels or kinematics: abstain.
- Unobserved future SetSpeed/Pause: abstain with
  `DO_NOT_RELAX_SETSPEED_GATE`.
- Missing physical release latency: abstain.
- Excess uncertainty or under-44ms lead: abstain.
- H7 proxy evidence: math is available for diagnostics, but Motor-ready is
  forbidden.
- Only independently verified Stock pixel evidence may produce `MOTOR_READY`.

No HUD v5 vector is introduced.  H7 did not establish a live Stock pixel
sensor, so extending the 368D HUD would incorrectly promote a proxy to a real
observation.  Existing 263D/278D/368D definitions remain unchanged.

## Fixed Train25 outcome

The H5 evaluator knows all 25 durations, but those labels are not supplied to
this policy.  With no live Stock mesh capture, every fixed Hold lacks verified
pixel evidence and must abstain from production release control.  The report
may count these evaluator rows posthoc, but cannot convert them into forecasts.

## H8 decision

The interval estimator and safety gates are implemented and unit-tested.
Production HUD-only release remains blocked pending live Stock pixel evidence:

`HOLD_RELEASE_ESTIMATOR_READY_SENSOR_UNVERIFIED`
