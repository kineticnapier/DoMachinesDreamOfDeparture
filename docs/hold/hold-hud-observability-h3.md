# Hold HUD observability (H3)

## Result

The frozen 263D HUD has no Hold feature.  Its per-floor marker tuple is
`Twirl`, `SetSpeed`, `MultiPlanet`, `Pause`, `Checkpoint`, and
`SetFloorIcon`; Hold is absent.  Therefore chart JSON containing Hold does not
become a Student observation merely because the simulator parsed it.

The audited stock code does establish four rendered sprite classes:

| Position | duration = 0 | duration > 0 |
| --- | --- | --- |
| Hold start floor | `HoldArrowShort` | `HoldArrowLong` |
| following/release floor | `HoldReleaseShort` | `HoldReleaseLong` |

`scnGame` chooses Long for every positive integer duration.  Consequently the
icon stream distinguishes zero from positive duration, but does not distinguish
duration 1, 2, or 3.  It also does not directly expose an end timestamp.

`scrHoldRenderer.CreateMesh()` uses `holdLength` when constructing a path mesh,
so the loop count exists in rendered geometry in this audited code snapshot.
No live-game capture or robust pixel recognizer has yet demonstrated that the
integer loop count can be recovered under camera motion, overlap, skins, and
occlusion.  The v4 contract therefore does **not** expose it.

Final H3 classification: **exact Hold release time is non-identifiable from the
verified icon-only observation**.  This does not say that a future mesh sensor
is impossible; it says that such a sensor remains unverified.

## Implemented observation contract

`src/dmdod/features/hold_hud_v4.py` adds an independent renderer/sensor boundary.
The renderer may use internal Hold state to reproduce an already-visible stock
icon.  The Student observation contains only:

- causal render and observation timestamps;
- a validity bit per existing visible floor slot;
- no-Hold-icon-visible, start/release, and Short/Long classifications.

No duration, `holdLength`, absolute floor index, target time, or release time is
present.  Missing, stale, unrecognized, or future frames are never imputed.
An absent Hold icon is not proof of no Hold because another event icon may have
won stock's icon selection.  The contract therefore names this state
`NO_HOLD_ICON_VISIBLE`, not `NO_HOLD`.

HUD v4 is 368D.  It preserves the complete 278D Countdown HUD v3 as an exact
prefix, which in turn preserves the original 263D HUD as an exact prefix.
Nothing in the existing HUD263, Timing v1.4, or HUD v3 Composite is changed or
enabled by default.

## Counterfactual proof

At 120 BPM, otherwise identical charts with a Hold on floor 1 produce:

| Hold duration | icon pair | compiled release time |
| --- | --- | ---: |
| 1 | ArrowLong / ReleaseLong | 1.750 s |
| 3 | ArrowLong / ReleaseLong | 3.750 s |

The v4 icon observation vectors are identical while evaluator-only release
truth differs by 2 seconds.  A policy using only this verified observation
cannot safely choose between those release times.

## What is and is not established

Established by code audit and deterministic tests:

- stock icon selection for Short/Long start and release;
- causal renderer/sensor separation;
- HUD263 and HUD v3 compatibility;
- duration 1 versus 3 icon-only non-identifiability;
- explicit missing/stale/unknown handling.

Not established:

- live-game pixel recognition accuracy or first-visible time;
- reliable loop counting from the rendered mesh;
- icon visibility under every competing event/skin/camera configuration;
- the active stock Hold setting mode from pixels;
- exact release timing from HUD-only observations.

H4 may connect an explicitly supplied, verified Hold interval to Motor state,
but must not call an Oracle/chart-derived duration a HUD-only prediction.
