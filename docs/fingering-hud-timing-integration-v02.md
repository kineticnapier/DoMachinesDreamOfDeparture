# HUD-only Timing -> Fingering -> 12-axis Motor integration design v0.2

## Decision

Integrate the deterministic Timing System v1.4 first, behind a versioned
forecast interface.  Do not deploy neural Timing Student v3: its frozen
Development20 result was 3/1340 practical matches with 221 false forecasts,
and its countdown diagnostic showed 21.36 ms raw due MAE and 9.90 ms step MAE.
The interface remains estimator-agnostic so a future Timing model can replace
v1.4 only after its own predeclared gate passes.

Keep Timing, Fingering, reservation scheduling, Motor control, and evaluation
as separate modules.  Exact AutoPlay time remains an Oracle comparator and
posthoc label source, never a fallback available to the HUD-only arm.

## Existing pieces to reuse

| Responsibility | Existing implementation | Decision |
|---|---|---|
| 263D causal HUD | `envs/n_key.py` encoder | Freeze legacy 8-key/263D semantics |
| HUD timing | `hud_timing_tracker_transition_v14.py` | First integrated estimator |
| local event identity | `CausalTimingTracker` / `CausalForecastReconciler` | Reuse truth-free reconciliation semantics |
| abstract contract | `planning/causal_contract.py` | Preserve; it is 8-key and must not be silently widened |
| 12-axis choices | `fingering_network_v01.py` | Freeze checkpoint and 65D context shape |
| 12-axis/16-contact physics | `contact_16k.py`, `contact_chart_16k.py` | Freeze constants and scorer |
| Oracle experiment | `audit_fingering_causal_early_v02.py` | Comparator only |

## Required observation boundary

The old 263D vector is specifically `239 + 3 * 8`: 24 Motor fields followed
by orbit/floors/HUD.  A raw 12-axis encoding would be 275D and would shift all
geometry offsets.  Treating it as 263D would silently change the old Student
contract and is forbidden.

The first implementation must therefore add a separate, explicitly versioned
`ContactTimingHUD263Adapter` for Timing v1.4 only:

- preserve the 263 positions and the current v2 BPM/geometry encoding;
- construct geometry, orbit, BPM, and visible event-marker fields from the
  current contact-chart observation only;
- reserve the first 24 entries as a documented neutral legacy Motor slice;
- assert that Timing v1.4 never reads those 24 entries;
- reject this adapter for MLP/GRU/v3 inference, because those models may use
  the old 8-key Motor slice and a neutral/projection substitute would not be a
  fair input;
- do not modify `encode_n_key_hud_real_chart_observation` or any old
  checkpoint metadata.

This keeps the HUD-only timing path causal without pretending that twelve
physical axes fit the frozen eight-key observation.  A future neural 12-axis
Timing input requires a new feature version and new training protocol; it is
outside this integration.

## Versioned forecast contract

Add `src/dmdod/planning/timing_fingering_contract_v02.py`; do not mutate the
8-key `causal_contract.py` types.

```text
TimingEventToken
  local_id: monotonically increasing tracker ID (never a target/floor ordinal)
  due_local_s: local control clock deadline
  remaining_s: due_local_s - step * 0.010
  chord_size: 1..8
  confidence: optional calibrated evidence, not automatically 1.0
  uncertainty_s: optional conservative timing bound
  confirmations: causal forecast confirmations
  confirmed_now: whether the current HUD reconfirmed it

TimingFrameUpdate
  step, status, covered_until_s
  pending tokens
  new_ids, reused_ids, canceled_ids, elapsed_ids
  abstention_reason
  source/version
```

`due_local_s` uses only the monotonic policy clock `step * 10 ms`.  The
contract contains no chart time, floor number, target ordinal, true timestamp,
judge window, teacher choice, or evaluator result.  Equal-time forecasts are
one chord token; multiple non-simultaneous events inside one 10 ms control
interval remain separate tokens.

Timing v1.4 confidence must not be called a calibrated probability.  Emit its
geometric status and an optional deterministic uncertainty bound separately.
If no validated bound exists, leave it absent and use the explicit
`eligible`/`eligible_partial`/abstention status.

## Data flow

```text
current visible contact-chart scene
        |
        +--> ContactTimingHUD263Adapter -- current/past 263D only
        |             |
        |      Timing v1.4 + local reconciler
        |             |
        |      TimingFrameUpdate (local IDs/deadlines only)
        |             |
current 12-axis Motor observation ----> Fingering reservation planner
                                      (frozen Network v0.1 ranker)
                                                   |
                                      persistent physical reservations
                                                   |
                                      10 ms action / 1 ms Motor
                                                   |
                         evaluator truth ----------+--> posthoc metrics only
```

The 12-axis Motor observation is the planner's own current state, not a Timing
input and not hidden chart information.  The Timing estimator and Fingering
ranker remain independently replaceable and independently scoreable.

## Frozen Fingering checkpoint adapter

`FingeringRanker v0.1` expects a 65D context: 56 current Motor fields plus
three `(relative_time, chord_size, present)` event slots.  Preserve that input
shape and normalization exactly.

Add `encode_context_from_timing_update(...)` that converts the first three
pending tokens into those nine event fields using
`max(0, due_local_s - current_local_s)`.  It must be bit-for-bit equal to the
existing `encode_context` when supplied the same relative times and chord
sizes.  Do not rename or reinterpret old checkpoint tensors, and do not pass
confidence into the frozen network because that would change its input spec.
Confidence/uncertainty gates belong in the outer reservation planner.

## Decision and actuation deadlines

The existing Motor has about 39--42 ms cold KeyDown latency and the frozen
controller uses a 44 ms normal lead.  The Causal Lead experiment decides up to
69 ms ahead.  Timing v1.4 exposes at most 80 ms, so the available preparation
margin is small and must be measured, not assumed.

For each token, compute without evaluator data:

```text
decision deadline = predicted due - planning/actuation lead - uncertainty guard
```

- Reserve a finger at first stable acquisition, before this deadline.
- Start the existing Motor command according to its frozen 44 ms rule (and a
  separately selected/frozen Causal Lead rule only in a named comparison).
- If first acquisition is already past the deadline, mark
  `insufficient_observable_lead`; do not backfill with Oracle time.
- If the uncertainty interval makes both avoiding TooEarly and meeting the
  physical deadline impossible, abstain and record `uncertain_deadline`.
- Values for confirmation count, uncertainty gate, and retiming tolerance
  must be fixed on Train fixtures before any Validation comparison.

## Persistent reservation state machine

The failure analysis shows that target-resolution-based cleanup is unsafe.
Reservations must follow physical intent, never evaluator target state:

```text
proposed -> reserved -> armed/in_flight -> keydown_observed -> releasing -> free
              |                 |
              +-> canceled      +-> canceled_in_flight -> releasing -> free
```

Rules:

1. Key reservations are keyed by Timing `local_id`, not target ordinal.
2. A reconfirmed token may update its due time before arming.  After arming,
   retiming is bounded; the chosen finger is not silently swapped.
3. A positive-moving, activated, pressed, or not-yet-reset axis is unavailable
   even if its forecast token was canceled or elapsed.
4. An eligible update may cancel a contradicted token.  Cancellation before
   arming frees the reservation; cancellation in flight commands brake/release
   and keeps the axis quarantined until reset.
5. An estimator abstention does not assert that an event vanished.  Existing
   reservations become `unconfirmed`; it creates no new reservations and does
   not immediately cancel old ones.
6. `keydown_observed` comes from the public physical contact transition.  It
   closes that physical intent but is not evidence of an evaluator Hit.
7. Multiple events inside one control interval retain distinct reservations.
   Chords reserve the required distinct physical axes atomically.
8. Never free a reservation because `env._used[target]` changed.  `_used`,
   `_missed`, `judgement_trace`, and AutoPlay ordinals are evaluator-only.

These rules directly prevent the in-flight reuse found at events 507, 559,
and 642 without changing the frozen ranker's weights.

## Updates, uncertainty, and special events

- **Ordinary/Twirl:** use v1.4 only when both visible-direction hypotheses
  agree or observed motion disambiguates direction.  Twirl beyond a verified
  boundary remains partial/abstained.
- **SetSpeed:** a future SetSpeed/Pause floor permits only the incoming event.
  Inside SetSpeed, use visible Tile BPM only to the current floor exit; never
  extrapolate averaged BPM beyond it.
- **Pause:** while motion is stationary the remaining pause is unobservable.
  Mark reservations unconfirmed and add none.  Resume only after a full
  causal 10 ms motion observation passes v1.4's consistency guard.
- **Full turn:** require prior-frame signed motion; high-speed direction alias
  abstains.
- **Midspin/MultiPlanet/unsupported markers:** abstain.  No Oracle fallback.
- **High BPM/density:** preserve multiple tokens.  If visible horizon is
  shorter than Motor preparation latency, record insufficient lead rather
  than treating a late prediction as success.

## Leakage enforcement

Implement separate policy and evaluator objects.  The policy loop may receive
only `(step, hud263, TimingFrameUpdate, current 12-axis Motor observation,
its own reservations)`.  The evaluator receives the completed action/event
trace plus exact AutoPlay truth afterward.

Tests must fail if a policy-facing dataclass contains any of:
`target_index`, `target_ordinal`, `floor_index`, `target_time_s`, `chart_time`,
`judgement`, `_used`, `_missed`, or `segment.targets`.  Prefix-invariance tests
must confirm that appending future HUD frames cannot change already emitted
updates.  Runtime logs must record forecast provenance separately from
posthoc truth matches.

## Fair Oracle comparison

Use three arms with identical chart, initial Motor state, control/physics
rates, checkpoint, policy style, and scorer:

1. Oracle Timing + frozen Fingering + frozen Motor (ceiling/reference).
2. HUD v1.4 Timing + frozen Fingering + frozen Motor (primary integration).
3. HUD v1.4 Timing + deterministic CenterFirst + frozen Motor (attribution).

The Oracle adapter must emit the same versioned token type and use the same
reservation scheduler; only the timing source differs.  Report separately:

- Timing: unique event recall/precision at 1/5/10/20 ms, false forecasts,
  abstention, first-seen lead, and special-event cohorts.
- Scheduling: acquired-too-late, unconfirmed/canceled, retimed after arm,
  reservation collision, and in-flight cancellation counts.
- Physical: Hit/Miss/TooEarly/Overload, KeyDown count, X accuracy, Timing MAE.
- Attribution: forecast absent/wrong, insufficient lead, fingering
  infeasible, Motor late/no retrigger, or evaluator association mismatch.

Repeated frame forecasts count once by local ID.  Validation20 is already
opened and may be descriptive only; it cannot select gates or parameters.

## Implementation order for the next session

1. Add the versioned contract and leakage/unit tests.
2. Add the v1.4 contact-HUD adapter with fixed geometry-index and no-read-first-
   24 assertions.  Reject neural checkpoints on this adapter.
3. Add Oracle and v1.4 adapters producing identical token types; test local-ID
   reconciliation, multiple events, cancellation, and abstention.
4. Add `encode_context_from_timing_update` and prove frozen-context parity.
5. Implement the persistent physical reservation state machine and tests for
   the exact in-flight reuse pattern from events 507/559/642.
6. Run deterministic 120/240/480 BPM ordinary fixtures; no training.
7. Run one predeclared short Train80 segment containing SetSpeed/Twirl/Pause.
8. Only after all gates pass, run one fixed full Train80 chart.  Do not open a
   new holdout or rerun Validation20 broadly in this stage.

Likely change locations:

- new `src/dmdod/planning/timing_fingering_contract_v02.py`
- new `src/dmdod/planning/contact_timing_hud_v01.py`
- new `src/dmdod/planning/persistent_fingering_scheduler_v01.py`
- small parity helper in `src/dmdod/training/fingering_network_v01.py`
- new integration runner under `scripts/`
- dedicated tests; no checkpoint/config/physics edits

## Predeclared minimum experiment gates

The next implementation is PASS only if all of the following hold:

1. **Causality:** forbidden-field and prefix-invariance tests pass; the HUD arm
   never calls AutoPlay/compiled target APIs before actions are complete.
2. **Compatibility:** the Oracle-token adapter reproduces existing frozen
   context tensors and decisions exactly on the smoke fixture.
3. **Reservation safety:** zero double-booked/in-flight-reused axes and zero
   evaluator-state reads; the three diagnosed counterexample fixtures retain
   an axis until physical reset.
4. **Simple HUD integration:** on fixed 120/240/480 ordinary fixtures, every
   v1.4 forecast with at least the frozen required Motor lead is uniquely
   reconciled, produces no false forecast, and the integrated Motor has zero
   Miss/TooEarly/Overload.  Insufficient-lead events are reported separately,
   not counted as supported successes.
5. **Special behavior:** Pause/Midspin/MultiPlanet ambiguity causes no new
   action; SetSpeed/Twirl/full-turn outputs match v1.4's existing fixture
   contract and never extrapolate beyond verified coverage.
6. **No hidden overall claim:** full-chart Hit rate, forecast coverage, and
   lead distribution are all reported.  Perfect performance on only the
   eligible subset is not called full-chart success.

Any Oracle parity drift, false HUD event, reservation race, truth access, or
physical Miss among sufficiently early supported fixture events is FAIL and
stops expansion.  No threshold search or retraining follows automatically.

