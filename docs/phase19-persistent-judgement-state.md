# Phase 19 — Persistent judgement state tracking

## Result

`REPOSITORY_FIFO_PERSISTENT_STATE_TRACKED_STOCK_PARITY_PARTIAL`

The research-only v02 estimator maintains a set of cumulative histories from
Timing local evidence, own Motor evidence, visible counter deltas, elapsed
time, and the preceding estimator state.  It is not connected to production
control and makes no KeyDown, delay, cancellation, Motor, network, checkpoint,
training, or hyperparameter change.

The Stock audit is in `docs/phase19-stock-judgement-audit.md`; the machine
report is `reports/phase19-persistent-judgement-state-20261021-r1.json`.

## State design

Every `PersistentStateHypothesis` retains:

- all KeyDown-to-intended-ID-to-judged-ID attributions accumulated so far;
- target resolutions by KeyDown, automatic Miss, or Timing cancellation;
- pending KeyDown IDs and possibly unresolved Timing local IDs;
- observation consistency, observation/reset/display gaps, and ambiguous
  within-interval label order;
- automatic-Miss possibilities and explicit model assumptions;
- `unknown_search_remainder`, which represents compatible states omitted by a
  predecessor scope, hypothesis cap, counter discontinuity, or model mismatch.

New evidence transitions only the current set.  Concrete histories eliminated
by an observation are never rebuilt from the complete event list at a later
KeyDown.  A reset preserves prior histories and adds an UNKNOWN remainder; it
does not clear state and claim alignment.

`UNCHANGED` means only “no visible increment in this captured frame.” It does
not eliminate delayed, hidden, or unconfirmed Miss events.  Judgement category
and target identity are distinct fields.

## Search-safety invariant

`max_predecessors=3` and `max_hypotheses=128` remain resource limits, not
identifiability theorems.  If either can omit a compatible state, v02 inserts a
persistent UNKNOWN hypothesis and returns `TRUNCATED`.  Identification requires
exactly one concrete, observation-consistent hypothesis **and** no UNKNOWN
remainder.  Keeping only one item because the cap is one therefore still cannot
produce `MODEL_IDENTIFIED`.

Generation is stopped at the cap rather than enumerating an unbounded product;
the stopped portion is represented by UNKNOWN.  This protects both runtime and
the coverage claim.

## Event 37–90 replay

The bounded replay uses the existing Phase 16 diagnostic and begins at 3.15 s.
Earlier KeyDowns are outside this bounded state and are not silently imported.
Within the replay, the online path sees only Timing local ID/due/status history,
own reservations/KeyDowns, Phase 17 counter deltas, elapsed time, and its prior
hypotheses.  Scorer target index, cursor, `_used`, `_missed`, and true target
time are joined only afterward.

Repository FIFO model results:

| Milestone | Persistent result |
|---|---|
| ID 37 abstention at 3.190 s | ID 37 remains a possible unresolved target; no input was reserved. |
| ID 38 / visible VeryLate at 3.270 s | One complete repository-model history maps KeyDown 38 to target 37. |
| ID 38–89 | The cumulative history retains the one-target offset rather than reconstructing an aligned state at each interval. |
| 4.070 s two-input/two-judgement interval | Motor edge order is retained; visible label order is enumerated. FIFO state advances between the two inputs. |
| ID 89 / 5.260 s | KeyDown 89 maps to 88 and ID 89 remains possibly unresolved before its repository deadline. |
| ID 90 / 5.320 s | Repository auto-expiry resolves 89, then KeyDown 90 maps to 90; this supports repository-model resynchronization. It is not Stock auto-Miss parity. |

Posthoc Oracle evaluation covers 51 KeyDowns with intended IDs 38–90 (there
are no emitted inputs for the abstained IDs). The actual target is contained in
the final repository-model hypothesis set for **51/51** inputs. Oracle fields
are absent from serialized online updates.

## Phase 18 comparison

Phase 18 recreated a local attribution set for each visible interval. It could
identify the important 38, 89, and 90 events but did not carry the complete
history forward, and its search limits could look like exhaustiveness.

Phase 19 changes the epistemic contract:

- cumulative hypotheses are transitioned rather than regenerated;
- later evidence can resolve earlier ambiguity without reviving discarded
  histories;
- multiple KeyDowns are applied sequentially inside each label-order branch;
- time can create pending/automatic-Miss branches;
- reset, invalid counters, missing display, and model contradiction retain an
  UNKNOWN remainder;
- incomplete predecessor or hypothesis search can never identify a state.

Accuracy is unchanged at 51/51 for this bounded repository-model case. The
material improvement is that “not established” remains represented as such.

## Counterexamples and tests

`tests/test_persistent_judgement_state_estimator_v02.py` covers:

1. a consuming target more than four predecessors back;
2. more than 128 compatible hypotheses;
3. ambiguity resolved by later observation;
4. non-revival of an eliminated history;
5. multiple internal states for the same visible history;
6. two KeyDowns and two judgements in one interval, including FIFO advancement;
7. missing/unconfirmed Miss display plus `UNCHANGED`;
8. counter reset without false confidence;
9. different window models; and
10. a non-FIFO allocation countermodel plus reservation/Timing cancellation.

The required safety cases assert that truncation leaves
`search_complete=False`, `identifiable=False`, and an UNKNOWN hypothesis.

## Remaining non-identifiability

- Stock runtime build identity and the exact visible counter surface remain
  unverified.
- Repository auto-expiry is not Stock's audited angular post-floor Miss path.
- A counter reset or missing/delayed display can leave arbitrarily many hidden
  transitions.
- Exact-timestamp simultaneous inputs and unordered labels can preserve
  multiple histories.
- Alternative window/allocation models can fit the same visible prefix. Phase
  19 analyzes them independently rather than merging them into a false Stock
  conclusion.
- The bounded replay has no pre-3.15 estimator snapshot; earlier input state is
  explicitly outside scope.

## Reproduction

```powershell
uv run pytest -q tests/test_persistent_judgement_state_estimator_v02.py
uv run python scripts/audit_phase19_persistent_judgement_state_v02.py `
  --source reports/phase16-firsttown12-event89-causal-20261021-r2.json `
  --output reports/phase19-persistent-judgement-state-20261021-r1.json
```

The report writer refuses overwrite. No Validation20, FINAL, Train16 Motor A/B,
or training run is part of this procedure.
