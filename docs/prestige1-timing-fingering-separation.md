# Prestige I — Separating Timing, Fingering, and Motor (interface v0)

## Decision and motivation

Keep **three separately trainable/evaluable problems**, without requiring
three disjoint neural networks:

- **Timing**: infer *when* the next visible rhythm event is due, how many
  simultaneous physical presses it needs, and uncertainty.
- **Fingering**: use Timing's predicted events together with causal, visible
  motor state and history to select *which* physical fingers to use.
  IRL or imitation may eventually learn its choices/costs.
- **Motor**: turn the joined *intent* into 8 continuous force commands, taking
  into account physical finger movement and release.

A shared temporal encoder may later serve two readout heads; that is an
architectural possibility, not a completed model. The interface boundary
is a way to identify whether a Miss is caused by the event forecast,
fingering feasibility, or motor actuation rather than merging all errors
under a single imitation loss.

## What was implemented

\`src/dmdod/planning/causal_contract.py\` provides **interfaces only**:

\`\`\`text
  observable 263D HUD samples at 10ms steps (history, not future)
                             |
                      CausalHUDHistory
                             |
                  +----------+----------+
                  |                     |
           TimingEstimator       (history available)
                  |                     |
           TimingPrediction             |
        relative time, chord,           |
             confidence                 |
                  +----------> FingeringPlanner
                                      |
                              FingeringProposal
                              event slot -> keys
                  +----------+----------+
                             |
                      join_predictions
                             |
                         JointIntent
                             |
                       MotorController
                             |
                     NKeyAction [-1,1]^8
                             |
                    1ms physical simulator
\`\`\`

The diagram is conceptual. The example \`join_predictions\` function only
enforces *structural consistency* (event slot exists, chord size matches,
no duplicate finger in a chord). It **does not** claim to solve physical
readiness, fatigue, hand coordination, or overload. Those require a
separate planner implementation and the existing motor simulator.

### Causality

The only observation adapter is
\`VisibleHUDFrame.from_observation(step, NKeyHudRealChartObservation)\`,
which calls \`encode_n_key_hud_real_chart_observation\`. Its vector
contains exactly 263 current visible values: 8 key positions, 8
velocities, 8 pressed flags, the orbit position, floor visibility
features, and HUD feedback. These fields and their ordering are
unchanged from the existing simulator. A monotonically increasing
sample index is a relative counter, **not** privileged chart time.

Do **not** pass evaluator-only \`TargetContract\`, \`segment.targets\`,
\`env.privileged_episode_time_s()\`, per-note timestamp, future floor,
oracle teacher action, or hidden overload diagnostics into Student
Timing/Fingering/Motor. \`TimingPrediction\` only carries model
predictions, never ground-truth events.

The Python dataclasses validate shape and semantics. As usual, a
numeric interface cannot prove that a malicious upstream caller did
not encode secret information into its numbers; data provenance
and evaluator/Student separation must also be verified in the data
pipeline. No such pipeline is wired in this step.

### Observable failure and uncertainty

- \`TimingPrediction(events=())\` means no predicted near event.
- \`FingeringProposal(allocations=())\` is a valid abstention, not
  a request to fabricate a center-first "ground-truth" fingering.
- \`join_predictions\` returns explicit \`abstained_event_indices\`
  for unassigned predicted events.
- The initial Timing interface restricts relative event time to
  **0..200ms** and chord size to **1..8**, matching Stage A v2's
  near-event task. This interface does **not** handle overdue
  signed-time observations. Any late-event extension requires a
  separately designed, versioned contract instead of silently
  reinterpreting the existing output.

## How IRL could fit (not implemented)

The current evaluator already defines physical Hit/Miss, timing error,
overload, and extra KeyDown. An IRL experiment should *not* need to
relearn these as if they were unknown.

A future Fingering research branch can learn a **preference/cost**
over available choices conditioned on visible history and estimated
events (e.g., future readiness, fatigue, hand movement), using
expert demonstrations with trustworthy per-finger key assignments.
Classical maximum-entropy IRL, inverse optimal control, or
preference-based imitation are possible baselines, not selected
implementations.

Expert finger identity is **not** uniquely correct for ADOFAI;
equally safe alternate fingerings can achieve the same target Hit.
Current \`n_key_actuation_loss\` deliberately treats pressed keys as
permutation-invariant where appropriate. Do not undo this just to
make IRL or ordinary behavior cloning easier.

A reward inferred from demonstrations may not be identifiable:
different reward weights can explain the same behavior. Validate
choice stability and physical feasibility before claiming human
technique was recovered. No human fingering corpus or IRL training
is created by this contract-only change.

## Independent future evaluation criteria (NOT executed)

| Component | Appropriate measurements |
|---|---|
| Timing | Near-event MAE, event detection recall/FPR, chord-count accuracy, calibration, observability |
| Fingering | Physical feasibility, missed/extra presses after motor execution, per-hand and per-finger load, choice consistency |
| Motor | True 1ms KeyDown timing, Hit/Miss, TooEarly/Overload, action effort |
| Integrated | Full-chart SAFE, Hit accuracy, throughput, unseen-chart generalization |

Timing and Fingering can be trained with distinct losses and datasets
even if they share an encoder; the final decision must be jointly
validated through Motor. A good fingering cannot compensate for an
unobservable note; accurate timing does not imply feasible fingering.

## Preservation / current status

This is *not* Prestige I v3.1 or a completed Student architecture.
No current policy, teacher, Stage A training or gate, 263D observation
encoder, physical model, Train20 experiment, checkpoint, Validation,
or FINAL evaluator was modified.

Stage A v2 remains **FAIL**
(\`gru_near_event_mae_not_20pct_better_than_constant_and_single\`).
\`may_run_stage_c=false\` stays unchanged; Student training remains
blocked. The offline Stage C teacher-only experiments remain
exploratory and separate.

CPU contract tests:

\`\`\`powershell
uv run pytest -q tests/test_prestige1_causal_planning_contract.py
\`\`\`

This test checks shape, history ordering, nonprivileged observation
encoding, chord consistency, abstention, and motor command range;
it does **not** test a trained Timing head, an IRL policy, or gameplay.
