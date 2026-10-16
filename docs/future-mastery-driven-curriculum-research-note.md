# DMDOD future research note: mastery-driven practice and transfer

Status: **idea / deferred** (2026-10-16). This document is a research memo,
NOT an approved implementation or a change to the frozen Prestige I pipeline.

## Core idea

Practice **one chart intensively until it is cleared**. Diagnose failed
sections, experiment with fingering/timing/motor strategies in the simulator,
and consolidate successful behavior into the same Student. Then test whether
the acquired patterns are reusable on other charts. Track whether later charts
take fewer practice attempts to master.

This reflects the human distinction between deliberate fingering research
during practice and automated execution during a performance. A separate
"thinking" neural network is *not* required as an initial assumption, and
whether any deliberate/automatic division emerges inside the same recurrent
circuit is a separate, much harder research question.

## Make time-indexed memorization less effective

A deterministic chart and repeatable initial state permit a recurrent policy
to use an internal clock and replay fixed actions without meaningfully
interpreting the HUD. Do not assume that clearing the training chart proves
transferable perception or motor skill.

**First intervention: vary each training chart while retaining the same
recognizable pattern family**, with changes that are visible/derivable from
the available 263D human-visible HUD:

- Randomize episode entry/start location and pre-entry waiting duration.
- Slightly vary the BPM and update the genuine chart/HUD timing accordingly.
- Move familiar motifs to other chart positions, or rearrange combinations
  of motifs while preserving legal, playable chart structure.
- Retain a stable unperturbed reference rendition to establish actual mastery.

The goal is **not** to prevent memorizing movement patterns: reusable motor
skills are exactly what the Student should retain. It is to make the policy
depend on visible timing/geometry instead of absolute elapsed-step indices.

**Possible later intervention**: gradually decay or perturb *appropriate*
RNN hidden-state components so that an arbitrarily persistent episode clock
becomes less useful. This is only a hypothesis. Naive global decay is
undesirable: precise current phase and physical motor state must remain
available, especially near extreme BPM. Recurrent dynamics can reconstruct
a clock even in the presence of decay, and model weights themselves can
memorize a chart. Therefore test chart variation alone before considering
hidden-state decay; do not change the fixed MaleCNS4096 as part of this note.

## Evaluation contract

Predefine and freeze evaluation charts **before** training. Measure:

1. Clear rate / X-Accuracy / overload and timing on the practiced chart.
2. Zero-shot performance on unseen charts containing the same movement
   motifs in different locations, with changed BPM, and in new combinations.
3. Zero-shot performance on structurally different, unseen charts.
4. Number of attempts or simulator interactions needed to clear the *next*
   chart, compared against learning from scratch and relevant baselines.
5. Retention of previous charts and motifs (catastrophic forgetting).
6. Sensitivity to removing/shuffling current HUD evidence, to detect a
   policy that merely replays memorized actions.

Do not tune against the frozen evaluation charts. Report both benefits and
regressions and avoid interpreting pure training-chart mastery as proof of
generalization.

## Order of operations / constraints

- **Do not implement or initiate experiments yet.** First stabilize the
  current HUD timing, future-event prediction, and motor execution contracts.
- Keep this proposal isolated from the current Prestige I / Stage A / Stage T
  experiments; Stage A FAIL and existing checkpoints remain frozen.
- No changes to original Geometry/Hybrid or MaleCNS4096 from this memo.
- At 20,000 BPM, the current 10ms action-hold interface can traverse
  multiple notes per action. Chart practice/randomization alone does not
  guarantee intra-frame motor control; solve/measure this separately.
- Training chart adaptation and true unseen-chart transfer must be measured
  as distinct phenomena.

Working label: **Mastery-Driven Curriculum** (provisional).
