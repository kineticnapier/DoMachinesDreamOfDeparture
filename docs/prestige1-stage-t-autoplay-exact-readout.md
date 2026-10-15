# Prestige I Stage T — Exact Autoplay Timing observability (experimental)

## Scope

Timing uses a **compiled-chart exact timestamp oracle**, and three
models are compared on the same causal 263D HUD:
- Constant: Train-fit near-event median remaining time / class prior
- Single-frame MLP: 263D observation without temporal context
- Forward-only GRU: historical observation frames up through "now"

This does **not** train Fingering, Motor, MaleCNS4096, or the former
Stage C Student. The original Stage A v2 FAIL is frozen.

## Input and labels

- Prediction input: **only** \`encode_n_key_hud_real_chart_observation()\`
  (263D). Future chart time, event indices, target oracle and the expert
  control signal are NOT passed to the Timing model.
- Privileged evaluator-side labels: exact \`AutoplayTimingTeacher\`
  target event times from compiled playable chart, cross-checked
  against the previously exported frozen oracle Train20 JSON.
- Neither timing labels nor predictions are clipped to 200ms.
  \`near_event <=200ms\` is only a classifier output and evaluation slice.
  Time loss has fixed near:far weights **1.0:0.25** (not tuned).
- Network output is nonnegative unbounded time via softplus, plus
  one near-event logit; no finger assignments or motor actions.
- Train-time float32 labels approximate the Python double precision
  oracle timestamps (no explicit 10ms or 1ms rounding).

**Limitations:** Captured visible trajectories are driven by the
privileged center-first expert. Expert anticipation may leak a
*behavioral cue* through observable motor states, even though the
model never receives a hidden chart schedule. This experiment
assesses **open-loop** prediction, not independent closed-loop play.
It predicts only the next event at each 10ms sampled instant;
two distinct events inside one 10ms control frame are not both
represented as separate predictions. High-density event-queue
planning remains future work.

## Protected evaluation

- Input charts: exact frozen **Train20** manifest.
- Split: deterministic **chart-SHA256 disjoint** Train-fit / Train-heldout,
  using the existing Prestige fixed seed.
- Models: fixed same train chunks and update budget, no heldout
  hyperparameter sweep or heldout checkpoint choice.
- Evaluation: both near-MAE and all-visible-MAE, near median/p95,
  near-event recall/FPR/balanced accuracy, anchor-level counts/errors
  and Train-fit vs Train-heldout breakdown.
- No Validation/FINAL data are captured or evaluated.
- The separate research criterion compares GRU near MAE to both
  constant and MLP (20% improvement), requires at least 3
  heldout near-positive anchors and nondegenerate classification.
- **Research evidence PASS does not override original Stage A**:
  \`motor_student_training_permitted=false\` remains in all cases.

## Code and usage

- \`src/dmdod/training/prestige1_autoplay_readout.py\`
- \`scripts/train_prestige1_autoplay_timing.py\`
- \`tests/test_prestige1_autoplay_readout.py\`

Frozen inputs: \`stage_a_v2.json\`,
\`stage_timing_autoplay_exact_train20.json\`, the frozen v3.0 report
and saved plans (SHAs verified).

Write-once research outputs in \`diagnostics/prestige1/\`:

- \`stage_timing_autoplay_readout_train20.json\`
- \`stage_timing_autoplay_mlp.pt\`
- \`stage_timing_autoplay_gru.pt\`

Run from the Windows repository root:

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_readout.py
uv run python -u scripts/train_prestige1_autoplay_timing.py --config configs/training/prestige1.toml
~~~

Use the training command **only if tests PASS**. It uses the
Prestige TOML's device (CUDA by default) but is not a Motor simulation
or Motor Student training run. Output files cannot be overwritten.
An authentic native ADOFAI Autoplay parity study remains separate.
