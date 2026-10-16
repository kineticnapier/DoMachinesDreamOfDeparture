# Stage T — Own-action physical shadow for 10ms Motor predictive arming

**Status:** isolated CPU deterministic diagnostic, **not** a new trained Student or an adopted Motor controller. Existing Stage A FAIL, original Geometry/Hybrid, 8-Key MaleCNS4096 and all checkpoints are untouched; no training, Validation, FINAL, or model tuning.

## Why

At 4,000 BPM the original 8-Key HUD visible-prefix controller often produces a KeyDown for an intended target *after* that target's timestamp. Its fixed cold-key onset assumption ignores the hand/body dynamics and the motion already in progress. Instead of adjusting physical constants or using evaluator timestamps, this experiment asks whether a model-based forecast of the actual physical KeyDown can improve **command timing**.

The user notes that 4,000 BPM gameplay is normally handled with **16-key fingering** and can be tiring with 8 keys. This comparison intentionally **does not** change the fixed Prestige I 8-Key/263D observation contract. Any 16-Key capacity experiment must be reported separately; these 8-Key results are not evidence of a human 16-Key limit.

## Experimental architecture

- Keep the **same** single-frame 263D HUD visible-prefix forecaster, 70ms visible-ahead horizon, deterministic finger assignment, continuous per-key reservations, **10ms held policy commands and 1ms physical substeps**.
- New `ShadowDynamicsPlanner` maintains an **independent physical NKeyMotorEnv** model. Initialize it from rest and run **the exact actions that this planner previously issued**. Never read/clone the actual environment's internal Motor state.
- At each control frame, compare the independently reconstructed **public** motor position, velocity and pressed flag with the current 263D HUD. Abort if they disagree.
- To decide when to arm an unpressed reserved finger, clone **the own-action shadow**, issue a hypothetical `+1` for that finger, hold other already-chosen commands, and physically integrate at 1ms for at most **90ms**. Predict when its first KeyDown would arrive.
- Begin arming when the predicted time-to-KeyDown reaches the reserved time-to-target **minus a fixed 5ms half-step command quantization margin**. Once a command starts, sustain it until a visible pressed flag is seen; release with `-1` using the baseline logic.
- This is a *per-finger candidate rollout with coupled physics*, **not** a joint optimization of all eight finger trajectories. The independent forecast can be inaccurate when other keys begin moving simultaneously.
- All event timestamps from the evaluator and actual KeyDown trace are obtained **after selecting and executing actions** and used only for scores, not the decision process. A posthoc KeyDown-intent pairing is chronological on the same assigned finger and may be ambiguous if key reuse is imperfect.

## Fixed A/B results: original visible-prefix versus shadow physical onset

All nine fixed straight-floor fixtures; no hyperparameter tuning or adaptive chart selection:

| BPM | Notes | Baseline Hit | Shadow Hit | Baseline X | Shadow X |
|---:|---:|---:|---:|---:|---:|
| 1,200 | 18 | 18 | 18 | 95.28% | **100.00%** |
| 2,000 | 18 | 18 | 18 | 85.00% | **100.00%** |
| 4,000 | 18 | 17 | 17 | 63.06% | 63.06% |
| 1,200 | 72 | 72 | 72 | 98.47% | **100.00%** |
| 2,000 | 72 | 72 | 72 | 88.89% | **100.00%** |
| 4,000 | 72 | 65 | **63 (regression)** | 42.43% | **41.32% (regression)** |
| 1,200 | 144 | 144 | 144 | 99.24% | **100.00%** |
| 2,000 | 144 | 144 | 144 | 88.54% | **100.00%** |
| 4,000 | 144 | 127 | 127 | 41.25% | **38.44% (regression)** |

On the 2,000 BPM / 144-note fixture, mean intended-to-physical-KeyDown error for matched shadow arming commands is ~1.43ms. The shadow onset predictor's estimated-versus-actual KeyDown MAE is ~3.73ms.

For 4,000 BPM / 144 notes, the **arm-onset estimate error is only ~2.35ms**, but the mean deviation from *intended note time* is ~38.35ms. This is crucial: **predicting how a chosen finger will move is not enough to resolve which finger should be activated, how early, and how successive notes must be mapped to the evaluator queue**. Nor does it guarantee that a motor with 8 keys can maintain those rates given the fixed coupling and recovery parameters.

**Do not adopt this controller universally:** the 4,000 BPM regressions are real on these fixtures. This is a read-only mechanism test and a diagnostic baseline, not a claim of end-to-end 4,000/20,000 BPM performance.

## Commands

~~~powershell
uv run pytest -q tests/test_prestige1_autoplay_state_aware_motor_audit.py
uv run python -u scripts/audit_prestige1_autoplay_state_aware_motor.py --config configs/training/prestige1.toml
~~~

Unique report, overwrite refused:

    diagnostics/prestige1/stage_timing_shadow_dynamics_motor_arming_readonly.json

## Suggested next research step (not implemented)

Before changing Student or increasing key count inside its frozen HUD, compare joint multi-finger trajectory scheduling and explicit physical press/release recovery within a **separate** model-based controller, with online recognition of available keys. Separately create a fair 8K vs 16K *Motor-only capacity benchmark* so different finger counts and human practices are not conflated with Stage T perception.
