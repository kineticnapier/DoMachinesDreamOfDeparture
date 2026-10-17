# DMDOD Prestige I — Autoplay-supervised neural Timing Student final report

## Verdict

**FAIL: practical unknown-chart lookahead was not acquired.**

The exact-Autoplay, causal 263D, multi-event training/evaluation path is
implemented and reproducible, but the neural Student did not improve Timing
System v1.4. Qualification20 and Reserve20 remain unopened. Timing v1.4
remains the supported predictor; neither Stage A nor Motor training is
unlocked.

## What was implemented

- Exact, unrounded `AutoplayTimingTeacher` events supervise up to 16 ordered
  future onsets in the next 80ms. Equal-time inputs remain one chord.
- A forward-only GRU receives the current/past 263D visible HUD v2 and emits
  monotonically ordered event times, per-slot existence confidence, count,
  and explicit abstention. It has no Fingering or Motor head.
- Multiple events within one 10ms controller interval remain separate labels.
- A causal event reconciler assigns only local prediction IDs, merges repeated
  frame forecasts, cancels contradicted forecasts, and credits each truth at
  most once. No target ID is available to the reconciler.
- Matched comparison covers Timing v1.4, the saved exact-Autoplay MLP/GRU,
  and the new Student at 1/5/10/20ms tolerances. Lead, count error,
  confidence calibration, abstention, high density, and special cohorts are
  reported separately.

The saved MLP/GRU expect the legacy 263D BPM encoding. Their comparator input
is generated at exactly the same observation instants, while v1.4 and the new
Student receive the current unclipped-BPM HUD v2. An early neural checkpoint
(`936c670`) copied the legacy readout encoding; it was rejected before any
Development evaluation, retained as provenance, and replaced by HUD-v2
checkpoints.

## Causality and leakage audit

Student inference calls only `model(visible[T,263])`. It never receives the
compiled chart, episode/chart clock, target time, floor/target index, future
HUD, teacher action, labels, special-event cohort tags, or evaluator match.
Full-sequence GRU evaluation is prefix-causal and is covered by a prefix
invariance test. Exact times and special tags enter only label construction
and posthoc scoring.

The real-chart capture trajectory is driven by the same privileged
center-first Motor teacher used by the saved MLP/GRU. Thus public motor-state
channels can contain a **behavioral cue**, even though actions are not model
inputs. This limitation is explicit. The separate high-density stress uses
zero-key trajectories and still fails, so the reported failure is not hidden
by that cue.

## Frozen data protocol

1. Train20 is SHA256-identical to the saved exact-Autoplay experiment.
2. The new model trains on its fixed fit15 and selects confidence only on its
   already-defined heldout5.
3. The already-opened, checkpoint-unseen Development20 is the model-selection
   comparison. Charts are SHA-sorted after excluding all 20 checkpoint hashes.
4. The previously opened Final10 is not reused as holdout.
5. Qualification20 requires a machine-readable Development PASS. Both neural
   revisions failed, so Qualification20 and Reserve20 were not evaluated.

Primary success was frozen before Development: at 10ms tolerance and 10–80ms
first-prediction lead, event F1 must exceed v1.4 by at least 0.02, false
predictions per target must not increase, MAE must be at most 10ms, and median
matched lead must be at least 20ms.

## Unknown-chart Development20 result

There are 1,340 exact Autoplay events over 13,705 HUD frames. The table is the
primary practical endpoint; predictions with less than 10ms lead receive no
credit.

| Predictor | Match / target | False | Tracking | Precision | F1 | MAE | Median lead |
|---|---:|---:|---:|---:|---:|---:|---:|
| Timing v1.4 | 1075 / 1340 | 0 | 80.224% | 100.000% | 0.89027 | 0.000005ms | 74.44ms |
| Saved MLP | 7 / 1340 | 10 | 0.522% | 41.176% | 0.01032 | 2.47ms | 79.20ms |
| Saved GRU | 8 / 1340 | 146 | 0.597% | 5.195% | 0.01071 | 3.85ms | 76.41ms |
| New v2 Student | 2 / 1340 | 17 | 0.149% | 10.526% | 0.00294 | 3.89ms | 38.33ms |
| New v3 stable-time Student | 3 / 1340 | 221 | 0.224% | 1.339% | 0.00384 | 5.87ms | 38.13ms |

The v3 Student at all leads emitted 224 unique forecasts. Matches by tolerance
were 0/3/3/7 at 1/5/10/20ms; false forecasts were 224/222/221/217. Its
frame-level 80ms count MAE was 0.385 and abstention rate 29.13%, but these
better frame statistics did not produce stable correct event times.

Timing v1.4 cohort tracking at the primary endpoint was 24/28 high-density,
371/458 high-BPM, 23/29 SetSpeed, 192/262 Twirl, and 6/8 full-turn. The v3
Student was 0/28, 0/458, 0/29, 1/262, and 0/8 respectively. The Development
windows contained no separately tagged Pause target; Pause is covered by the
controlled stress below.

## Confidence reliability

The selected v3 threshold is 0.5, chosen only on Train-heldout5. Development
primary predictions had ECE **0.514**. Of 216 forecasts in the 0.4–0.6 bin,
none was correct; the mean confidence was 0.520. The five 0.6–0.8 predictions
had 40% accuracy, and three 0.8–1.0 predictions had 33.3% accuracy. Therefore
the confidence is not calibrated and must not be interpreted as a probability
of correct timing.

## Density and special-event stress

The fixed stress uses genuine 263D HUD traces from the 1ms physical simulator,
sampled on a 10ms controller clock at phase offsets 0/1/2/5ms. It is a
deterministic stress test, not holdout generalization.

| Cohort | 10ms targets | Multi-event frames | v1.4 matches / false | v1.4 exact multi | v3 matches / false | v3 exact multi |
|---|---:|---:|---:|---:|---:|---:|
| 8k/12k/20k BPM, CW+CCW | 1244 | 436 | 837 / 0 | 277 | 0 / 0 | 0 |
| Pause | 360 | 0 | 356 / 0 | 0 | 0 / 0 | 0 |
| SetSpeed | 360 | 80 | 319 / 0 | 79 | 0 / 0 | 0 |
| Twirl | 360 | 0 | 360 / 0 | 0 | 0 / 0 | 0 |
| Full-turn pattern | 360 | 44 | 338 / 0 | 44 | 0 / 0 | 0 |

The v3 Student did not abstain in this stress. At 20,000 BPM CW its first
predicted offset stayed between **23.69 and 30.85ms** (median 24.72ms), even
when one to four truths were due inside 10ms. This directly explains the zero
short-horizon tracking: event existence/count and due-time countdown are not
coupled by the learned representation.

## Failed hypotheses and causes

1. **v2 softmax partition:** the model emitted an event on almost every frame,
   but the due time slid forward instead of counting down. Only 19 practical
   forecasts survived reconciliation.
2. **v3 cumulative monotone time head:** heldout slot F1 improved from 0.521
   to 0.778 and count MAE from 0.654 to 0.178, but Development produced many
   stable wrong due times (221 false). This falsifies the hypothesis that the
   partition parameterization alone caused the failure.
3. **Generalization:** 15 fit charts and a bounded 240-update GRU learned event
   presence more readily than geometric phase/time-to-contact. The exact
   deterministic v1.4 geometry is vastly more sample-efficient and remains
   essentially exact where eligible.
4. **Multi-event capacity is only structural:** 16 output slots represent
   dense sequences, but the network did not learn to place them at the correct
   offsets. Structural capacity is not evidence of acquired timing skill.
5. **Confidence is decoupled from temporal correctness:** high existence
   confidence persisted for forecasts with wrong due time.

No additional threshold search, architecture sweep, Qualification run, or
Reserve run was performed after these failures.

## Protection status and scope

- Original Stage A remains FAIL with
  `gru_near_event_mae_not_20pct_better_than_constant_and_single`,
  `validation_used_for_selection=false`, and `may_run_stage_c=false`.
- `stage_a_v2.json` SHA256:
  `c4f71aadc5dc6cca49679a9902bcb7583a0f89b40f92b7fb283a50c3faad0558`.
- Saved exact-Autoplay MLP/GRU SHA256:
  `9c1acfaa826d8489e79f126f5610c7b5b0eed6b87448204382e2bfa54cd63468`
  and `9804e119b207743fd06e47195fc064d8af9847b56d2c862a6e251e115eb5948e`.
- Final v3 checkpoint SHA256:
  `bca41284d85e92c6a9be2c0a2b134201609310d8e2a534f1c7f518900c30ca1a`.
- No Fingering or Motor model was trained. No physical key command was issued;
  this was intentionally Timing-only. The 1ms Motor environment was used only
  as a HUD/physics trace source for stress evaluation.
- The user's pre-existing `configs/training/prestige1.toml` modification was
  not staged, overwritten, or reverted.

## Reproduction

From the repository root on GPU:

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_prestige1_autoplay_sequence_student.py `
  tests/test_autoplay_sequence_evaluation_v1.py `
  tests/test_prestige1_autoplay_timing.py `
  tests/test_prestige1_autoplay_readout.py `
  tests/test_prestige1_autoplay_future_events_audit.py `
  tests/test_hud_timing_tracker_transition_v14.py `
  tests/test_jrp_unseen_split_v2.py `
  --basetemp .pytest-tmp-autoplay-final

.\.venv\Scripts\python.exe -u scripts/train_prestige1_autoplay_sequence_student.py `
  --config configs/training/prestige1.toml --device cuda `
  --checkpoint checkpoints/prestige1/autoplay_sequence_student_v3-REPRO.pt `
  --output reports/prestige1-autoplay-sequence-train-v3-REPRO.json `
  --status-file logs/prestige1-autoplay-sequence-train-v3-REPRO.status.json

.\.venv\Scripts\python.exe -u scripts/audit_prestige1_autoplay_sequence_student.py `
  --config configs/training/prestige1.toml --device cuda `
  --checkpoint checkpoints/prestige1/autoplay_sequence_student_v3-REPRO.pt `
  --partition development `
  --output reports/prestige1-autoplay-sequence-v3-development-REPRO.json `
  --status-file logs/prestige1-autoplay-sequence-v3-development-REPRO.status.json

.\.venv\Scripts\python.exe -u scripts/audit_prestige1_autoplay_sequence_stress.py `
  --config configs/training/prestige1.toml --device cuda `
  --checkpoint checkpoints/prestige1/autoplay_sequence_student_v3-REPRO.pt `
  --output reports/prestige1-autoplay-sequence-stress-REPRO.json `
  --status-file logs/prestige1-autoplay-sequence-stress-REPRO.status.json
```

Use `--device cpu` for CPU reproduction. Output paths are write-once; choose
new names. A failed Development report cannot authorize Qualification.

## Saved artifacts

- Final checkpoint:
  `checkpoints/prestige1/autoplay_sequence_student_v3-e07d934.pt`
- Training report/status/log:
  `reports/prestige1-autoplay-sequence-train-v3-e07d934.json` and
  `logs/prestige1-autoplay-sequence-train-v3-e07d934.*`
- Primary Development report/status/log:
  `reports/prestige1-autoplay-sequence-v3-development-794f5da.json` and
  `logs/prestige1-autoplay-sequence-v3-development-794f5da.*`
- Stress report/status/log:
  `reports/prestige1-autoplay-sequence-stress-f07c9dd.json` and
  `logs/prestige1-autoplay-sequence-stress-f07c9dd.*`
- Superseded v2 and rejected legacy runs remain committed for auditability.

Experiment source commits are `e07d934` (final model), `794f5da` (checkpoint),
`40978a8` (Development FAIL), `f07c9dd` (stress code), and `a5dbbb4` (stress
results). The final report commit SHA is reported by `git rev-parse HEAD` and
in the completion message.
