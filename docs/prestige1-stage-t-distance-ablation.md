# Prestige I Stage T — Distance diagnosis + one near-only loss ablation

## Purpose and evidence

The saved Stage T Train-heldout results were constant near MAE 41.924ms,
MLP 59.649ms and GRU 47.328ms. GRU all-visible MAE was 175.667ms,
but the GRU failed to beat the constant's near MAE. This does not yet
distinguish loss conflict, optimization undertraining and HUD
partial observability. The experiment below tests only loss conflict.

## 1. Read-only four-bin distance diagnosis (run this first)

Load BOTH frozen Stage T checkpoints. Re-capture Train20 expert-driven
HUD frames. Verify source SHA256, config, exact-Autoplay oracle, split,
model weights and restored evaluation metrics before proceeding.
Separate true timing distance into four disjoint bins:
- 0-50ms: 0 <= t < 50ms
- 50-100ms: 50 <= t < 100ms
- 100-200ms: 100 <= t <= 200ms (exact oracle near flag)
- over-200ms: oracle says event is not near

For constant, MLP and GRU, report count, absolute error MAE, median,
P95, signed prediction bias, and per-anchor values. Train-fit
(15 chart windows) and **chart-disjoint** Train-heldout (5 windows)
are reported separately. Nothing is trained; no old file is changed.

Files:
- src/dmdod/training/prestige1_autoplay_diagnostics.py
- scripts/audit_prestige1_autoplay_timing_distances.py

Write-once output:
diagnostics/prestige1/stage_timing_autoplay_distance_bins_train20.json

## 2. One fixed change: near-only regression GRU (run second)

Controlled hypothesis: far-distance regression consumes capacity and
gradient budget needed for accurate near-event timestamps.

Keep identical model, fresh seeded weights, training data, chart
partition, 118-chunk one-pass coverage, batch frames, optimizer,
LR, seed, and near-event BCE classifier. Change one thing:

**Remove all >200ms timestamp regression from the GRU loss**.
The time Smooth L1 term is supervised only at <=200ms frames.
Far frames remain labeled by Autoplay and remain supervised by
the BCE near/not-near classifier. Raw full-distance oracle truth
is retained and all-visible MAE is still reported.

This ablation can worsen all-visible MAE while helping near MAE;
it is diagnostic, not automatic model adoption. The new GRU is
saved separately from the frozen MLP/GRU. The report includes the
same near/far bins and compares with the original checkpoint
on identically re-captured heldout HUD frames.

Files:
- src/dmdod/training/prestige1_autoplay_near_ablation.py
- scripts/train_prestige1_autoplay_near_only.py
- tests/test_prestige1_autoplay_distance_and_ablation.py

Write-once outputs:
- diagnostics/prestige1/stage_timing_autoplay_near_only_gru_train20.json
- diagnostics/prestige1/stage_timing_autoplay_near_only_gru.pt

## Windows commands

Run from the current develop/v0.1-simulator branch. Check status
before fast-forward pull and preserve all uncommitted files.

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_readout.py tests/test_prestige1_autoplay_distance_and_ablation.py
~~~

After tests PASS, run read-only distance diagnosis first:

~~~powershell
uv run python -u scripts/audit_prestige1_autoplay_timing_distances.py --config configs/training/prestige1.toml
~~~

Only after the diagnosis is successful, run one fixed GRU ablation:

~~~powershell
uv run python -u scripts/train_prestige1_autoplay_near_only.py --config configs/training/prestige1.toml
~~~

**No old results are replaced or resumed.** Do not delete, reset,
stash or repurpose saved experimental outputs.

## Interpretation and limitations

- Improved near MAE supports a loss-competition explanation under the
  fixed budget. It does not establish independent Student observability.
- Failed near MAE means further investigation of data coverage,
  optimizability and HUD partial observability is still necessary.
- The HUD is from a privileged center-first motor trajectory, so
  anticipatory movement may make some true timestamps easier to infer.
- Only the next event is represented at a 10ms control step, not a
  potentially denser queue of multiple events.
- Regardless of outcome, the frozen original Stage A remains FAIL.
  No Motor/Fingering Student training, Validation/FINAL evaluation,
  hyperparameter sweep, checkpoint selection or automatic adoption.
