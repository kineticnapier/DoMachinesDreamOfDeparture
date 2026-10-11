# Human-visible beta training ablation

## Purpose

Compare four **two-update** DAgger training candidates against the same
selected-best model and saved AdamW optimizer moments:

| Arm | Teacher intervention probability | Failure continuation |
| --- | ---: | --- |
| `beta1_full` | 1.0 | Natural horizon |
| `beta05_full` | 0.5 | Natural horizon |
| `beta0_full` | 0.0 | Natural horizon |
| `beta0_post1s` | 0.0 | At most 1.0 s after first terminal failure |

All Train variants use **the same 20 Train anchors**, fixed expert trajectories,
seed, balanced replay plan construction and exactly **two AdamW updates**.
The same **20 fixed Validation anchors** are evaluated once for each candidate;
Validation is never used to choose a training action or update.

This is a **separate experiment** from the curriculum trainer and its alpha
sweep. The selected-best checkpoint and FINAL holdout are not altered.
Overload-free anchors are **not necessarily full-cleared charts**.

## Execution (PowerShell)

```powershell
git pull --ff-only
uv run pytest -q tests/test_beta_training_ablation.py tests/test_beta_collection_probe.py tests/test_n_key_dagger_continuation.py
uv run python -u scripts/train_human_visible_beta_ablation.py --config configs/training/human_visible_beta_training_ablation.toml --report reports/human_visible_beta_training_ablation.json --save-candidates
```

With `--save-candidates`, the four checkpoints are named after
`reports/human_visible_beta_training_ablation.*.pt`. They have
`training_mode=human_visible_beta_ablation_candidate` and are deliberately
**not selected-best curriculum checkpoints**. Without the flag, only a new
JSON report is written. Existing report/checkpoint paths are rejected before
training begins. To repeat, use a *different* `--report` path.

## Guardrails

- The script verifies the selected-best role, source optimizer moments, saved
  Validation identity and saved baseline Validation summary.
- The `beta1_full` result must reproduce the previously observed epoch-3
  candidate: `[SAFE=9, Hits=1546, TooEarly=250, Keydowns=1796]`.
  If not, it fails instead of presenting incomparable results.
- Arms restore model parameters and saved AdamW state, reset random seeds, and
  reuse the same plan seed. Only collection policy beta and post-failure cap
  vary; resultant replay windows may differ because trajectories differ.
- The 1-second cap changes only frames **after** first terminal failure;
  first-terminal scoring must be identical to the uncapped beta=0 rollout.
- The report includes all arms, collection summaries, update metrics and fixed
  Validation results. **No automatic winner** is declared.
- Because all four arms share the same fixed Validation, the results are
  diagnostic. Any final tuning claim should be checked on independent charts.
