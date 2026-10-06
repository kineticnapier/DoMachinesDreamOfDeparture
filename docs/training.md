# Training architecture

Current training code has a stable entry point:

- `scripts/train_real_chart.py`
- `scripts/run_training.ps1`
- `src/dmdod/training/`
- `configs/training/*.toml`

## Rule for new experiments

Do **not** add another `train_real_chart_vXYZ.py` for parameter changes or small
algorithm variants.

Use one of these instead:

1. Change or copy a TOML file when only hyperparameters, paths, limits, or
   budgets differ.
2. Add a new module under `src/dmdod/training/` when the algorithm changes.
3. Register a new stable `mode` in `dmdod.training.runner` when a genuinely
   different training strategy is introduced.
4. Keep checkpoint compatibility/version information in checkpoint metadata,
   not in the script filename.

Historical `train_real_chart_v*.py` files are legacy experiment records. They
remain temporarily for reproducibility and old checkpoints, but current code
must not import them. This is enforced by
`tests/test_training_architecture.py`.

## Current modules

- `config.py`: typed TOML configuration and validation.
- `real_chart.py`: dataset windows, model reconstruction, evaluation,
  Train-safety guard, checkpoint primitives, DAgger collection.
- `action_trust.py`: actor-only action-space trust optimization.
- `budget.py`: adaptive trust-radius controller.
- `runner.py`: configured orchestration and checkpoint metadata.

The current stable modes are:

- `budget_action_trust` — actor-only DAgger with adaptive action-space trust.
- `trajectory_probe` — generate one bounded actor candidate and compare it
  against the source policy in paired closed-loop rollouts, reporting the first
  physical/score/overload divergence per Train anchor.

## Running

Short form:

```powershell
.\scripts\run_school.ps1
```

Generic form:

```powershell
.\scripts\run_training.ps1 -Config configs\training\school_8h.toml
```

Trajectory divergence probe:

```powershell
.\scripts\run_training.ps1 -Config configs\training\trajectory_probe.toml
```

Direct Python form:

```powershell
uv run --no-sync python -u scripts\train_real_chart.py --config configs\training\school_8h.toml
```

The PowerShell launcher runs the stable training tests first, prevents Windows
sleep while the process is active, merges native stderr safely for logging, and
writes logs under `logs/`.

## Creating an experiment

Copy an existing config:

```powershell
Copy-Item configs\training\school_8h.toml configs\training\trajectory_probe.toml
```

Then edit the new TOML. No new Python entry point is needed.

## Legacy cleanup

After the stable entry point has passed local regression tests and has been used
successfully for current training, legacy versioned trainers can be moved to an
archive in a separate cleanup change. Do not combine mass deletion/moves with
algorithm changes.
