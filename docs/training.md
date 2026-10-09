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
- `human_visible.py`: replay-chunk DAgger updates for the relaxed
  human-visible residual controller.
- `trajectory_trust.py`: paired closed-loop trajectory, key-boundary, and
  event-topology diagnostics/guards.
- `budget.py`: adaptive trust-radius controller.
- `runner.py`: configured orchestration and checkpoint metadata.

The current stable modes are:

- `human_visible_dagger` — keeps the inherited MaleCNS recurrence, sensory
  adapter, and baseline actor fixed, then adds a trainable residual controller
  that directly reads the same human-visible observation already present in the
  263D encoder. Visible floors stay ordered and go through a small 1D
  convolutional encoder; motor/orbit state and HUD feedback have direct paths;
  a GRUCell supplies controller memory; the fixed connectome state is supplied
  in parallel. The residual action head is initialized to exactly zero, so a
  fly-connectome parent is behaviourally unchanged before the first update.
  No chart time, exact target timestamps, absolute floor index, invisible future
  floors, teacher actions, or evaluator-only overload state are added. Training
  uses AdamW on cached realistic recurrent-state chunks and student-state DAgger.
  The existing SAFE→overload survival guard remains a rollback rule, but there
  is no action-RMS trust radius.
- `budget_action_trust` — actor-only DAgger with adaptive action-space trust.
- `budget_survival_trust` — actor-only DAgger with bounded action updates;
  reject only when an anchor that is SAFE in the current accepted policy becomes
  overloaded. Recovered anchors become protected on later accepted steps.
  Safe non-best continuations may cross short valleys, but after
  `budget.max_nonbest_accepts` consecutive non-best accepts the search restarts
  from the selected best and shrinks the action radius once.
  Best selection is survival-aware: number of SAFE anchors first, then hits,
  X-Accuracy, fewer TooEarly presses, and fewer keydowns. Therefore a newly
  recovered anchor cannot be discarded merely because its first recovery loses
  some hits elsewhere.
  Candidate evaluation also stops immediately when a currently SAFE anchor
  becomes overloaded, avoiding evaluation of the remaining anchors for a
  candidate that is already guaranteed to be rejected.
- `budget_boundary_trust` — deprecated compatibility alias for
  `budget_survival_trust`.
- `trajectory_probe` — generate one bounded actor candidate and compare it
  against the source policy in paired closed-loop rollouts, reporting the first
  physical/score/overload divergence per Train anchor.

## Running

Short form:

```powershell
.\scripts\run_school.ps1
```

Human-visible controller smoke:

```powershell
.\scripts\run_human_visible_smoke.ps1
```

Human-visible 8-hour school run:

```powershell
.\scripts\run_school_human_visible.ps1
```

Run the smoke before the 8-hour profile; the new controller changes the model
state shape and training path even though its initial actions exactly match the
parent.

Survival-trust school run:

```powershell
.\scripts\run_school_survival.ps1
```

Quick survival smoke run:

```powershell
.\scripts\run_survival_smoke.ps1
```

This uses 6 Train anchors, 1 Validation anchor, and at most 2 trials. The older
20-anchor / 5-trial profile is retained as:

```powershell
.\scripts\run_survival_smoke_full.ps1
```

While a survival run is active, the current continuation is also written to
`<output>.progress.pt`. This is a crash-recovery artifact; successful completion
removes it. The normal output checkpoint always remains the selected best.

The older `run_school_boundary.ps1` remains for compatibility.

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
