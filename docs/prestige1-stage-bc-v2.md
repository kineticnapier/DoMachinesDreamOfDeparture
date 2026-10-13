# Prestige I — Stage B/C v2

This is an **exploratory** Motor-learning comparison, separate from the
original failed Stage A 20%-MAE gate. Do not alter the old Stage A report.

## Preflight

Preserve local `configs/training/prestige1.toml` (the `core_path` must point to
the existing `male-cns-v1.0-core4096-w5.pt`).

```powershell
git pull --ff-only
uv run pytest -q tests/test_prestige1_transfer.py
uv run python -u scripts/run_prestige1_transfer.py stage-b `
  --config configs/training/prestige1.toml
```

Stage B checks SHA256 provenance and GRU vs GRUCell numerical parity; no Motor
updates or Validation evaluation are performed.

When Stage B passes, the optional GPU pilot is:

```powershell
uv run python -u scripts/run_prestige1_transfer.py stage-c-v2 `
  --config configs/training/prestige1.toml
```

Stage C v2 trains two **new** continuous 8D Motor Students. Each uses
new randomly initialized MaleCNS4096 controller weights and the same
**Stage A v2 pretrained temporal weights**. Control uses Motor BC only;
Prestige uses the identical Motor BC plus Timing auxiliary loss.
The fixed minibatch schedule is identical across arms. The number of updates
per arm is `max(stage_c_updates, full_Train20_chunk_count)`; the default
is at least one full Train20 pass, **not** 24 updates.

Both arms finish Train learning before Validation20 physical evaluation.
No FINAL, no DAgger, no old Best weight or optimizer warm-start.
Do not use Validation20 metrics to tune the model or select a checkpoint;
this is not an independent holdout after the already inspected Stage A runs.

Artifacts (all separate from the existing `stage_a*.json`, `gru*.pt`,
`stage_c.json`, and old Best):

- `diagnostics/prestige1/stage_b_v2.json`
- `diagnostics/prestige1/stage_c_v2.json`
- `checkpoints/prestige1/control_v2.pt`
- `checkpoints/prestige1/prestige_v2.pt`

Stage C v2 deliberately refuses to overwrite output files and currently
**has no partial-training resume**. The full CUDA run should only be started
when the user can allow both arms to complete.
