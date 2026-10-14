# DMDOD Prestige I — Stage C v2.1 Read-only Failure Diagnostics

This is a new, **read-only** diagnostic for completed Stage C v2
checkpoints, not additional training, not model selection, and not FINAL.
The stored Stage A MAE gate remains FAILED, unchanged.

## Run

Use the existing local config (including your locally fixed MaleCNS core path).

```powershell
git pull --ff-only
uv run pytest -q tests/test_prestige1_failure_diagnostics.py
uv run python -u scripts/run_prestige1_failure_diagnostics.py `
  --config configs/training/prestige1.toml
```

Default Validation anchor IDs: **7 Tomori, 11 Researching, 18 Curtain Call**.
A custom subset can be specified with `--anchors 7 11 18`.

Read-only replay checks all model/core, Stage A/B/C report, and checkpoint
hash provenance; mismatched physical results **fail closed** instead of
silently making a new result. The old Best is never loaded as a Student.

New output (no overwrite): `diagnostics/prestige1/stage_c_v21_diagnostics.json`

### Data fields per anchor and arm

- Stored-vs-replayed Hit, Miss, TooEarly, Overload, KeyDown and chart identity
- Physical termination reason, time and overload gauge
- Actual unresolved/unjudged note count `targets - hits - misses`
- Last **5 × 10ms** motor control frames: 8K `tanh(mean)` action,
  8K motor positions, velocities and pressed states, temporal head's
  time-to-next-event, 200ms probability, chord class
- Last **150ms** physical KeyDown events and TooEarly events, including
  `overload_before`, `overload_after`, `fail_overload`
- All TooEarly events in the recorded episode for gauge context
- Timing classifications and MAE on **on-policy** HUD observations
- Timing classifications and MAE on the **separate Expert-controlled** HUD
  trajectory (the trajectories are not frame-aligned)
- SHA256 read-back of protected inputs before and after diagnosis

Physics 1ms, Control 10ms, 263D visible input, 8K output. Target time and
physical state are **evaluator-only diagnostic fields**, not Student input.
No `optimizer.step()`, checkpoint write or modification, FINAL access, or
DAgger collection.

This replay diagnoses *where* the model fails. It does not prove a causal
mechanism for overload without the per-event trace, and it must not be used
to adapt hyperparameters on the already-inspected Validation20.
