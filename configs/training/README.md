# Training configs

Each TOML file describes one reproducible training run. Prefer creating a new
config over creating a new Python trainer.

Required top-level field:

```toml
mode = "budget_action_trust"
```

The current schema uses these tables:

- `[run]`: dataset/checkpoint/output/device.
- `[data]`: anchor, validation, and chunk limits.
- `[action_trust]`: actor optimizer and action-space trust settings.
- `[budget]`: wall-clock budget and adaptive radius policy.

All values used by a run are copied into the resulting checkpoint metadata as
`training_config`.
