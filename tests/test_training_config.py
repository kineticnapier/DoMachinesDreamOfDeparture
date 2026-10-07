from __future__ import annotations

from dmdod.training.config import load_training_config


def test_load_training_config(tmp_path) -> None:
    path = tmp_path / "train.toml"
    path.write_text(
        """
mode = "budget_action_trust"

[run]
dataset = "data/example"
checkpoint = "checkpoints/in.pt"
output = "checkpoints/out.pt"
device = "cuda"

[data]
anchor_limit = 20
validation_limit = 10
chunk_steps = 192

[action_trust]
actor_steps = 4
lr = 0.0003
stay_coef = 20
initial_action_rms = 0.01
min_action_rms = 0.000001

[budget]
hours = 2
reserve_minutes = 5
max_trials = 99
reject_shrink = 0.5
safe_grow = 1.25

[trajectory_probe]
candidate_action_rms = 0.000025

[boundary_trust]
preserve_safe_only = true
mismatch_grace_s = 0.030
""",
        encoding="utf-8",
    )

    config = load_training_config(path)

    assert config.mode == "budget_action_trust"
    assert config.run.dataset == "data/example"
    assert config.run.device == "cuda"
    assert config.data.anchor_limit == 20
    assert config.action_trust.actor_steps == 4
    assert config.action_trust.min_action_rms == 1e-6
    assert config.budget.hours == 2.0
    assert config.budget.max_trials == 99
    assert config.trajectory_probe.candidate_action_rms == 2.5e-5
    assert config.boundary_trust.preserve_safe_only is True
    assert config.boundary_trust.mismatch_grace_s == 0.03


def test_invalid_radius_order_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
mode = "budget_action_trust"

[run]
dataset = "x"
checkpoint = "in.pt"
output = "out.pt"

[action_trust]
initial_action_rms = 0.001
min_action_rms = 0.01
""",
        encoding="utf-8",
    )

    try:
        load_training_config(path)
    except ValueError as exc:
        assert "cannot exceed" in str(exc)
    else:
        raise AssertionError("invalid action trust radius order must fail")
