from __future__ import annotations

from pathlib import Path

import pytest

from dmdod.training.config import load_training_config, validate_checkpoint_paths


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
max_nonbest_accepts = 4

[trajectory_probe]
candidate_action_rms = 0.000025

[boundary_trust]
preserve_safe_only = true
mismatch_grace_s = 0.030

[human_visible]
updates_per_trial = 16
lr = 0.0002
min_lr = 0.000002
controller_lr = 0.00003
updates_per_epoch = 12
transition_updates_per_epoch = 6
transition_epochs = 4
burn_in_steps = 64
supervised_steps = 128
anchors_per_batch = 3
progression_streak = 2
catastrophic_patience = 2
max_epochs = 7
weight_decay = 0.0001
grad_clip = 1.5
seed = 99
allow_legacy_train_identity_resume = true
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
    assert config.budget.max_nonbest_accepts == 4
    assert config.trajectory_probe.candidate_action_rms == 2.5e-5
    assert config.boundary_trust.preserve_safe_only is True
    assert config.boundary_trust.mismatch_grace_s == 0.03
    assert config.human_visible.updates_per_trial == 16
    assert config.human_visible.lr == 2e-4
    assert config.human_visible.min_lr == 2e-6
    assert config.human_visible.controller_lr == 3e-5
    assert config.human_visible.updates_per_epoch == 12
    assert config.human_visible.transition_updates_per_epoch == 6
    assert config.human_visible.transition_epochs == 4
    assert config.human_visible.burn_in_steps == 64
    assert config.human_visible.supervised_steps == 128
    assert config.human_visible.anchors_per_batch == 3
    assert config.human_visible.progression_streak == 2
    assert config.human_visible.catastrophic_patience == 2
    assert config.human_visible.max_epochs == 7
    assert config.human_visible.grad_clip == 1.5
    assert config.human_visible.seed == 99
    assert config.human_visible.allow_legacy_train_identity_resume is True


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


def test_invalid_nonbest_restart_limit_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad-restart.toml"
    path.write_text(
        """
mode = "budget_survival_trust"

[run]
dataset = "x"
checkpoint = "in.pt"
output = "out.pt"

[budget]
max_nonbest_accepts = 0
""",
        encoding="utf-8",
    )

    try:
        load_training_config(path)
    except ValueError as exc:
        assert "max_nonbest_accepts" in str(exc)
    else:
        raise AssertionError("zero non-best restart limit must fail")


def test_invalid_human_visible_lr_order_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad-visible.toml"
    path.write_text(
        """
mode = "human_visible_dagger"

[run]
dataset = "x"
checkpoint = "in.pt"
output = "out.pt"

[human_visible]
lr = 0.000001
min_lr = 0.00001
""",
        encoding="utf-8",
    )

    try:
        load_training_config(path)
    except ValueError as exc:
        assert "human_visible.min_lr" in str(exc)
    else:
        raise AssertionError("invalid human-visible learning-rate order must fail")


def _write_checkpoint_path_config(
    tmp_path: Path, *, checkpoint: Path, output: Path, mode: str = "budget_action_trust"
) -> Path:
    config_file = tmp_path / "paths.toml"
    config_file.write_text(
        f'mode = "{mode}"\n\n'
        f'[run]\ndataset = "unused"\n'
        f'checkpoint = "{checkpoint.as_posix()}"\n'
        f'output = "{output.as_posix()}"\n',
        encoding="utf-8",
    )
    return config_file


@pytest.mark.parametrize(
    "mode",
    ["budget_action_trust", "human_visible_dagger", "human_visible_curriculum_dagger"],
)
def test_checkpoint_source_cannot_be_output(tmp_path: Path, mode: str) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"original-checkpoint")
    config_file = _write_checkpoint_path_config(
        tmp_path, checkpoint=source, output=source, mode=mode
    )
    with pytest.raises(ValueError, match="same file"):
        load_training_config(config_file)
    assert source.read_bytes() == b"original-checkpoint"


def test_checkpoint_alias_with_parent_segments_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    (tmp_path / "sub").mkdir()
    alias = tmp_path / "sub" / ".." / "source.pt"
    config_file = _write_checkpoint_path_config(
        tmp_path, checkpoint=source, output=alias
    )
    with pytest.raises(ValueError, match="same file"):
        load_training_config(config_file)


def test_checkpoint_hardlink_alias_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.pt"
    try:
        alias.hardlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"hard links unavailable: {exc}")
    with pytest.raises(ValueError, match="same file"):
        validate_checkpoint_paths(source, alias)


def test_checkpoint_symlink_alias_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    alias = tmp_path / "alias.pt"
    try:
        alias.symlink_to(source)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic links unavailable: {exc}")
    with pytest.raises(ValueError, match="same file"):
        validate_checkpoint_paths(source, alias)


def test_distinct_checkpoint_output_remains_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    output = tmp_path / "different.pt"
    config_file = _write_checkpoint_path_config(
        tmp_path, checkpoint=source, output=output
    )
    config = load_training_config(config_file)
    assert Path(config.run.checkpoint) == source
    assert Path(config.run.output) == output
    assert source.read_bytes() == b"original"
    assert not output.exists()


def test_runner_rechecks_manually_constructed_config(tmp_path: Path) -> None:
    from dataclasses import replace
    from dmdod.training.runner import _prepare

    source = tmp_path / "source.pt"
    source.write_bytes(b"original")
    config_file = _write_checkpoint_path_config(
        tmp_path, checkpoint=source, output=tmp_path / "output.pt"
    )
    valid = load_training_config(config_file)
    invalid = replace(valid, run=replace(valid.run, output=str(source)))
    with pytest.raises(ValueError, match="same file"):
        _prepare(invalid)
    assert source.read_bytes() == b"original"


def test_legacy_train_identity_opt_in_requires_boolean(tmp_path: Path) -> None:
    path = tmp_path / "bad-legacy-identity.toml"
    path.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "x"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        '[human_visible]\nallow_legacy_train_identity_resume = "false"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="allow_legacy_train_identity_resume"):
        load_training_config(path)
