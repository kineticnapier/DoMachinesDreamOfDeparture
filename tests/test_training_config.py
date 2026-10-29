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


@pytest.mark.parametrize("section, key", [
    ("action_trust", "lr"),
    ("action_trust", "stay_coef"),
    ("action_trust", "initial_action_rms"),
    ("action_trust", "min_action_rms"),
    ("action_trust", "min_lr"),
    ("budget", "hours"),
    ("budget", "reserve_minutes"),
    ("budget", "reject_shrink"),
    ("budget", "safe_grow"),
    ("trajectory_probe", "candidate_action_rms"),
    ("boundary_trust", "mismatch_grace_s"),
    ("human_visible", "lr"),
    ("human_visible", "min_lr"),
    ("human_visible", "controller_lr"),
    ("human_visible", "weight_decay"),
    ("human_visible", "grad_clip"),
])
@pytest.mark.parametrize("invalid", ["nan", "+inf", "-inf"])
def test_nonfinite_training_float_is_rejected(
    tmp_path: Path, section: str, key: str, invalid: str
) -> None:
    path = tmp_path / "nonfinite.toml"
    path.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "x"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        f'[{section}]\n{key} = {invalid}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"{section}\.{key} must be finite"):
        load_training_config(path)


def test_finite_training_float_boundaries_remain_allowed(tmp_path: Path) -> None:
    path = tmp_path / "finite.toml"
    path.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "x"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        '[action_trust]\nstay_coef = 0.0\n'
        '[budget]\nreserve_minutes = 0.0\nsafe_grow = 1.0\n'
        '[boundary_trust]\nmismatch_grace_s = 0.0\n'
        '[human_visible]\nweight_decay = 0.0\n',
        encoding="utf-8",
    )
    config = load_training_config(path)
    assert config.action_trust.stay_coef == 0.0
    assert config.budget.reserve_minutes == 0.0
    assert config.budget.safe_grow == 1.0
    assert config.boundary_trust.mismatch_grace_s == 0.0
    assert config.human_visible.weight_decay == 0.0


@pytest.mark.parametrize("literal", ("2.5", "3.0", "true", '"3"', "nan", "inf"))
def test_nonbest_restart_limit_must_be_toml_integer(tmp_path, literal: str) -> None:
    path = tmp_path / "noninteger-restart.toml"
    path.write_text(
        'mode = "budget_survival_trust"\n'
        '[run]\n'
        'dataset = "unused"\n'
        'checkpoint = "in.pt"\n'
        'output = "out.pt"\n'
        '[budget]\n'
        f'max_nonbest_accepts = {literal}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="budget.max_nonbest_accepts must be an integer"):
        load_training_config(path)


# TOML integers must not be silently truncated or coerced to booleans.
# All listed fields previously used int(...) directly or through
# _optional_positive_int, including values that control training budgets.
@pytest.mark.parametrize(
    "section,key",
    [
        ("data", "anchor_limit"),
        ("data", "validation_limit"),
        ("data", "chunk_steps"),
        ("action_trust", "actor_steps"),
        ("action_trust", "lr_backoffs"),
        ("budget", "max_trials"),
        ("human_visible", "updates_per_trial"),
        ("human_visible", "updates_per_epoch"),
        ("human_visible", "transition_updates_per_epoch"),
        ("human_visible", "transition_epochs"),
        ("human_visible", "burn_in_steps"),
        ("human_visible", "supervised_steps"),
        ("human_visible", "anchors_per_batch"),
        ("human_visible", "progression_streak"),
        ("human_visible", "catastrophic_patience"),
        ("human_visible", "max_epochs"),
        ("human_visible", "seed"),
    ],
)
@pytest.mark.parametrize("literal", ("2.5", "3.0", "true", '"3"', "nan", "inf"))
def test_integer_config_fields_reject_implicit_conversion(
    tmp_path: Path, section: str, key: str, literal: str
) -> None:
    config_file = tmp_path / "noninteger.toml"
    config_file.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "unused"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        f'[{section}]\n{key} = {literal}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"{section}\.{key} must be an integer"):
        load_training_config(config_file)


def test_integer_config_zero_boundaries_remain_valid(tmp_path: Path) -> None:
    config_file = tmp_path / "integer-boundaries.toml"
    config_file.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "unused"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        '[action_trust]\nlr_backoffs = 0\n'
        '[human_visible]\ntransition_epochs = 0\nburn_in_steps = 0\nseed = 0\n',
        encoding="utf-8",
    )
    config = load_training_config(config_file)
    assert config.action_trust.lr_backoffs == 0
    assert config.human_visible.transition_epochs == 0
    assert config.human_visible.burn_in_steps == 0
    assert config.human_visible.seed == 0


# These switches control training policy and optional diagnostics. Using
# bool("false") would silently enable a switch and change experiment conditions.
@pytest.mark.parametrize(
    "section,key",
    [
        ("boundary_trust", "preserve_safe_only"),
        ("human_visible", "alpha_sweep_enabled"),
        ("human_visible", "alpha_sweep_compare_transitions"),
        ("human_visible", "alpha_sweep_recovery_probe_enabled"),
    ],
)
@pytest.mark.parametrize("literal", ('"false"', '"true"', "0", "1", "0.0", "1.0"))
def test_training_boolean_switches_reject_implicit_conversion(
    tmp_path: Path, section: str, key: str, literal: str
) -> None:
    path = tmp_path / "nonboolean-switch.toml"
    path.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "unused"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        f'[{section}]\n{key} = {literal}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=rf"{section}\.{key} must be boolean"):
        load_training_config(path)


@pytest.mark.parametrize("enabled", (False, True))
def test_training_boolean_switches_preserve_explicit_values(
    tmp_path: Path, enabled: bool
) -> None:
    literal = "true" if enabled else "false"
    path = tmp_path / "boolean-switches.toml"
    path.write_text(
        'mode = "human_visible_curriculum_dagger"\n'
        '[run]\ndataset = "unused"\ncheckpoint = "in.pt"\noutput = "out.pt"\n'
        f'[boundary_trust]\npreserve_safe_only = {literal}\n'
        f'[human_visible]\nalpha_sweep_enabled = {literal}\n'
        f'alpha_sweep_compare_transitions = {literal}\n'
        f'alpha_sweep_recovery_probe_enabled = {literal}\n'
        'alpha_sweep_expected_best = [0, 0, 0, 0]\n'
        'alpha_sweep_expected_candidate = [0, 0, 0, 0]\n',
        encoding="utf-8",
    )
    config = load_training_config(path)
    assert config.boundary_trust.preserve_safe_only is enabled
    assert config.human_visible.alpha_sweep_enabled is enabled
    assert config.human_visible.alpha_sweep_compare_transitions is enabled
    assert config.human_visible.alpha_sweep_recovery_probe_enabled is enabled
