from __future__ import annotations

import torch

from dmdod.connectome.fly_policy import NKeyFlyConnectomeActorCritic
from dmdod.connectome.human_visible_policy import (
    NKeyHumanVisibleControllerActorCritic,
)
from dmdod.connectome.malecns import build_malecns_weighted_core, save_malecns_core
from dmdod.envs.n_key import n_key_hud_real_chart_input_dim
from dmdod.training.human_visible import (
    HumanVisibleReplayChunk,
    freeze_human_visible_controller,
    train_human_visible_replay,
)
from dmdod.training.real_chart import (
    build_connectome_policy_from_checkpoint,
    build_human_visible_policy_from_parent_checkpoint,
)


def _core(tmp_path):
    artifact = build_malecns_weighted_core(
        torch.tensor([1, 2, 3, 4, 1, 3]),
        torch.tensor([2, 3, 4, 1, 3, 1]),
        torch.tensor([5, 6, 7, 8, 9, 10]),
        node_limit=4,
        min_weight=1,
    )
    path = tmp_path / "fly-core.pt"
    save_malecns_core(path, artifact)
    return path


def _models(tmp_path):
    path = _core(tmp_path)
    input_dim = n_key_hud_real_chart_input_dim(4)
    torch.manual_seed(11)
    parent = NKeyFlyConnectomeActorCritic(
        input_dim=input_dim,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.8,
        projection_seed=17,
    )
    relaxed = NKeyHumanVisibleControllerActorCritic(
        input_dim=input_dim,
        key_count=4,
        core_path=path,
        sensory_dim=3,
        recurrent_gain=0.8,
        projection_seed=17,
        controller_hidden_dim=16,
        connectome_context_dim=8,
        floor_context_dim=8,
        motor_context_dim=8,
        hud_context_dim=4,
    )
    relaxed.warm_start_from_fly_checkpoint(parent.state_dict())
    return parent, relaxed


def test_human_visible_warm_start_preserves_parent_actions(tmp_path) -> None:
    parent, relaxed = _models(tmp_path)
    observations = torch.randn(7, parent.input_dim)
    parent_state = parent.initial_state(torch.device("cpu"))
    relaxed_state = relaxed.initial_state(torch.device("cpu"))

    with torch.no_grad():
        for observation in observations:
            parent_mean, _, parent_value, parent_state = parent.forward_step(
                observation,
                parent_state,
            )
            relaxed_mean, _, relaxed_value, relaxed_state = relaxed.forward_step(
                observation,
                relaxed_state,
            )
            assert torch.allclose(relaxed_mean, parent_mean, atol=1e-7, rtol=1e-6)
            assert torch.allclose(relaxed_value, parent_value, atol=1e-7, rtol=1e-6)
            assert torch.allclose(
                relaxed_state[: parent.hidden_dim],
                parent_state,
                atol=1e-7,
                rtol=1e-6,
            )


def test_human_visible_step_and_sequence_match(tmp_path) -> None:
    _, model = _models(tmp_path)
    observations = torch.randn(6, model.input_dim)
    initial = model.initial_state(torch.device("cpu"))

    sequence_means, sequence_values, final_state = model.forward_sequence(
        observations,
        initial,
    )
    state = initial
    means = []
    values = []
    for observation in observations:
        mean, _, value, state = model.forward_step(observation, state)
        means.append(mean)
        values.append(value)

    assert torch.allclose(sequence_means, torch.stack(means), atol=1e-6, rtol=1e-5)
    assert torch.allclose(sequence_values, torch.stack(values), atol=1e-6, rtol=1e-5)
    assert torch.allclose(final_state, state, atol=1e-6, rtol=1e-5)


def test_human_visible_training_keeps_parent_path_frozen(tmp_path) -> None:
    _, model = _models(tmp_path)
    trainable = freeze_human_visible_controller(model)
    inherited_before = {
        "sensory": model.sensory.weight.detach().clone(),
        "actor": model.actor_mean.weight.detach().clone(),
        "projection": model.input_projection.detach().clone(),
    }
    floor_before = model.floor_encoder[0].weight.detach().clone()

    observations = torch.randn(8, model.input_dim)
    target = torch.zeros(8, 4)
    target[:, 0] = 1.0
    chunk = HumanVisibleReplayChunk(
        observations=observations,
        teacher_actions=target,
        initial_state=model.initial_state(torch.device("cpu")),
        source="test",
        start=0,
    )
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    metrics = train_human_visible_replay(
        model,
        [chunk],
        optimizer=optimizer,
        updates=3,
        grad_clip=1.0,
        seed=7,
    )

    assert metrics.updates == 3
    assert not torch.equal(model.controller_delta.weight, torch.zeros_like(model.controller_delta.weight))
    assert not torch.equal(model.floor_encoder[0].weight, floor_before)
    assert torch.equal(model.sensory.weight, inherited_before["sensory"])
    assert torch.equal(model.actor_mean.weight, inherited_before["actor"])
    assert torch.equal(model.input_projection, inherited_before["projection"])


def test_human_visible_metadata_records_information_contract(tmp_path) -> None:
    _, model = _models(tmp_path)
    metadata = model.checkpoint_metadata()
    assert metadata["n_key_policy_backend"] == "human_visible_controller"
    assert metadata["human_visible_privileged_inputs"] is False
    assert metadata["human_visible_floor_slots"] == 15
    assert metadata["human_visible_hud_feature_dim"] == 12
    assert metadata["human_visible_residual_warm_start"] is True


def test_human_visible_checkpoint_upgrade_and_reload(tmp_path) -> None:
    parent, _ = _models(tmp_path)
    parent_checkpoint = {
        **parent.checkpoint_metadata(),
        "model_state": parent.state_dict(),
    }

    upgraded = build_human_visible_policy_from_parent_checkpoint(
        parent_checkpoint,
        device=torch.device("cpu"),
    )
    assert isinstance(upgraded, NKeyHumanVisibleControllerActorCritic)

    checkpoint = {
        **upgraded.checkpoint_metadata(),
        "model_state": upgraded.state_dict(),
    }
    restored = build_connectome_policy_from_checkpoint(
        checkpoint,
        device=torch.device("cpu"),
    )
    assert isinstance(restored, NKeyHumanVisibleControllerActorCritic)
    assert restored.policy_state_dim == upgraded.policy_state_dim

    observation = torch.randn(parent.input_dim)
    with torch.no_grad():
        expected, _, _, _ = upgraded.forward_step(
            observation,
            upgraded.initial_state(torch.device("cpu")),
        )
        actual, _, _, _ = restored.forward_step(
            observation,
            restored.initial_state(torch.device("cpu")),
        )
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
