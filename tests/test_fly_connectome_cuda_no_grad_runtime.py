from __future__ import annotations

import pytest
import torch

from dmdod.fly_connectome_policy import NKeyFlyConnectomeActorCritic
from dmdod.malecns_connectome import build_malecns_weighted_core, save_malecns_core


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


def _model(tmp_path) -> NKeyFlyConnectomeActorCritic:
    return NKeyFlyConnectomeActorCritic(
        input_dim=6,
        key_count=4,
        core_path=_core(tmp_path),
        sensory_dim=3,
        recurrent_gain=0.8,
        projection_seed=11,
    )


def test_cpu_no_grad_keeps_sparse_runtime_and_dense_cuda_buffer_is_nonpersistent(tmp_path) -> None:
    model = _model(tmp_path)
    state = model.initial_state(torch.device("cpu"))

    selected = model._no_grad_recurrent_weight(state)

    assert selected.layout == torch.sparse_csr
    assert model._recurrent_no_grad_cuda_runtime.numel() == 0
    assert "_recurrent_no_grad_cuda_runtime" not in model.state_dict()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_no_grad_uses_dense_recurrent_while_grad_path_stays_sparse(tmp_path) -> None:
    device = torch.device("cuda")
    model = _model(tmp_path).to(device)
    model.prepare_recurrent_runtime()

    dense_runtime = model._recurrent_no_grad_cuda_runtime
    assert dense_runtime.layout == torch.strided
    assert tuple(dense_runtime.shape) == (model.hidden_dim, model.hidden_dim)
    assert dense_runtime.device.type == "cuda"
    assert "_recurrent_no_grad_cuda_runtime" not in model.state_dict()

    state = torch.randn(model.hidden_dim, device=device)
    injected = torch.randn(model.hidden_dim, device=device)
    dense_reference = model.recurrent_weight.detach().cpu().to_dense().to(device)

    # Poison only the sparse forward runtime.  no-grad must ignore it and keep
    # matching the dense fixed recurrent matrix; grad-enabled forward must still
    # consume the sparse runtime used by the custom autograd path.
    crow = torch.zeros(model.hidden_dim + 1, dtype=torch.int64, device=device)
    col = torch.empty(0, dtype=torch.int64, device=device)
    values = torch.empty(0, dtype=torch.float32, device=device)
    model._recurrent_weight_runtime = torch.sparse_csr_tensor(
        crow,
        col,
        values,
        size=(model.hidden_dim, model.hidden_dim),
        device=device,
    )

    with torch.no_grad():
        actual_no_grad = model._advance_injected(injected.clone(), state)
    expected_no_grad = torch.tanh(injected + torch.mv(dense_reference, state))
    assert torch.allclose(actual_no_grad, expected_no_grad, atol=1e-7, rtol=1e-6)

    grad_injected = injected.detach().clone().requires_grad_(True)
    actual_grad = model._advance_injected(grad_injected, state)
    expected_grad = torch.tanh(grad_injected)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-7, rtol=1e-6)
