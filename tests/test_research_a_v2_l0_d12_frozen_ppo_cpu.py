"""D12: frozen PPO input and CPU-update comparison contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d12_frozen_ppo_cpu as d12


def example():
    capture = {"initial_state_sha256": "weights", "batch_sha256": "batch",
               "episodes": 8, "targets": 128, "hits": 1, "extras": 3,
               "physical_keydowns": 4}
    native = {
        "condition": "native", "initial_state_sha256": "weights",
        "batch_sha256": "batch",
        "runtime": {"python": "3.11", "torch": "2.5.1+cpu", "platform": "Linux",
                    "cpu_model": "CPU", "torch_threads": 1,
                    "cpu_capability": "AVX2"},
        "episodes": 8, "targets": 128, "hits": 1, "extras": 3,
        "physical_keydowns": 4, "ppo_updates": 1, "adam_steps": 2,
        "new_training_episodes": 0, "motor_steps": 0,
        "new_physical_keydowns": 0,
        "weights_after": {"actor.weight": [0, 1065353216]},
        "actor_and_critic_gradients": [{"actor.weight": [0]}, {"actor.weight": [0]}],
        "adam_after": {"actor.weight": {"exp_avg": [0], "exp_avg_sq": [0]}},
        "losses": {"loss": 1.0, "entropy": 0.5},
    }
    default = deepcopy(native)
    default["condition"] = "default"
    default["runtime"]["cpu_capability"] = "NO AVX"
    return native, default, capture


def test_float32_bits():
    assert d12.tensor_bits(torch.tensor([0.0, 1.0])) == [0, 1065353216]


def test_nonfinite_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        d12.tensor_bits(torch.tensor([float("nan")]))


def test_identical_update_is_negative_result():
    n, d, c = example()
    out = d12.compare(n, d, c)
    assert out["classification"] == "no_ppo_update_difference_observed"
    assert out["weights"]["different_elements"] == 0
    assert out["physical_keydowns"] == 4


def test_one_ulp_update_difference():
    n, d, c = example()
    d["weights_after"]["actor.weight"][1] += 1
    out = d12.compare(n, d, c)
    assert out["classification"] == "ppo_update_difference_observed"
    assert out["weights"]["different_elements"] == 1
    assert out["weights"]["first_difference"]["index"] == 1


def test_adam_difference_detected():
    n, d, c = example()
    d["adam_after"]["actor.weight"]["exp_avg"][0] = 1
    out = d12.compare(n, d, c)
    assert out["adam"]["exp_avg"]["different_elements"] == 1
    assert out["classification"] == "ppo_update_difference_observed"


def test_loss_difference_detected():
    n, d, c = example()
    d["losses"]["loss"] += 0.125
    assert d12.compare(n, d, c)["classification"] == "ppo_update_difference_observed"


def test_dispatch_must_differ():
    n, d, c = example()
    d["runtime"]["cpu_capability"] = n["runtime"]["cpu_capability"]
    assert d12.compare(n, d, c)["classification"] == "dispatch_not_distinct"


def test_runtime_must_match():
    n, d, c = example()
    d["runtime"]["torch"] = "different"
    assert d12.compare(n, d, c)["classification"] == "invalid_diagnostic"


def test_batch_identity_must_match():
    n, d, c = example()
    d["batch_sha256"] = "different"
    with pytest.raises(ValueError, match="identity"):
        d12.compare(n, d, c)


def test_physical_accounting_must_match():
    n, d, c = example()
    d["physical_keydowns"] = 5
    with pytest.raises(ValueError, match="physical"):
        d12.compare(n, d, c)


def test_update_count_must_be_one():
    n, d, c = example()
    d["ppo_updates"] = 32
    with pytest.raises(ValueError, match="one-update"):
        d12.compare(n, d, c)


def test_compare_rejects_different_tensor_shape():
    with pytest.raises(ValueError, match="shapes"):
        d12.compare_bits({"a": [0]}, {"a": [0, 1]})


def test_batch_digest_requires_eight_episodes():
    with pytest.raises(ValueError, match="eight"):
        d12.batch_digest([])
