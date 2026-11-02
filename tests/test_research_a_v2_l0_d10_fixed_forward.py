"""D10: pre-registered bit-exact CPU forward-stage diagnostic contracts."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d10_fixed_forward as d10


def fake_pair():
    stages = {s: {"bits": [0], "values": [0.]} for s in d10.STAGES}
    row = {"label": "case", "stages": stages}
    native = {"condition": "native", "initial_state_sha256": "same",
              "input_bits_sha256": "same-input", "probe_count": 13,
              "stages": list(d10.STAGES),
              "probes": [deepcopy(row) for _ in range(13)],
              "runtime": {"torch": "2.5.1+cpu", "platform": "Linux",
                          "cpu_model": "CPU", "torch_threads": 1,
                          "cpu_capability": "AVX2"},
              "new_training_episodes": 0, "ppo_updates": 0,
              "motor_steps": 0, "new_physical_keydowns": 0}
    default = deepcopy(native)
    default["condition"] = "default"
    default["runtime"]["cpu_capability"] = "NO AVX"
    return native, default


def test_no_stage_difference_is_not_causal_proof():
    native, default = fake_pair()
    result = d10.compare(native, default)
    assert result["classification"] == "no_stage_divergence_on_fixed_probes"
    assert result["first_difference"] is None
    assert result["historical_d3_d5_cause_verified"] is False


def test_first_stage_uses_stage_order_not_probe_order():
    native, default = fake_pair()
    default["probes"][0]["stages"]["entropy"] = {"bits": [1], "values": [1.]}
    default["probes"][4]["stages"]["shared.linear1"] = {"bits": [2], "values": [2.]}
    result = d10.compare(native, default)
    assert result["classification"] == "stage_divergence_observed"
    assert result["first_difference"]["stage"] == "shared.linear1"
    assert result["first_difference"]["probe_index"] == 4
    assert result["different_elements_by_stage"]["entropy"] == 1


def test_same_dispatch_is_not_distinct():
    native, default = fake_pair()
    default["runtime"]["cpu_capability"] = "AVX2"
    assert d10.compare(native, default)["classification"] == "dispatch_not_distinct"


def test_different_cpu_invalidates_diagnostic():
    native, default = fake_pair()
    default["runtime"]["cpu_model"] = "different"
    assert d10.compare(native, default)["classification"] == "invalid_diagnostic"


@pytest.mark.parametrize("field", ["initial_state_sha256", "input_bits_sha256"])
def test_mismatched_shared_inputs_refused(field):
    native, default = fake_pair()
    default[field] = "changed"
    with pytest.raises(ValueError, match="differ"):
        d10.compare(native, default)


def test_training_or_motor_activity_refused():
    native, default = fake_pair()
    native["motor_steps"] = 1
    with pytest.raises(ValueError, match="must not execute"):
        d10.compare(native, default)


def test_stage_length_mismatch_refused():
    native, default = fake_pair()
    default["probes"][0]["stages"]["actor.logits"]["bits"].append(0)
    with pytest.raises(ValueError, match="lengths differ"):
        d10.compare(native, default)


def test_d8_archive_validation():
    n = {"condition": "native", "initial_state_sha256": "digest",
         "runtime": {"torch": "2.5.1+cpu"}}
    d = {"condition": "default", "initial_state_sha256": "digest",
         "loaded_original_native_weights": True,
         "runtime": {"torch": "2.5.1+cpu"}}
    assert d10.validate_d8(n, d) == "digest"
    d["initial_state_sha256"] = "different"
    with pytest.raises(ValueError, match="not identical"):
        d10.validate_d8(n, d)


def test_original_actor_forward_matches_decomposed_layers():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = d10.l0.ActorCritic().eval()
    x = torch.tensor([1., -1., .75, -.75, .5, -.5, .25, -.25])
    stages = d10.forward_stages(model, x)
    assert list(stages) == list(d10.STAGES)
    assert len(d10.bits(stages["actor.logits"])) == 4
    assert len(d10.bits(stages["entropy"])) == 1


def test_fixed_probe_count_without_motor_step(monkeypatch):
    class FakeEnv:
        def __init__(self, *, arm, interval_ms, start_s):
            assert arm == "a0"
        def reset(self):
            return [0.] * 8
        def step(self, *_):
            raise AssertionError("Motor step prohibited")
    monkeypatch.setattr(d10.l0, "LoggedLowSpeedEnv", FakeEnv)
    probes = d10.inputs()
    assert len(probes) == 13
    assert all(x.shape == (8,) for _, x in probes)
    assert [label for label, _ in probes][-4:] == [s[0] for s in d10.SYNTHETIC]
