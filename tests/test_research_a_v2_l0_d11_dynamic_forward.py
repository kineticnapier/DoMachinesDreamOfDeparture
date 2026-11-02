"""D11: pre-registered dynamic observation and CPU comparison contracts."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d11_dynamic_forward as d11


def pair(monkeypatch):
    archive = {"stage0": {
        "argmax": {"mean_entropy": 0.0},
        "sampled": {"mean_entropy": 0.0}}}
    monkeypatch.setattr(d11, "load_archives",
                        lambda *_: (archive, archive, "weights"))
    rows = []
    for i in range(45):
        rows.append({"mode": "argmax" if i < 9 else "sampled",
                     "case_index": i if i < 9 else (i - 9) // 4,
                     "repetition": 0 if i < 9 else (i - 9) % 4,
                     "stages": [[[0] for _ in d11.STAGES]]})
    native = {"condition": "native", "initial_state_sha256": "weights",
              "observations_sha256": "observations", "episodes": 45,
              "steps": 45, "stages": list(d11.STAGES),
              "rows": rows,
              "runtime": {"torch": "2.5.1+cpu", "platform": "Linux",
                          "cpu_model": "CPU", "torch_threads": 1,
                          "cpu_capability": "AVX2"},
              "new_training_episodes": 0, "ppo_updates": 0,
              "motor_steps": 0, "new_physical_keydowns": 0}
    default = deepcopy(native)
    default["condition"] = "default"
    default["runtime"]["cpu_capability"] = "NO AVX"
    capture = {"observations_sha256": "observations", "steps": 45,
               "native_physical_keydowns": 49, "native_hits": 5,
               "native_extras": 44}
    return native, default, capture


def run_compare(n, d, c):
    return d11.compare(n, d, c, Path("unused"), Path("unused"))


def test_frozen_observations_without_stage_difference(monkeypatch):
    n, d, c = pair(monkeypatch)
    result = run_compare(n, d, c)
    assert result["classification"] == "no_dynamic_stage_divergence"
    assert result["first_difference"] is None
    assert result["archived_entropy_exact"]["native"]["sampled"] is True
    assert result["historical_d3_d5_cause_verified"] is False


def test_first_stage_precedes_first_observation(monkeypatch):
    n, d, c = pair(monkeypatch)
    d["rows"][0]["stages"][0][d11.STAGES.index("softmax")] = [1065353216]
    d["rows"][4]["stages"][0][d11.STAGES.index("shared.linear1")] = [1065353216]
    result = run_compare(n, d, c)
    assert result["classification"] == "dynamic_stage_divergence_observed"
    assert result["first_difference"]["stage"] == "shared.linear1"
    assert result["first_difference"]["row_index"] == 4
    assert result["different_elements_by_stage"]["softmax"] == 1


def test_entropy_mismatch_stops_without_tolerance(monkeypatch):
    n, d, c = pair(monkeypatch)
    d["rows"][0]["stages"][0][d11.STAGES.index("entropy")] = [1065353216]
    result = run_compare(n, d, c)
    assert result["classification"] == "archived_entropy_not_reproduced"
    assert result["archived_entropy_exact"]["default"]["argmax"] is False


def test_same_cpu_dispatch_invalid(monkeypatch):
    n, d, c = pair(monkeypatch)
    d["runtime"]["cpu_capability"] = "AVX2"
    assert run_compare(n, d, c)["classification"] == "invalid_diagnostic"


def test_different_cpu_model_invalid(monkeypatch):
    n, d, c = pair(monkeypatch)
    d["runtime"]["cpu_model"] = "other CPU"
    assert run_compare(n, d, c)["classification"] == "invalid_diagnostic"


def test_mismatched_observation_digest_refused(monkeypatch):
    n, d, c = pair(monkeypatch)
    d["observations_sha256"] = "not-frozen"
    with pytest.raises(ValueError, match="conditions differ"):
        run_compare(n, d, c)


def test_forward_must_not_run_motor(monkeypatch):
    n, d, c = pair(monkeypatch)
    d["motor_steps"] = 1
    with pytest.raises(ValueError, match="must not step Motor"):
        run_compare(n, d, c)


def test_float32_bit_roundtrip():
    assert d11.float_from_bits(1065353216) == 1.0
    assert d11.float_from_bits(-1082130432) == -1.0
    assert d11.digest([[1, 2], [3, 4]]) == d11.digest([[1, 2], [3, 4]])
    assert d11.digest([[1, 2]]) != d11.digest([[2, 1]])


def test_capture_episode_records_before_motor_step(monkeypatch):
    @dataclass
    class Score:
        physical_keydowns: int = 0
        matched_keys: tuple = ()

    class FakeEnv:
        def __init__(self, *, arm, interval_ms, start_s):
            assert arm == "a0"
            self.down_trace = []
            self.did_step = False

        def reset(self):
            return [0.0] * 8

        def step(self, action):
            assert not self.did_step
            self.did_step = True
            assert action in range(4)
            return [0.0] * 8, 0.0, True

        def result(self):
            return Score()

    class FakeModel:
        def __call__(self, x):
            return torch.zeros(4), torch.tensor(0.0)

    monkeypatch.setattr(d11.l0, "LoggedLowSpeedEnv", FakeEnv)
    row, observations = d11.capture_episode(FakeModel(), "argmax", 210, .835, None)
    assert row["steps"] == 1
    assert row["physical_trace"] == []
    assert len(observations) == 1
    assert observations[0] == [0] * 8
