"""D8 no-training CPU-dispatch diagnostic contract tests."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d8_cpu_dispatch_entropy as d8


def fake_expected():
    stage = {}
    for mode, episodes in d8.EXPECTED_EPISODES.items():
        stage[mode] = {
            "episodes": episodes, "targets": episodes * 16,
            "hits": 0, "extra_keydowns": 0,
            "physical_left": 0, "physical_right": 0,
            "mean_entropy": 1.3561848871858129, "rows": [],
        }
    return {"original_initial_sha256": "same", "original_stage0": stage}


def fake_evaluations():
    expected = fake_expected()
    native = {
        "condition": "native", "initial_state_sha256": "same",
        "runtime": {"cpu_model": "cpu", "torch": "2.5.1+cpu",
                    "platform": "Linux", "cpu_capability": "AVX2",
                    "torch_threads": 1},
        "stage0": deepcopy(expected["original_stage0"]),
        "archived_r1_first_difference": None,
    }
    default = deepcopy(native)
    default["condition"] = "default"
    default["runtime"]["cpu_capability"] = "DEFAULT"
    return expected, native, default


def test_dispatch_effect_when_entropy_differs():
    expected, native, default = fake_evaluations()
    default["stage0"]["argmax"]["mean_entropy"] += 1e-8
    default["archived_r1_first_difference"] = {"path": "entropy"}
    result = d8.compare(expected, native, default)
    assert result["classification"] == "dispatch_effect_observed"
    assert result["stage0_first_difference"] is not None
    assert result["physical"]["argmax"]["native"] == result["physical"]["argmax"]["default"]
    assert result["new_training_episodes"] == result["ppo_updates"] == 0


def test_no_effect_when_evaluations_identical():
    expected, native, default = fake_evaluations()
    result = d8.compare(expected, native, default)
    assert result["classification"] == "no_dispatch_effect_observed"
    assert result["stage0_first_difference"] is None


def test_identical_dispatch_is_not_evidence():
    expected, native, default = fake_evaluations()
    default["runtime"]["cpu_capability"] = "AVX2"
    assert d8.compare(expected, native, default)["classification"] == "dispatch_not_distinct"


def test_different_cpu_is_invalid():
    expected, native, default = fake_evaluations()
    default["runtime"]["cpu_model"] = "different"
    assert d8.compare(expected, native, default)["classification"] == "invalid_diagnostic"


def test_different_torch_is_invalid():
    expected, native, default = fake_evaluations()
    default["runtime"]["torch"] = "other"
    assert d8.compare(expected, native, default)["classification"] == "invalid_diagnostic"


def test_different_initial_weights_refused():
    expected, native, default = fake_evaluations()
    default["initial_state_sha256"] = "changed"
    with pytest.raises(ValueError, match="initial model weights"):
        d8.compare(expected, native, default)


def test_wrong_condition_label_refused():
    expected, native, default = fake_evaluations()
    default["condition"] = "native"
    with pytest.raises(ValueError, match="condition label"):
        d8.compare(expected, native, default)


def test_thread_count_refused():
    expected, native, default = fake_evaluations()
    default["runtime"]["torch_threads"] = 2
    with pytest.raises(ValueError, match="thread count"):
        d8.compare(expected, native, default)


def test_archival_validation_rejects_nonexact_d3(monkeypatch, tmp_path):
    expected = fake_expected()
    original = {
        "arm": "a0", "seed": 0,
        "checkpoint_sha256": "final", "initial_state_sha256": "same",
        "evaluations": [{"episode": 0, **expected["original_stage0"]},
                        {"episode": 128}, {"episode": 256}],
        "training": [{"episode": 8}],
    }
    replay = {
        "mode": "r1", "arm": "a0", "seed": 0,
        "original_reproduced": False,
        "source_checkpoint_sha256": "final",
        "replay_checkpoint_sha256": "final",
        "evaluations": original["evaluations"],
        "training_log": original["training"],
        "all_updates": [{}] * 32,
    }
    monkeypatch.setattr(d8, "source_models", lambda *args: ([original], {}))
    file = tmp_path / "d3.json"
    file.write_text(json.dumps({"models": [replay]}))
    with pytest.raises(ValueError, match="not exact"):
        d8.validate_archives(tmp_path, file, file)


def test_archival_validation_accepts_exact_d3(monkeypatch, tmp_path):
    expected = fake_expected()
    original = {
        "arm": "a0", "seed": 0,
        "checkpoint_sha256": "final", "initial_state_sha256": "same",
        "evaluations": [{"episode": 0, **expected["original_stage0"]},
                        {"episode": 128}, {"episode": 256}],
        "training": [{"episode": 8}],
    }
    replay = {
        "mode": "r1", "arm": "a0", "seed": 0,
        "original_reproduced": True,
        "source_checkpoint_sha256": "final",
        "replay_checkpoint_sha256": "final",
        "evaluations": original["evaluations"],
        "training_log": original["training"],
        "all_updates": [{}] * 32,
    }
    monkeypatch.setattr(d8, "source_models", lambda *args: ([original], {}))
    file = tmp_path / "d3.json"
    file.write_text(json.dumps({"models": [replay]}))
    result = d8.validate_archives(tmp_path, file)
    assert result["original_initial_sha256"] == "same"
    assert result["d3_r1_reproduced"] is True
