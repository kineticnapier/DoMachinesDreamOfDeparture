"""D9 parser and precommitted classification contracts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d9_historical_runtime as d9


def log_bytes():
    return ("""2026-11-01T13:08:57Z Current runner version: '2.338.2'
2026-11-01T13:08:57Z Image: ubuntu-26.04
2026-11-01T13:08:57Z Image Release: https://github.com/actions/runner-images/releases/tag/ubuntu26%2F20261025.179
2026-11-01T13:08:58Z   PYTHONHASHSEED: 0
2026-11-01T13:08:58Z   OMP_NUM_THREADS: 1
2026-11-01T13:08:58Z   MKL_NUM_THREADS: 1
2026-11-01T13:08:58Z pythonLocation: /opt/hostedtoolcache/Python/3.11.17/x64
2026-11-01T13:09:18Z Successfully installed torch-2.5.1+cpu
""").encode()


def fixture():
    runtime = {"torch": "2.5.1+cpu", "torch_threads": 1,
               "mkldnn_enabled": True,
               "cpu_model": "AMD EPYC 7763 64-Core Processor"}
    d7 = {"classification": "initial_evaluation_diverged",
          "stage0_difference": {"expected": d9.ORIGINAL_ENTROPY,
                                "actual": d9.NEW_ENTROPY},
          "runtime": runtime}
    native = {"initial_state_sha256": "same",
              "runtime": {**runtime, "cpu_capability": "AVX2"},
              "stage0": {"argmax": {"mean_entropy": d9.NEW_ENTROPY,
                                   "hits": 0, "extra_keydowns": 0,
                                   "physical_left": 0, "physical_right": 0,
                                   "targets": 144},
                         "sampled": {"hits": 5, "extra_keydowns": 44,
                                     "physical_left": 49, "physical_right": 0,
                                     "targets": 576}}}
    default = {"initial_state_sha256": "same",
               "runtime": {**runtime, "cpu_capability": "NO AVX"},
               "stage0": {"argmax": {"mean_entropy": d9.DEFAULT_ENTROPY}}}
    return d7, native, default


def test_parse_log():
    x = d9.parse_log(log_bytes())
    assert x["runner_version"] == "2.338.2"
    assert x["image"] == "ubuntu-26.04"
    assert x["python_version"] == "3.11.17"
    assert x["torch_version"] == "2.5.1+cpu"
    assert x["PYTHONHASHSEED"] == "0"
    assert x["cpu_model_in_log"] is None


def test_all_evidence_but_unknown_cpu():
    d7, native, default = fixture()
    r = d9.audit({k: log_bytes() for k in d9.JOBS}, d7, native, default)
    assert r["classification"] == "historical_cpu_unresolved"
    assert r["visible_environment_consistent"]
    assert r["unknown_historical_cpu_models"] == ["r1", "d3", "d5"]
    assert r["new_training_episodes"] == r["ppo_updates"] == 0
    assert r["new_physical_keydowns"] == 0


def test_missing_log_invalid():
    d7, native, default = fixture()
    r = d9.audit({"r1": log_bytes()}, d7, native, default)
    assert r["classification"] == "invalid_archive_audit"


def test_conflicting_image_invalid():
    d7, native, default = fixture()
    logs = {k: log_bytes() for k in d9.JOBS}
    logs["d3"] = logs["d3"].replace(b"ubuntu-26.04", b"ubuntu-24.04")
    r = d9.audit(logs, d7, native, default)
    assert r["classification"] == "invalid_archive_audit"


def test_mismatched_entropy_invalid():
    d7, native, default = fixture()
    d7["stage0_difference"]["actual"] += 1e-5
    r = d9.audit({k: log_bytes() for k in d9.JOBS}, d7, native, default)
    assert r["classification"] == "invalid_archive_audit"


def test_mismatched_weights_invalid():
    d7, native, default = fixture()
    default["initial_state_sha256"] = "different"
    r = d9.audit({k: log_bytes() for k in d9.JOBS}, d7, native, default)
    assert r["classification"] == "invalid_archive_audit"


def test_cpu_model_disagreement_invalid():
    d7, native, default = fixture()
    d7["runtime"]["cpu_model"] = "different"
    r = d9.audit({k: log_bytes() for k in d9.JOBS}, d7, native, default)
    assert r["classification"] == "invalid_archive_audit"


def test_incomplete_version_invalid():
    d7, native, default = fixture()
    logs = {k: log_bytes() for k in d9.JOBS}
    logs["r1"] = logs["r1"].replace(b"torch-2.5.1+cpu", b"unknown")
    r = d9.audit(logs, d7, native, default)
    assert r["classification"] == "invalid_archive_audit"
