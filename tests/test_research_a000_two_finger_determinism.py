"""Experiment 000 infrastructure tests, not an experiment-002 training result."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a000_two_finger_replay import deterministic_actions, replay, summary


def test_000_same_actions_exact_same_trace_with_fresh_environments():
    a = replay(seed=123, count=320)
    b = replay(seed=123, count=320)
    assert a["action_sha256"] == b["action_sha256"]
    assert a["trace_sha256"] == b["trace_sha256"]
    assert a["trace"] == b["trace"]
    assert a["keydowns"] > 0
    assert a["keyups"] > 0


def test_000_seeds_do_not_share_precomputed_actions():
    first = replay(seed=123, count=180)
    other = replay(seed=456, count=180)
    again = replay(seed=123, count=180)
    assert first["action_sha256"] != other["action_sha256"]
    assert first["trace_sha256"] == again["trace_sha256"]


@pytest.mark.parametrize("same_hand", [True, False])
def test_000_different_episode_order_does_not_contaminate_initial_state(same_hand):
    a = replay(seed=91, count=120, same_hand=same_hand)
    replay(seed=92, count=120, same_hand=not same_hand)
    replay(seed=93, count=80, same_hand=same_hand)
    b = replay(seed=91, count=120, same_hand=same_hand)
    assert a["trace_sha256"] == b["trace_sha256"]


def test_000_cross_process_hash_seed_does_not_change_cpu_replay():
    cmd = [
        sys.executable,
        "scripts/a000_two_finger_replay.py",
        "--seed", "123", "--actions", "320",
    ]
    root = Path(__file__).resolve().parents[1]
    results = []
    for seed in ("0", "917"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["OMP_NUM_THREADS"] = "1"
        completed = subprocess.run(
            cmd, cwd=root, env=env, check=True,
            text=True, capture_output=True, timeout=90,
        )
        results.append(json.loads(completed.stdout))
    assert results[0] == results[1]
    assert results[0]["keydowns"] > 0


def test_000_input_and_report_contain_no_teacher_or_oracle():
    result = summary(replay(seed=24, count=64))
    assert set(result) == {
        "schema", "seed", "actions", "same_hand",
        "control_dt_s", "physics_dt_s", "keydowns", "keyups",
        "action_sha256", "trace_sha256",
    }
    assert "oracle" not in json.dumps(result).lower()
    assert "teacher" not in json.dumps(result).lower()


def test_000_rejects_empty_action_sequence():
    with pytest.raises(ValueError, match="positive"):
        deterministic_actions(0, 0)
