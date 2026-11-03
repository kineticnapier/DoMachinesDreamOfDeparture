"""D33 first PPO directional projection: arithmetic and provenance controls."""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a002v2_l0_d33_step413_gradient_projection as d33


def vec(*values):
    return torch.tensor(values, dtype=torch.float32)


def test_dot_by_group_signed_sgd_direction():
    names = ["shared.weight", "actor.bias", "critic.bias"]
    out = d33.dot_by_group(names, [vec(1, 2), vec(3), vec(0)],
                           [vec(4, -1), vec(-2), vec(7)])
    assert out["by_parameter_group"]["shared"] == -2.0
    assert out["by_parameter_group"]["actor"] == 6.0
    assert out["by_parameter_group"]["critic"] == 0.0
    assert out["total"] == 4.0


def test_negative_direction_is_valid_measurement():
    out = d33.dot_by_group(["actor.bias"], [vec(1)], [vec(2)])
    assert out["total"] == -2.0


def test_zero_direction_is_valid_measurement():
    out = d33.dot_by_group(["actor.bias"], [vec(0)], [vec(2)])
    assert out["total"] == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_gradient(bad):
    with pytest.raises(ValueError):
        d33.dot_by_group(["actor.bias"], [vec(1)], [vec(bad)])


def test_reject_mismatched_names_or_shapes():
    with pytest.raises(ValueError):
        d33.dot_by_group(["actor.bias", "shared.weight"], [vec(1)], [vec(2)])
    with pytest.raises(ValueError):
        d33.dot_by_group(["actor.bias"], [vec(1, 2)], [vec(3)])
    with pytest.raises(ValueError):
        d33.dot_by_group(["new_layer.weight"], [vec(1)], [vec(2)])


def sample_projection(values):
    return {"by_parameter_group": dict(zip(d33.GROUPS, values)),
            "total": sum(values)}


def test_reconstruction_accepts_positive_negative_and_zero():
    row = {"none": (1.0, 2.0, 0.0), "right": (-2.0, 1.0, 0.0),
           "left": (3.0, -1.0, 0.0), "both": (0.0, 0.0, 0.0)}
    d = {n: sample_projection(v) for n, v in row.items()}
    d["policy"] = sample_projection((2.0, 2.0, 0.0))
    d["auxiliary"] = sample_projection((-1.0, 3.0, 0.0))
    d["full"] = sample_projection((1.0, 5.0, 0.0))
    assert d33.validate_reconstruction(d) == {"policy": 0.0, "full": 0.0}


def test_reconstruction_rejects_bad_policy():
    d = {n: sample_projection((0.0, 0.0, 0.0))
         for n in d33.NAMES + ("policy", "auxiliary", "full")}
    d["policy"] = sample_projection((0.01, 0.0, 0.0))
    with pytest.raises(ValueError):
        d33.validate_reconstruction(d)


def test_reconstruction_rejects_bad_full():
    d = {n: sample_projection((0.0, 0.0, 0.0))
         for n in d33.NAMES + ("policy", "auxiliary", "full")}
    d["full"] = sample_projection((0.01, 0.0, 0.0))
    with pytest.raises(ValueError):
        d33.validate_reconstruction(d)


def synthetic_d17():
    return {"schema": "research-a-v2-l0-d17-action-gradient-v1",
            "classification": "valid_action_gradient_decomposition",
            "initial_state_sha256": "initial", "batch_sha256": "batch",
            "arm": "a0", "seed": 0, "targets": 128, "hits": 0,
            "extras": 9, "physical_keydowns": 9,
            "counts": list(d33.COUNTS),
            "by_action": {n: {"index": i, "count": d33.COUNTS[i],
                             "l2_norm": 0.0}
                          for i, n in enumerate(d33.NAMES)},
            "policy_gradient_l2": 0.0, "full_ppo_gradient_l2": 0.0}


def test_d17_archive_validates_zero_gradient():
    z = [vec(0.0)]
    d33.validate_archived_d17(synthetic_d17(), "initial", "batch",
                               list(d33.COUNTS), [z] * 4, z, z)


@pytest.mark.parametrize("change", [
    lambda s: s.update({"seed": 99}),
    lambda s: s.update({"batch_sha256": "other"}),
    lambda s: s.update({"counts": [1204, 1170, 783, 642]}),
    lambda s: s["by_action"]["left"].update({"l2_norm": 1.0}),
    lambda s: s["by_action"]["both"].update({"index": 1}),
    lambda s: s.update({"classification": "invalid"}),
])
def test_d17_archive_rejects_corruption(change):
    row = synthetic_d17()
    change(row)
    z = [vec(0.0)]
    with pytest.raises(ValueError):
        d33.validate_archived_d17(row, "initial", "batch",
                                   list(d33.COUNTS), [z] * 4, z, z)
