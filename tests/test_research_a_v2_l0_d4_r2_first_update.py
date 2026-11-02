"""D4: fixed first-update diagnosis, strict comparisons and archival guards."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a002v2_l0_d4_r2_first_update import (
    ARM, SEED, R2_RUN, D3_RUN, first_difference,
    first_update_difference, first_batch_row, select_r2,
)


def test_fixed_source_and_one_model():
    assert (ARM, SEED) == ("a0", 0)
    assert R2_RUN == 41312136897
    assert D3_RUN == 41369778494


def test_first_difference_reports_first_nested_exact_value():
    a = {"z": [1, 2], "a": {"x": 1.0}}
    b = {"z": [1, 3], "a": {"x": 1.0}}
    assert first_difference(a, a) is None
    assert first_difference(a, b) == {
        "path": "$.z[1]", "expected": 2, "actual": 3}
    assert first_difference({"x": 1.0}, {"x": 1.0000000001}) is not None


def test_tensor_comparison_is_exact_and_indexed():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = a.clone()
    assert first_difference(a, b) is None
    b[1, 0] = 3.25
    assert first_difference(a, b)["path"] == "$[1,0]"


def test_first_update_localizes_earliest_update():
    rows_a = [{"episode": 8 * i, "entropy": float(i)} for i in range(1, 33)]
    rows_b = [dict(x) for x in rows_a]
    rows_b[3]["entropy"] = 4.125
    rows_b[12]["entropy"] = 13.125
    first = first_update_difference(rows_a, rows_b)
    assert first["update"] == 4
    assert first["episodes_completed"] == 32
    assert first["path"] == "training.update4.entropy"
    with pytest.raises(ValueError, match="32 updates"):
        first_update_difference(rows_a, rows_b[:31])


def test_first_batch_record_preserves_original_fields():
    batch = [
        {"score": {"hits": 1, "extra_keydowns": 2, "reward_sum": 3.0},
         "interval_ms": 200, "start_s": .834567891}
        for _ in range(8)
    ]
    row = first_batch_row(batch, {"entropy": 1.3, "policy_loss": -.2})
    assert row["episode"] == 8
    assert row["train_hit_rate"] == 1 / 16
    assert row["train_extra_keydowns"] == 16
    assert row["mean_train_reward"] == 3.0
    assert row["starts_s"] == [.834568] * 8


def test_wrong_or_missing_archives_rejected():
    src = {
        "schema": "research-a-v2-l0-r2-penalty-comparison-v1",
        "r2_models": [{"arm": "a0", "seed": 0, "checkpoint_sha256": "a",
                       "evaluations": [{}, {}, {}]}],
    }
    d3 = {
        "schema": "research-a-v2-l0-d3-original-ppo-update-replay-v1",
        "models": [{"mode": "r2", "arm": "a0", "seed": 0,
                    "source_checkpoint_sha256": "b",
                    "evaluations": [{}, {}, {}]}],
    }
    with pytest.raises(ValueError, match="different original checkpoint"):
        select_r2(src, d3)
    d3["models"][0]["source_checkpoint_sha256"] = "a"
    assert select_r2(src, d3)[0]["checkpoint_sha256"] == "a"
    with pytest.raises(ValueError, match="wrong original R2 schema"):
        select_r2({**src, "schema": "wrong"}, d3)
