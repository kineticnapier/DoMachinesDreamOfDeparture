"""Unit checks for preregistered D7 comparison and stop conditions."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a002v2_l0_d7_first_ppo_divergence import (
    MAX_UPDATES, classify, physical_summary, training_row, validate_archives,
)
from a002v2_l0_d4_r2_first_update import first_difference


def sample_archives():
    rows = [{"episode": i * 8, "entropy": float(i)} for i in range(1, 33)]
    evals = [{"episode": i} for i in (0, 128, 256)]
    original = {
        "arm": "a0", "seed": 0, "checkpoint_sha256": "original",
        "training": rows, "evaluations": evals,
    }
    d3 = {
        "mode": "r1", "arm": "a0", "seed": 0,
        "original_reproduced": True,
        "source_checkpoint_sha256": "original",
        "replay_checkpoint_sha256": "original",
        "training_log": list(rows), "evaluations": list(evals),
        "all_updates": [{"update": i, "actor_gradients_by_epoch": [{}, {}]}
                        for i in range(1, 33)],
        "all_training_keydown_events": [{"update": 1, "hit": False},
                                        {"update": 2, "hit": True}],
    }
    return original, d3


def test_valid_d3_r1_archive():
    a, b = sample_archives()
    grouped = validate_archives(a, b)
    assert len(grouped[1]) == 1
    assert len(grouped[2]) == 1
    assert grouped[3] == []


def test_wrong_checkpoint_fails_before_training():
    a, b = sample_archives()
    b["replay_checkpoint_sha256"] = "different"
    with pytest.raises(ValueError, match="checkpoint"):
        validate_archives(a, b)


def test_missing_gradient_fails_before_training():
    a, b = sample_archives()
    b["all_updates"][4]["actor_gradients_by_epoch"] = [{}]
    with pytest.raises(ValueError, match="gradients"):
        validate_archives(a, b)


def test_changed_training_row_fails_before_training():
    a, b = sample_archives()
    b["training_log"][0] = {"episode": 8, "entropy": 0.0}
    with pytest.raises(ValueError, match="training"):
        validate_archives(a, b)


def test_classify_initial_difference():
    assert classify({"path": "stage0"}, None, 0) == "initial_evaluation_diverged"


def test_classify_first_update_difference():
    assert classify(None, {"update": 4}, 4) == "first_update_divergence_identified"


def test_classify_all_16_exact():
    assert MAX_UPDATES == 16
    assert classify(None, None, 16) == "no_divergence_through_128"
    assert classify(None, None, 15) == "invalid_diagnostic"


def test_first_difference_is_exact():
    assert first_difference({"a": [1.0]}, {"a": [1.0]}) is None
    diff = first_difference({"a": [1.0]}, {"a": [1.0 + 1e-10]})
    assert diff["path"] == "$.a[0]"


def test_training_row_matches_original_formula():
    batch = [{
        "score": {"hits": 1, "extra_keydowns": 2, "reward_sum": 3.0},
        "interval_ms": 200, "start_s": 0.812345678,
    } for _ in range(8)]
    row = training_row(batch, {"entropy": 1.5}, 3)
    assert row["episode"] == 24
    assert row["train_hit_rate"] == 8 / 128
    assert row["train_extra_keydowns"] == 16
    assert row["mean_train_reward"] == 3.0
    assert row["starts_s"] == [0.812346] * 8


def test_physical_accounting():
    events = [
        {"hit": True, "onset_step": 1, "onset_reused": False},
        {"hit": False, "onset_step": None, "onset_reused": False},
        {"hit": False, "onset_step": 2, "onset_reused": True},
    ]
    assert physical_summary(events) == {
        "physical_keydowns": 3, "hits": 1, "extras": 2,
        "unmatched_onsets": 1, "reused_onsets": 1,
    }
