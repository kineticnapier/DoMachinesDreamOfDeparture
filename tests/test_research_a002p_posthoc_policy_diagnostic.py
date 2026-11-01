"""Post-hoc diagnostic contracts; original checkpoints not required for tests."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from a002_exploratory_ppo import ActorCritic, evaluate, EVAL_INTERVALS_MS, EVAL_STARTS_S
from a002p_posthoc_policy_diagnostic import (
    ACTION_LABELS, EXPECTED_MODEL_SHA256, SOURCE_RUN, aggregate,
    bin_abs_ms, checked_checkpoint, summarize, traced_episode,
)


def test_pinned_source_and_all_four_model_hashes():
    assert SOURCE_RUN == 41296142433
    assert set(EXPECTED_MODEL_SHA256) == {("a0",0),("a0",1),("a1",0),("a1",1)}
    assert len(set(EXPECTED_MODEL_SHA256.values())) == 4


def test_instrumented_replay_keeps_original_hit_and_extra_count():
    torch.manual_seed(8)
    model = ActorCritic()
    model.eval()
    with torch.no_grad():
        source = evaluate(model, arm="a1")
        rows = [
            traced_episode(model, arm="a1", interval_ms=period, start_s=start)
            for period in EVAL_INTERVALS_MS for start in EVAL_STARTS_S
        ]
    replay = aggregate(rows)
    assert replay["hits"] == source["hits"]
    assert replay["targets"] == source["targets"]
    assert replay["extra_keydowns"] == source["extra_keydowns"]
    assert replay["physical_keydowns"] == sum(
        r["physical_keydowns"] for r in source["rows"]
    )
    for a,b in zip(rows,source["rows"]):
        assert a["matched_keys"] == b["matched_keys"]
    assert abs(sum(replay["average_softmax_probability"].values())-1.) < 1e-6
    assert abs(sum(replay["action_selection_fraction"].values())-1.) < 1e-6
    assert replay["mean_actor_entropy_nats"] <= math.log(4.)+1e-7
    assert sum(replay["physical_down_by_key"].values()) == replay["physical_keydowns"]


def test_a0_physical_mask_and_action_labels_reported_separately():
    torch.manual_seed(4)
    model = ActorCritic()
    sample = traced_episode(model, arm="a0", interval_ms=105, start_s=.83)
    assert set(sample["average_action_probability"]) == set(ACTION_LABELS)
    assert set(sample["physical_down_by_key"]) <= {"left"}
    assert len(sample["first_10_keydowns"]) <= 10


def test_histogram_boundaries_and_abs_values():
    hist = bin_abs_ms([-3, 5, 10, 20, 40, 80, 110])
    assert hist == {
        "0-5":1, "5-10":1, "10-20":1, "20-40":1,"40-80":1,"80+":2
    }
    assert summarize([])["median"] is None
    assert summarize([-10, 20])["mean_abs"] == 15


def test_checksum_mismatch_is_rejected_before_deserialization(tmp_path):
    arbitrary = tmp_path / "a002p-a0-seed0-ep96.pt"
    arbitrary.write_bytes(b"garbage-not-a-checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checked_checkpoint(arbitrary,"a0",0)


def test_a_saved_genuine_state_dict_can_be_loaded_only_when_hash_matches(tmp_path, monkeypatch):
    torch.manual_seed(0)
    model=ActorCritic()
    checkpoint = tmp_path/"a002p-a0-seed0-ep96.pt"
    torch.save({
        "format":"research-a002p-pilot-v1","arm":"a0","seed":0,
        "episodes":96,"policy":model.state_dict()
    }, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setitem(EXPECTED_MODEL_SHA256,("a0",0),digest)
    recovered = checked_checkpoint(checkpoint,"a0",0)
    assert all(torch.equal(p,q) for p,q in zip(model.parameters(), recovered.parameters()))
