"""D2: frozen-policy PPO advantage proxy, no training and exact R1/R2 replay."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from a002_exploratory_ppo import ActorCritic
from a002v2_l0_low_speed_ppo import evaluate_episode
from a002v2_l0_d2_onset_advantage import (
    replay_one, verify_replay, normalized_advantages, aggregate,
    source_models, STEP_FIELDS, SOURCE_R1_RUN, SOURCE_R2_RUN,
)


def test_frozen_source_and_replay_fields():
    assert SOURCE_R1_RUN==41309837474
    assert SOURCE_R2_RUN==41312136897
    assert "physical_trace" in STEP_FIELDS
    assert "controls" in STEP_FIELDS


@pytest.mark.parametrize("arm",("a0","a1"))
def test_original_r1_policy_sampled_trace_is_exactly_reproduced(arm):
    torch.manual_seed(3)
    model=ActorCritic()
    before=[p.detach().clone() for p in model.parameters()]
    ref=evaluate_episode(
        arm=arm,interval_ms=210,start_s=.835,mode="sampled",
        model=model,seed=320000,
    )
    got=replay_one(
        model,arm=arm,mode="r1",interval_ms=210,start_s=.835,seed=320000,
    )
    verify_replay(got,ref,"r1 replay")
    assert all(torch.equal(x,y) for x,y in zip(before,model.parameters()))
    assert len(got["step_data"])==got["steps"]
    assert len(got["event_data"])==got["physical_keydowns"]


@pytest.mark.parametrize("arm",("a0","a1"))
def test_r2_penalty_does_not_change_physical_event_sequence(arm):
    torch.manual_seed(7)
    model=ActorCritic()
    before=[p.detach().clone() for p in model.parameters()]
    a=replay_one(model,arm=arm,mode="r1",interval_ms=240,start_s=1.075,seed=320400)
    b=replay_one(model,arm=arm,mode="r2",interval_ms=240,start_s=1.075,seed=320400)
    for field in ("hits","misses","targets","extra_keydowns","physical_keydowns",
                  "full_hit","matched_keys","physical_left","physical_right",
                  "controls","steps"):
        assert a[field]==b[field],field
    assert [(x["key"],x["time_s"],x["hit"]) for x in a["physical_trace"]]==[
        (x["key"],x["time_s"],x["hit"]) for x in b["physical_trace"]
    ]
    assert b["reward_sum"]<=a["reward_sum"]+1e-7
    assert all(torch.equal(x,y) for x,y in zip(before,model.parameters()))


def test_normalized_advantages_are_finite_and_onsets_point_to_real_steps():
    torch.manual_seed(4)
    model=ActorCritic()
    rows=[
        replay_one(model,arm="a1",mode="r2",interval_ms=250,
                   start_s=1.290,seed=320800+i)
        for i in range(2)
    ]
    normalized_advantages(rows)
    advantages=[s["advantage_normalized"] for r in rows for s in r["step_data"]]
    assert len(advantages)>100
    assert all(math.isfinite(x) for x in advantages)
    assert abs(sum(advantages)/len(advantages))<1e-5
    for row in rows:
        for e in row["event_data"]:
            idx=e["onset_step"]
            if idx is None:
                assert e["onset_advantage_normalized"] is None
            else:
                assert 0<=idx<len(row["step_data"])
                assert e["onset_to_keydown_ms"]>=-1e-4
                assert e["onset_advantage_normalized"]==row["step_data"][idx]["advantage_normalized"]
    result=aggregate(rows)
    assert result["targets"]==32
    assert result["hits"]+result["extras"]==sum(r["physical_keydowns"] for r in rows)
    assert sum(x["count"] for x in result["all_action_advantages"].values())==len(advantages)


def test_replay_rejects_archival_event_tampering():
    torch.manual_seed(1)
    model=ActorCritic()
    x=replay_one(model,arm="a0",mode="r1",interval_ms=210,start_s=.835,seed=320000)
    changed={**x,"physical_trace":x["physical_trace"]+[{"key":"left"}]}
    with pytest.raises(AssertionError,match="physical_trace"):
        verify_replay(x,changed,"tampered")


def test_no_archive_is_not_treated_as_zero_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_models(tmp_path,"r1")
