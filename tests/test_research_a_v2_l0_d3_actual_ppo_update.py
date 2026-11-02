"""D3: actual PPO update instrumentation must preserve the original trainer."""
from __future__ import annotations

from pathlib import Path
import random
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import a002v2_l0_low_speed_ppo as l0
from a002v2_l0_d3_actual_ppo_update import (
    TraceR1,TraceR2,Recorder,instrument,describe,summarize,
    compare_archived,R1_RUN,R2_RUN,
)
from a002v2_l0_d2_onset_advantage import source_models


def test_source_runs_and_frozen_ppo_conditions():
    assert R1_RUN==41309837474
    assert R2_RUN==41312136897
    assert l0.ARMS==("a0","a1")
    assert l0.SEEDS==(0,1)
    assert l0.TRAIN_INTERVALS_MS==(200,260)
    assert l0.EPISODES==256
    assert l0.BATCH_EPISODES==8
    assert l0.CHECKPOINTS==(0,128,256)


def test_empty_event_metrics_are_explicit():
    x=describe([])
    assert x["count"]==0
    assert x["both_positive_fraction"] is None
    assert summarize([])["unmatched_onsets"]==0


def test_positive_advantage_and_actual_logp_are_counted_separately():
    events=[
        {"hit":True,"key":"left","onset_step":0,
         "advantage_normalized":2.,"delta_logp":-.01,
         "onset_to_keydown_ms":30.,"onset_reused":False},
        {"hit":False,"key":"right","onset_step":1,
         "advantage_normalized":1.,"delta_logp":.01,
         "onset_to_keydown_ms":35.,"onset_reused":False},
        {"hit":False,"key":"right","onset_step":2,
         "advantage_normalized":-1.,"delta_logp":.02,
         "onset_to_keydown_ms":40.,"onset_reused":False},
    ]
    x=summarize(events)
    assert x["hit"]["positive_adv"]==1
    assert x["hit"]["increased_logp"]==0
    assert x["extra"]["positive_adv"]==1
    assert x["extra"]["increased_logp"]==2
    assert x["extra"]["both_positive"]==1
    assert x["extra"]["both_positive_fraction"]==.5


@pytest.mark.parametrize("arm",("a0","a1"))
@pytest.mark.parametrize("mode,cls",(("r1",TraceR1),("r2",TraceR2)))
def test_original_eight_episode_ppo_update_is_unchanged(arm,mode,cls):
    # Compare an uninstrumented and instrumented batch, from identical
    # initialization and RNG, through the *real* original update function.
    from a002v2_l0_r2_penalty_env import PenalizedLoggedLowSpeedEnv
    from unittest.mock import patch

    original_env=(l0.LoggedLowSpeedEnv if mode=="r1"
                  else PenalizedLoggedLowSpeedEnv)
    torch.set_num_threads(1)
    torch.manual_seed(9)
    baseline=l0.ActorCritic()
    baseline_optim=torch.optim.Adam(baseline.parameters(),lr=.0005)
    with patch.object(l0,"LoggedLowSpeedEnv",original_env):
        batch_a=[l0.train_episode(baseline,arm=arm,rng=local_rng)
                 for local_rng in []]  # no-op; RNG reset below
    rng_a=random.Random(9)
    with patch.object(l0,"LoggedLowSpeedEnv",original_env):
        batch_a=[l0.train_episode(baseline,arm=arm,rng=rng_a)
                 for _ in range(8)]
        losses_a=l0.update_ppo(baseline,baseline_optim,batch_a)

    torch.manual_seed(9)
    traced=l0.ActorCritic()
    traced_optim=torch.optim.Adam(traced.parameters(),lr=.0005)
    rng_b=random.Random(9)
    recorder=Recorder(mode,arm,9,cls)
    with instrument(recorder):
        batch_b=[l0.train_episode(traced,arm=arm,rng=rng_b)
                 for _ in range(8)]
        losses_b=l0.update_ppo(traced,traced_optim,batch_b)
    assert losses_a==losses_b
    for a,b in zip(baseline.parameters(),traced.parameters()):
        assert torch.equal(a,b)
    assert len(recorder.updates)==1
    update=recorder.updates[0]
    assert update["episodes_completed"]==8
    assert len(update["actor_gradients_by_epoch"])==2
    assert all(g["actor_gradient_l2"]>0 for g in update["actor_gradients_by_epoch"])
    assert all(len(g["actor_weight_gradient"])==128
               and len(g["actor_bias_gradient"])==4
               for g in update["actor_gradients_by_epoch"])
    assert update["training_hits"]==sum(x["score"]["hits"] for x in batch_a)
    assert update["training_extras"]==sum(x["score"]["extra_keydowns"] for x in batch_a)
    assert len(recorder.events)==sum(x["score"]["physical_keydowns"] for x in batch_a)
    assert update["unmatched_onsets"]==0
    assert update["reused_onsets"]==0
    for event in recorder.events:
        assert event["onset_step"] is not None
        assert event["onset_to_keydown_ms"]>=0
        assert isinstance(event["delta_logp"],float)
    assert not cls.capture_enabled
    assert not cls.completed


def test_reproduction_rejects_changed_original_training():
    entry={
        "initial_state_sha256":"a","checkpoint_sha256":"b",
        "training":[{}]*32,"evaluations":[{}]*3,
    }
    compare_archived(entry,dict(entry),"exact")
    with pytest.raises(AssertionError,match="32 original PPO"):
        compare_archived(entry,{**entry,"training":[{}]*31},"bad")
    with pytest.raises(AssertionError,match="final checkpoint"):
        compare_archived(entry,{**entry,"checkpoint_sha256":"other"},"bad")


def test_missing_original_archive_is_not_assumed_success(tmp_path):
    with pytest.raises(FileNotFoundError):
        source_models(tmp_path,"r1")
