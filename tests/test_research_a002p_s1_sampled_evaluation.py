"""Contract tests for A002-P-S1 frozen-policy action sampling vs argmax."""
from __future__ import annotations

from pathlib import Path
import math
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))

from a002_exploratory_ppo import ActorCritic
from a002p_s1_sampled_evaluation import (
    REPETITIONS, SOURCE_RUN, fixed_sampling_seed, run_episode,
    aggregate, compare_original, load_checkpoint, ACTION_LABELS,
)
from dmdod.research.a002_timing_only_env import TimingOnlyTwoFingerEnv
from dmdod.research.a002p_reward_ablation import NoPositiveNearMissEnv


def test_fixed_protocol_constants_and_distinct_random_seeds():
    assert SOURCE_RUN == 41300077879
    assert REPETITIONS == 16
    seeds=[fixed_sampling_seed(i,r) for i in range(9) for r in range(REPETITIONS)]
    assert len(seeds)==len(set(seeds))==144
    assert seeds[0]==100000
    assert seeds[-1]==100815
    with pytest.raises(ValueError):
        fixed_sampling_seed(9,0)


def test_identical_stochastic_seed_produces_identical_physical_trace():
    torch.manual_seed(8)
    model=ActorCritic()
    one=run_episode(model,env_cls=TimingOnlyTwoFingerEnv,arm="a1",
                    interval_ms=105,start_s=.83,action_mode="sampled",
                    generator_seed=100000)
    two=run_episode(model,env_cls=TimingOnlyTwoFingerEnv,arm="a1",
                    interval_ms=105,start_s=.83,action_mode="sampled",
                    generator_seed=100000)
    assert one==two
    assert one["targets"]==16
    assert sum(one["action_counts"])==one["steps"]
    assert sum((one["physical_left"],one["physical_right"]))==one["physical_keydowns"]


@pytest.mark.parametrize("reward_env",(TimingOnlyTwoFingerEnv,NoPositiveNearMissEnv))
def test_greedy_and_sampling_are_scored_with_identical_physical_env(reward_env):
    torch.manual_seed(5)
    model=ActorCritic()
    greedy=run_episode(model,env_cls=reward_env,arm="a1",interval_ms=115,
                       start_s=.97,action_mode="argmax")
    sampled=run_episode(model,env_cls=reward_env,arm="a1",interval_ms=115,
                        start_s=.97,action_mode="sampled",generator_seed=100101)
    assert greedy["targets"]==sampled["targets"]==16
    assert greedy["action_mode"]=="argmax"
    assert sampled["action_mode"]=="sampled"
    for row in (greedy,sampled):
        assert row["physical_keydowns"]==row["hits"]+row["extra_keydowns"]
        assert row["hit_switch_count"]<=row["hit_transition_count"]
        assert row["physical_switch_count"]<=row["physical_transition_count"]


def test_aggregate_records_both_physical_key_usage_and_switch_fractions():
    template={
        "steps":10,"action_counts":[5,3,2,0],"probability_sums":[2.5]*4,
        "entropy_sum":10.0,"targets":4,"hits":2,"extra_keydowns":2,
        "physical_keydowns":4,"physical_left":2,"physical_right":2,
        "physical_both_used":True,"physical_switch_count":2,
        "physical_transition_count":3,"hit_switch_count":1,
        "hit_transition_count":1,"full_hit":False,
    }
    totals=aggregate([template,template])
    assert totals["episodes"]==2
    assert totals["hits"]==4 and totals["targets"]==8
    assert totals["extra_keydowns_per_episode"]==2
    assert totals["episodes_with_both_physical_keys"]==2
    assert totals["physical_switch_fraction"]==pytest.approx(2/3)
    assert totals["matched_hit_switch_fraction"]==1.0
    assert sum(totals["action_selection_fraction"].values())==pytest.approx(1.)
    assert sum(totals["mean_action_probabilities"].values())==pytest.approx(1.)


def test_original_episode_verification_is_exact_not_summary_only():
    source={
        "interval_ms":105,"start_s":.83,
        "hits":3,"targets":16,"extra_keydowns":4,
        "physical_keydowns":7,"matched_keys":"LLL","full_hit":False,
    }
    compare_original(source,source,0)
    bad={**source,"matched_keys":"RRR"}
    with pytest.raises(AssertionError, match="matched_keys"):
        compare_original(bad,source,0)


def test_bad_original_checkpoint_is_refused_before_loading(tmp_path):
    path=tmp_path/"checkpoint.pt"
    path.write_bytes(b"not a pytorch checkpoint")
    with pytest.raises(ValueError,match="hash mismatch"):
        load_checkpoint(path,"0"*64,"a1",0)


def test_a0_stochastic_physical_finger_mask_works():
    torch.manual_seed(17)
    model=ActorCritic()
    row=run_episode(model,env_cls=TimingOnlyTwoFingerEnv,arm="a0",
                    interval_ms=125,start_s=1.21,action_mode="sampled",
                    generator_seed=100815)
    assert row["physical_right"]==0
    assert row["physical_both_used"] is False
    assert set(ACTION_LABELS)=={"none","right","left","both"}
