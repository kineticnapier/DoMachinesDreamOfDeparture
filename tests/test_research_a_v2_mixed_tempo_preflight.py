"""No-training mixed-tempo evaluation contract."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from a002_exploratory_ppo import ActorCritic
from a002v2_mixed_tempo_preflight import (
    fixed_seed,rollout,summary,SCHEDULES,STARTS,REPETITIONS,
)


def test_frozen_three_schedules_and_eval_seeds():
    assert len(SCHEDULES)==3
    assert len(STARTS)==3
    assert REPETITIONS==4
    assert len({fixed_seed(i,j) for i in range(9) for j in range(4)})==36
    assert fixed_seed(0,0)==250000
    assert fixed_seed(8,3)==250803
    with pytest.raises(ValueError):
        fixed_seed(9,0)


def test_untrained_random_seed_yields_same_physical_replay():
    row1=rollout(schedule=SCHEDULES[1],start_s=STARTS[1],arm="a1",
                reward="r0_original",mode="random",generator_seed=250101)
    row2=rollout(schedule=SCHEDULES[1],start_s=STARTS[1],arm="a1",
                reward="r0_original",mode="random",generator_seed=250101)
    assert row1==row2
    assert row1["targets"]==24


def test_mixed_segment_metrics_and_both_finger_masks():
    torch.manual_seed(0)
    model=ActorCritic()
    rows=[]
    for disable in (None,"left","right"):
        row=rollout(schedule=SCHEDULES[0],start_s=STARTS[0],arm="a1",
                    reward="r1_no_positive_near_miss",mode="sampled",
                    model=model,disabled_side=disable,generator_seed=250000)
        assert sum(s["hits"] for s in row["segments"])==row["hits"]
        assert sum(s["misses"] for s in row["segments"])==row["misses"]
        assert row["physical_keydowns"]==row["hits"]+row["extra_keydowns"]
        if disable=="left":
            assert row["left_physical_keydowns"]==0
        if disable=="right":
            assert row["right_physical_keydowns"]==0
        rows.append(row)
    assert summary(rows)["targets"]==72


def test_unchanged_random_baseline_a0_cannot_generate_right_downs():
    row=rollout(schedule=SCHEDULES[2],start_s=STARTS[2],arm="a0",
                reward="r0_original",mode="random",generator_seed=250803)
    assert row["right_physical_keydowns"]==0
    assert row["targets"]==24
