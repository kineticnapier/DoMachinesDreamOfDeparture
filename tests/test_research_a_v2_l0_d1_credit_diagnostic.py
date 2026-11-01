"""Research A v2-L0-D1: no-training forensic replay contracts."""
from __future__ import annotations

from pathlib import Path
import sys
import math

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from a002_exploratory_ppo import ActorCritic
from a002v2_l0_low_speed_ppo import evaluate_episode
from a002v2_l0_d1_credit_diagnostic import (
    ORIGINAL_RUN, SOURCE_COMMIT, PHASE_LABELS, phase_name, replay_one,
    check_exact_original, aggregate_replays, archival_history,
    model_from_archive,
)


def test_frozen_source_and_phase_boundaries():
    assert ORIGINAL_RUN==41309837474
    assert SOURCE_COMMIT=="fa456d0e07e90fb740d969715f593f81c1ee48c3"
    assert len(PHASE_LABELS)==4
    assert phase_name(110)==PHASE_LABELS[0]
    assert phase_name(50)==PHASE_LABELS[1]
    assert phase_name(0)==PHASE_LABELS[2]
    assert phase_name(-25)==PHASE_LABELS[3]


@pytest.mark.parametrize("arm",("a0","a1"))
@pytest.mark.parametrize("mode",("argmax","sampled"))
def test_instrumentation_reproduces_original_motor_scoring(arm,mode):
    torch.manual_seed(7)
    model=ActorCritic()
    start=.835
    period=210
    seed=320000 if mode=="sampled" else None
    reference=evaluate_episode(
        arm=arm,interval_ms=period,start_s=start,mode=mode,
        model=model,seed=seed,
    )
    traced=replay_one(
        model,arm=arm,period=period,start=start,mode=mode,
        random_seed=seed,
    )
    check_exact_original(traced,reference,label=f"{arm}/{mode}")
    assert sum(traced["controls"])==traced["steps"]
    assert traced["physical_left"]+traced["physical_right"]==traced["physical_keydowns"]
    assert sum(p["steps"] for p in traced["by_time_to_next_target"].values())==traced["steps"]


def test_reproducible_lag_measurements():
    torch.manual_seed(1)
    model=ActorCritic()
    x=replay_one(model,arm="a1",period=250,start=1.290,mode="sampled",random_seed=320803)
    y=replay_one(model,arm="a1",period=250,start=1.290,mode="sampled",random_seed=320803)
    assert x==y
    assert all(v>=-1e-4 for v in x["onset_to_physical_down_lag_ms"])


def test_multiple_replays_aggregate_without_cross_episode_pairing():
    torch.manual_seed(2)
    model=ActorCritic()
    rows=[
        replay_one(model,arm="a1",period=240,start=1.075,
                   mode="sampled",random_seed=320100+i)
        for i in range(2)
    ]
    total=aggregate_replays(rows)
    assert total["episodes"]==2
    assert total["targets"]==32
    assert total["hits"]==sum(r["hits"] for r in rows)
    assert total["physical_left"]+total["physical_right"]==sum(
        r["physical_keydowns"] for r in rows
    )
    assert total["mean_action_entropy"]>=0
    assert sum(
        phase["steps"] for phase in total["by_time_to_next_target"].values()
    )==sum(r["steps"] for r in rows)


def test_original_per_episode_tampering_is_not_silently_accepted():
    torch.manual_seed(0)
    m=ActorCritic()
    real=replay_one(m,arm="a0",period=210,start=.835,mode="argmax",
                    random_seed=None)
    modified={**real,"physical_right":real["physical_right"]+1}
    with pytest.raises(AssertionError,match="physical_right"):
        check_exact_original(real,modified,label="tamper")


def test_intermediate_128_policy_weights_are_explicitly_unavailable(tmp_path):
    record={"arm":"a0","seed":0}
    with pytest.raises(ValueError,match="intermediate weights unavailable"):
        model_from_archive(record,tmp_path,128)


def test_nonexistent_archived_final_weights_are_rejected(tmp_path):
    record={"arm":"a0","seed":0,
            "checkpoint_file":"absent.pt",
            "checkpoint_sha256":"e"*64}
    with pytest.raises(FileNotFoundError):
        model_from_archive(record,tmp_path,256)
