from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dmdod.motor.contact_16k import Coupled16ContactMotor
from dmdod.training import fingering_network_v01 as network


class Finger:
    def __init__(self, axis, mode="single", contacts=1):
        self.axis, self.mode, self.contact_count = axis, mode, contacts


def pattern(name="p"):
    return network.PatternSpec(name, (40.0, 30.0), (1, 1, 1), "rolling")


def test_context_is_physical_and_oracle_relative_without_teacher_scores():
    motor = Coupled16ContactMotor()
    events = [SimpleNamespace(episode_time_s=.10, chord_size=1),
              SimpleNamespace(episode_time_s=.14, chord_size=2)]
    encoded = network.encode_context(motor, events, .06, {"left_1"})
    assert encoded.shape == (network.CONTEXT_DIM,)
    assert network.CONTEXT_DIM == network.MOTOR_FEATURE_DIM + 9
    # Availability is after position, velocity and pressed blocks.
    availability = encoded[36:48]
    assert availability[network.PHYSICAL_AXES.index("left_1")] == 0
    assert availability.sum() == 11


def test_candidate_encoding_uses_axis_and_physical_contact_mode():
    single = network.encode_candidate((Finger("left_1"),))
    flat = network.encode_candidate((Finger("right_5", "flat", 2),))
    assert single.shape == flat.shape == (network.CANDIDATE_DIM,)
    assert single[-1] == pytest.approx(1 / 8)
    assert flat[-1] == pytest.approx(2 / 8)
    assert not torch.equal(single, flat)


def test_equal_teacher_candidates_are_not_forced_apart():
    context = torch.zeros(network.CONTEXT_DIM)
    candidates = torch.stack([
        network.encode_candidate((Finger("left_1"),)),
        network.encode_candidate((Finger("right_1"),)),
        network.encode_candidate((Finger("left_2"),)),
    ])
    example = network.RankingExample(
        "p", pattern().sha256, 0, context, candidates,
        torch.tensor([0.0, 0.5, 20.0]), torch.tensor([True, True, True]),
    )
    model = network.FingeringRanker(hidden=16)
    loss = network.ranking_loss(model, example)
    assert torch.isfinite(loss) and float(loss.detach()) > 0


def test_invalid_candidate_is_masked_at_inference():
    model = network.FingeringRanker(hidden=16)
    allocations = ((Finger("left_1"),), (Finger("right_1"),))
    chosen, scores = network.choose_candidate(
        model, torch.zeros(network.CONTEXT_DIM), allocations, (False, True)
    )
    assert chosen == 1 and scores[0] is None and scores[1] is not None


def test_pattern_hash_and_split_identity_cover_whole_sequence():
    first = pattern("train-roll")
    second = pattern("eval-roll")
    assert first.sha256 != second.sha256
    assert first.event_times_s() == pytest.approx((.08, .12, .15))
