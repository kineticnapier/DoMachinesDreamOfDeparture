from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v090_round_accel as accel


class _Sequence:
    def __init__(self, frames: int):
        self.frames = frames


class _Stable:
    def __init__(self, weights):
        self.loss_weights = torch.tensor(weights, dtype=torch.float32)
        self.sequence = _Sequence(len(weights))


def test_hybrid_eval_flattens_when_grouping_would_leave_workers_idle():
    assert accel._should_flatten({0.5: object()}, [1, 2, 3], 6)
    assert accel._should_flatten({0.5: object(), 0.25: object()}, [1, 2], 6)


def test_hybrid_eval_keeps_grouping_when_states_fill_worker_pool():
    states = {float(index): object() for index in range(6)}
    assert not accel._should_flatten(states, [1, 2, 3], 6)
    assert not accel._should_flatten({0.5: object()}, [1], 6)
    assert not accel._should_flatten({0.5: object()}, [1, 2], 1)


def test_sequence_weight_matches_chunk_denominator():
    stable = _Stable([0.25, 0.75, 0.0, 0.0, 2.0])
    # chunk=2 => max(1.0,1) + max(0,1) + max(2,1) = 4
    assert accel._sequence_weight(stable, 2) == 4.0
