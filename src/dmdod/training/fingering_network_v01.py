from __future__ import annotations

"""Small candidate-ranking Fingering Network v0.1.

The network sees only oracle-relative event timing, current physical Motor
state/availability and a proposed physical-axis allocation. Teacher costs and
counterfactual futures are labels only and never enter ``forward``.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from dmdod.motor.contact_16k import Coupled16ContactMotor, PHYSICAL_AXES, ThumbMode


FORMAT = "dmdod-fingering-network-v0.1-candidate-ranker"
MAX_EVENTS = 3
LOOKAHEAD_S = 0.155
GOOD_COST_TOLERANCE = 1.0
MOTOR_FEATURE_DIM = 56
EVENT_FEATURE_DIM = MAX_EVENTS * 3
CONTEXT_DIM = MOTOR_FEATURE_DIM + EVENT_FEATURE_DIM
CANDIDATE_DIM = len(PHYSICAL_AXES) * 4 + 1
HIDDEN = 96


@dataclass(frozen=True, slots=True)
class PatternSpec:
    pattern_id: str
    intervals_ms: tuple[float, ...]
    chord_sizes: tuple[int, ...]
    family: str

    def __post_init__(self) -> None:
        if not self.pattern_id or not self.intervals_ms or not self.chord_sizes:
            raise ValueError("empty fingering pattern")
        if len(self.chord_sizes) != len(self.intervals_ms) + 1:
            raise ValueError("chord_sizes must have one more item than intervals")
        if any(not isfinite(value) or value <= 0 for value in self.intervals_ms):
            raise ValueError("invalid pattern interval")
        if any(type(value) is not int or not 1 <= value <= 8 for value in self.chord_sizes):
            raise ValueError("invalid pattern chord size")

    @property
    def sha256(self) -> str:
        payload = json.dumps({
            "id": self.pattern_id, "intervals_ms": self.intervals_ms,
            "chord_sizes": self.chord_sizes, "family": self.family,
        }, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def event_times_s(self, *, first_s: float = 0.080) -> tuple[float, ...]:
        result = [first_s]
        for interval in self.intervals_ms:
            result.append(result[-1] + interval / 1000.0)
        return tuple(result)


@dataclass(frozen=True, slots=True)
class RankingExample:
    pattern_id: str
    pattern_sha256: str
    event_index: int
    context: torch.Tensor
    candidates: torch.Tensor
    teacher_costs: torch.Tensor
    eligible: torch.Tensor

    def __post_init__(self) -> None:
        count = int(self.candidates.shape[0]) if self.candidates.ndim == 2 else -1
        valid = (
            len(self.pattern_sha256) == 64
            and self.context.shape == (CONTEXT_DIM,)
            and self.candidates.shape == (count, CANDIDATE_DIM)
            and self.teacher_costs.shape == (count,)
            and self.eligible.shape == (count,)
            and self.eligible.dtype == torch.bool
            and count > 0 and bool(self.eligible.any())
            and bool(torch.isfinite(self.context).all())
            and bool(torch.isfinite(self.candidates).all())
            and bool(torch.isfinite(self.teacher_costs[self.eligible]).all())
        )
        if not valid:
            raise ValueError("invalid fingering ranking example")


class FingeringRanker(nn.Module):
    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        if hidden < 8:
            raise ValueError("hidden dimension too small")
        self.hidden = int(hidden)
        self.network = nn.Sequential(
            nn.Linear(CONTEXT_DIM + CANDIDATE_DIM, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, context: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if context.shape != (CONTEXT_DIM,) or candidates.ndim != 2 or candidates.shape[1] != CANDIDATE_DIM:
            raise ValueError("invalid causal fingering features")
        expanded = context.unsqueeze(0).expand(len(candidates), -1)
        return self.network(torch.cat((expanded, candidates), dim=1)).squeeze(1)


def _axis_availability(motor: Coupled16ContactMotor, occupied: set[str]) -> tuple[float, ...]:
    state = motor.motor.observe()
    return tuple(float(axis not in occupied and not state.pressed(axis)) for axis in PHYSICAL_AXES)


def encode_context(
    motor: Coupled16ContactMotor,
    upcoming: Sequence[object],
    now_s: float,
    occupied: set[str],
) -> torch.Tensor:
    """Encode only current physical state and oracle-relative future events."""

    if not isfinite(now_s):
        raise ValueError("nonfinite observation time")
    state = motor.motor.observe()
    actuation = motor.motor.keyboard.config.actuation_m
    position = [float(value / actuation) for value in state.positions_m]
    velocity = [float(value / max(actuation / 0.010, 1e-9)) for value in state.velocities_m_s]
    pressed = [float(value) for value in state.pressed_flags]
    available = list(_axis_availability(motor, occupied))
    modes = []
    for mode in motor._modes:
        modes.extend(float(mode is wanted) for wanted in (ThumbMode.OFF, ThumbMode.TIP, ThumbMode.FLAT))
    modes.extend(float(value) for value in motor._edge)
    physical = position + velocity + pressed + available + modes
    if len(physical) != MOTOR_FEATURE_DIM:
        raise RuntimeError("motor feature dimension drift")
    events = []
    for index in range(MAX_EVENTS):
        if index < len(upcoming):
            event = upcoming[index]
            relative = max(0.0, float(event.episode_time_s) - now_s)
            events.extend((min(relative / LOOKAHEAD_S, 2.0),
                           float(event.chord_size) / 8.0, 1.0))
        else:
            events.extend((0.0, 0.0, 0.0))
    output = torch.tensor(physical + events, dtype=torch.float32)
    if output.shape != (CONTEXT_DIM,) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("invalid fingering context encoding")
    return output


def encode_candidate(allocation: Sequence[object]) -> torch.Tensor:
    """Encode physical axes/modes, never logical keyboard bindings."""

    features = [0.0] * (len(PHYSICAL_AXES) * 4)
    contacts = 0
    mode_index = {"single": 0, "tip": 1, "flat": 2, "edge": 3}
    for finger in allocation:
        axis = str(finger.axis)
        mode = str(finger.mode)
        if axis not in PHYSICAL_AXES or mode not in mode_index:
            raise ValueError("invalid physical fingering allocation")
        slot = PHYSICAL_AXES.index(axis) * 4 + mode_index[mode]
        if features[slot]:
            raise ValueError("duplicate physical candidate component")
        features[slot] = 1.0
        contacts += int(finger.contact_count)
    output = torch.tensor(features + [contacts / 8.0], dtype=torch.float32)
    if output.shape != (CANDIDATE_DIM,):
        raise RuntimeError("candidate feature dimension drift")
    return output


def make_example(
    *, pattern: PatternSpec, event_index: int, context: torch.Tensor,
    allocations: Sequence[Sequence[object]], candidate_rows: Sequence[dict],
) -> RankingExample:
    if len(allocations) != len(candidate_rows):
        raise ValueError("teacher rows and allocations disagree")
    eligible = torch.tensor([bool(row["eligible"]) for row in candidate_rows], dtype=torch.bool)
    costs = torch.tensor([
        float(row["score"]) if row["eligible"] else float("inf")
        for row in candidate_rows
    ], dtype=torch.float32)
    return RankingExample(
        pattern.pattern_id, pattern.sha256, event_index, context,
        torch.stack([encode_candidate(value) for value in allocations]), costs, eligible,
    )


def ranking_loss(model: FingeringRanker, example: RankingExample) -> torch.Tensor:
    predicted = model(example.context, example.candidates)
    indices = torch.where(example.eligible)[0]
    scores = predicted[indices]
    costs = example.teacher_costs[indices]
    best = costs.min()
    good = costs <= best + GOOD_COST_TOLERANCE
    log_probability = F.log_softmax(-scores, dim=0)
    listwise = -log_probability[good].mean()

    pair_terms = []
    tie_terms = []
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            delta = float(costs[right] - costs[left])
            if abs(delta) <= GOOD_COST_TOLERANCE:
                tie_terms.append((scores[left] - scores[right]).abs())
            elif delta > 0:
                pair_terms.append(F.relu(0.1 - (scores[right] - scores[left])))
            else:
                pair_terms.append(F.relu(0.1 - (scores[left] - scores[right])))
    pairwise = torch.stack(pair_terms).mean() if pair_terms else scores.sum() * 0.0
    tie = torch.stack(tie_terms).mean() if tie_terms else scores.sum() * 0.0
    return listwise + pairwise + 0.05 * tie


def choose_candidate(
    model: FingeringRanker,
    context: torch.Tensor,
    allocations: Sequence[Sequence[object]],
    eligible: Sequence[bool],
) -> tuple[int | None, tuple[float | None, ...]]:
    """Select without teacher scores or future physical evaluation."""

    if len(allocations) != len(eligible) or not allocations:
        raise ValueError("invalid inference candidate set")
    candidates = torch.stack([encode_candidate(value) for value in allocations]).to(
        next(model.parameters()).device
    )
    with torch.no_grad():
        scores = model(context.to(candidates.device), candidates).cpu().tolist()
    masked = tuple(float(score) if valid else None for score, valid in zip(scores, eligible))
    valid_indices = [index for index, valid in enumerate(eligible) if valid]
    selected = min(valid_indices, key=lambda index: (scores[index], index)) if valid_indices else None
    return selected, masked


def train_fixed(
    model: FingeringRanker,
    examples: Sequence[RankingExample],
    *, epochs: int = 80, learning_rate: float = 5e-4, seed: int = 4101,
) -> list[float]:
    if not examples or epochs < 1:
        raise ValueError("empty fixed fingering training run")
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for _ in range(epochs):
        for index in torch.randperm(len(examples), generator=generator).tolist():
            example = examples[index]
            optimizer.zero_grad(set_to_none=True)
            loss = ranking_loss(model, example)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite fingering ranking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            history.append(float(loss.detach()))
    return history


@torch.no_grad()
def ranking_metrics(model: FingeringRanker, examples: Iterable[RankingExample]) -> dict:
    total = correct = good = different_good = 0
    regrets = []
    for example in examples:
        predictions = model(example.context, example.candidates)
        valid = torch.where(example.eligible)[0]
        chosen = int(valid[torch.argmin(predictions[valid])])
        costs = example.teacher_costs[valid]
        best_cost = float(costs.min())
        chosen_cost = float(example.teacher_costs[chosen])
        teacher_choice = int(valid[torch.argmin(costs)])
        total += 1
        correct += int(chosen == teacher_choice)
        is_good = chosen_cost <= best_cost + GOOD_COST_TOLERANCE
        good += int(is_good)
        different_good += int(is_good and chosen != teacher_choice)
        regrets.append(chosen_cost - best_cost)
    return {
        "examples": total,
        "exact_teacher_choice_rate": correct / total if total else None,
        "equivalent_good_rate": good / total if total else None,
        "different_but_equivalent_good": different_good,
        "mean_teacher_cost_regret": sum(regrets) / total if total else None,
        "max_teacher_cost_regret": max(regrets) if regrets else None,
    }
