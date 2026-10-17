from __future__ import annotations

"""Causal multi-event Timing Student for exact-Autoplay supervision.

This module is deliberately independent from the frozen Stage A Motor gate.
The neural network accepts only a forward prefix of the unchanged 263D HUD.
Exact target timestamps are used solely by label construction and scoring.
"""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite

import torch
from torch import nn
from torch.nn import functional as F

from .prestige1 import INPUT_DIM
from .prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "prestige1-autoplay-multievent-timing-student-v1"
CONTROL_DT_S = 0.010
HORIZON_S = 0.080
MAX_EVENTS = 16
TIME_EPS_S = 1e-9


@dataclass(frozen=True, slots=True)
class FutureEventLabels:
    """Evaluator-only ordered future event labels for one HUD instant."""

    offsets_s: tuple[float, ...]
    chord_sizes: tuple[int, ...]
    overflow: bool

    def __post_init__(self) -> None:
        if len(self.offsets_s) != len(self.chord_sizes) or len(self.offsets_s) > MAX_EVENTS:
            raise ValueError("invalid future event label cardinality")
        if any(not isfinite(x) or not 0.0 < x <= HORIZON_S + TIME_EPS_S for x in self.offsets_s):
            raise ValueError("future event offset outside 0..80ms")
        if any(b <= a for a, b in zip(self.offsets_s, self.offsets_s[1:])):
            raise ValueError("future event labels must be strictly ordered")
        if any(type(size) is not int or not 1 <= size <= 8 for size in self.chord_sizes):
            raise ValueError("invalid future chord size")


def future_labels(
    teacher: AutoplayTimingTeacher,
    now_s: float,
    *,
    horizon_s: float = HORIZON_S,
    max_events: int = MAX_EVENTS,
) -> FutureEventLabels:
    """Build exact, unrounded labels without exposing them to the Student.

    Events at or before ``now_s`` have no usable positive lead and are not
    targets.  Simultaneous inputs were already grouped as one chord by the
    exact Autoplay teacher.
    """

    if not isfinite(now_s) or not 0.0 < horizon_s <= HORIZON_S or not 1 <= max_events <= MAX_EVENTS:
        raise ValueError("invalid future-label query")
    times = tuple(event.episode_time_s for event in teacher.events)
    start = bisect_right(times, now_s + TIME_EPS_S)
    candidates = [
        event
        for event in teacher.events[start:]
        if event.episode_time_s - now_s <= horizon_s + TIME_EPS_S
    ]
    kept = candidates[:max_events]
    return FutureEventLabels(
        offsets_s=tuple(event.episode_time_s - now_s for event in kept),
        chord_sizes=tuple(event.chord_size for event in kept),
        overflow=len(candidates) > max_events,
    )


@dataclass(frozen=True)
class MultiEventTimingSequence:
    """One chart trajectory; only ``visible`` can enter model.forward."""

    chart_sha256: str
    visible: torch.Tensor  # [T,263], current/past public HUD only
    offsets_s: torch.Tensor  # [T,K], evaluator/training labels only
    present: torch.Tensor  # [T,K], evaluator/training labels only
    chord_sizes: torch.Tensor  # [T,K], evaluator labels only
    overflow: torch.Tensor  # [T], evaluator diagnostic only
    exact_event_times_s: tuple[float, ...]  # posthoc evaluator only
    special_event_tags: tuple[tuple[str, ...], ...]  # posthoc cohort labels only

    def __post_init__(self) -> None:
        t = int(self.visible.shape[0])
        k = int(self.offsets_s.shape[1]) if self.offsets_s.ndim == 2 else -1
        valid = (
            len(self.chart_sha256) == 64
            and t > 0
            and self.visible.shape == (t, INPUT_DIM)
            and self.offsets_s.shape == (t, MAX_EVENTS)
            and self.present.shape == (t, MAX_EVENTS)
            and self.chord_sizes.shape == (t, MAX_EVENTS)
            and self.overflow.shape == (t,)
            and k == MAX_EVENTS
            and self.present.dtype == torch.bool
            and self.overflow.dtype == torch.bool
            and len(self.special_event_tags) == len(self.exact_event_times_s)
            and bool(torch.isfinite(self.visible).all())
            and bool(torch.isfinite(self.offsets_s).all())
            and not bool((self.offsets_s < 0).any())
            and not bool((self.offsets_s > HORIZON_S + 1e-6).any())
        )
        if not valid:
            raise ValueError("invalid causal multi-event Timing sequence")
        counts = self.present.sum(dim=1)
        for row, count in enumerate(counts.tolist()):
            count = int(count)
            if bool(self.present[row, count:].any()):
                raise ValueError("event presence slots must form a prefix")
            if count > 1 and not bool(torch.all(
                self.offsets_s[row, 1:count] > self.offsets_s[row, :count - 1]
            )):
                raise ValueError("present event offsets must be strictly ordered")
            if count and not bool(torch.all(
                (self.chord_sizes[row, :count] >= 1) & (self.chord_sizes[row, :count] <= 8)
            )):
                raise ValueError("invalid present chord size")


def pack_label_rows(rows: list[FutureEventLabels], *, device: torch.device | str = "cpu"):
    """Pack evaluator labels; padding values never receive timing loss."""

    n = len(rows)
    offsets = torch.zeros((n, MAX_EVENTS), dtype=torch.float32, device=device)
    present = torch.zeros((n, MAX_EVENTS), dtype=torch.bool, device=device)
    chords = torch.zeros((n, MAX_EVENTS), dtype=torch.int64, device=device)
    overflow = torch.zeros(n, dtype=torch.bool, device=device)
    for row, label in enumerate(rows):
        count = len(label.offsets_s)
        if count:
            offsets[row, :count] = torch.tensor(label.offsets_s, dtype=torch.float32, device=device)
            present[row, :count] = True
            chords[row, :count] = torch.tensor(label.chord_sizes, dtype=torch.int64, device=device)
        overflow[row] = label.overflow
    return offsets, present, chords, overflow


class MultiEventTimingStudent(nn.Module):
    """Forward-only GRU with ordered event times and per-slot confidence."""

    def __init__(self, hidden: int = 128, *, max_events: int = MAX_EVENTS):
        super().__init__()
        if hidden < 1 or max_events != MAX_EVENTS:
            raise ValueError("unsupported model shape")
        self.hidden = hidden
        self.max_events = max_events
        self.encoder = nn.GRU(INPUT_DIM, hidden, batch_first=True)
        # K+1 positive gaps partition the horizon; K logits express count.
        self.time_head = nn.Linear(hidden, max_events + 1)
        self.presence_head = nn.Linear(hidden, max_events)

    def forward(self, visible: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if visible.ndim != 2 or visible.shape[1] != INPUT_DIM:
            raise ValueError("Timing Student input must be causal [T,263]")
        encoded, _ = self.encoder(visible.unsqueeze(0))
        encoded = encoded.squeeze(0)
        gaps = torch.softmax(self.time_head(encoded), dim=-1)
        ordered = torch.cumsum(gaps[:, : self.max_events], dim=-1) * HORIZON_S
        return ordered, self.presence_head(encoded)


def sequence_loss(
    predicted_offsets_s: torch.Tensor,
    presence_logits: torch.Tensor,
    sequence: MultiEventTimingSequence,
    start: int,
    end: int,
) -> torch.Tensor:
    """Set-cardinality plus ordered timing loss on one causal prefix chunk."""

    expected_shape = (end - start, MAX_EVENTS)
    if predicted_offsets_s.shape != expected_shape or presence_logits.shape != expected_shape:
        raise ValueError("multi-event model output shape mismatch")
    present = sequence.present[start:end]
    # Later slots are rarer; positive weighting prevents all-abstain collapse.
    positives = present.sum(dim=0).to(presence_logits.dtype)
    total = torch.tensor(float(max(end - start, 1)), device=presence_logits.device)
    pos_weight = torch.clamp((total - positives) / torch.clamp(positives, min=1.0), 1.0, 20.0)
    existence = F.binary_cross_entropy_with_logits(
        presence_logits, present.to(presence_logits.dtype), pos_weight=pos_weight
    )
    if bool(present.any()):
        timing = F.smooth_l1_loss(
            predicted_offsets_s[present] / HORIZON_S,
            sequence.offsets_s[start:end][present] / HORIZON_S,
        )
    else:
        timing = predicted_offsets_s.sum() * 0.0
    # Prefix consistency: P(slot n+1) must not exceed P(slot n).
    monotone = F.relu(
        torch.sigmoid(presence_logits[:, 1:]) - torch.sigmoid(presence_logits[:, :-1])
    ).mean()
    return timing + existence + 0.1 * monotone


@dataclass(frozen=True, slots=True)
class NeuralForecast:
    offsets_s: tuple[float, ...]
    confidences: tuple[float, ...]
    abstained: bool


def decode_forecast(
    offsets_s: torch.Tensor,
    presence_logits: torch.Tensor,
    *,
    confidence_threshold: float,
) -> NeuralForecast:
    """Decode one frame without chart truth; low first-slot confidence abstains."""

    if offsets_s.shape != (MAX_EVENTS,) or presence_logits.shape != (MAX_EVENTS,):
        raise ValueError("expected one multi-event frame")
    if not 0.0 < confidence_threshold < 1.0:
        raise ValueError("invalid confidence threshold")
    probabilities = torch.sigmoid(presence_logits).detach().cpu().tolist()
    values = offsets_s.detach().cpu().tolist()
    count = 0
    for probability in probabilities:
        if probability < confidence_threshold:
            break
        count += 1
    if count == 0:
        return NeuralForecast((), (), True)
    selected = tuple(float(x) for x in values[:count])
    if any(not 0.0 < x <= HORIZON_S + 1e-6 for x in selected) or any(
        b <= a for a, b in zip(selected, selected[1:])
    ):
        raise RuntimeError("model violated ordered timing contract")
    return NeuralForecast(selected, tuple(float(x) for x in probabilities[:count]), False)
