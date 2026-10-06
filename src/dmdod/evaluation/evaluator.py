from __future__ import annotations

from dataclasses import dataclass

from dmdod.motor.keyboard import KeyEvent
from dmdod.motor.env import TimedKeyEvent


@dataclass(frozen=True)
class TargetHit:
    time_s: float
    key: str = "left"


@dataclass(frozen=True)
class Judgement:
    target_time_s: float
    event_time_s: float | None
    error_s: float | None
    key: str


@dataclass(frozen=True)
class EvaluationSummary:
    targets: int
    hits: int
    misses: int
    mean_abs_error_ms: float | None
    max_abs_error_ms: float | None


class TimingEvaluator:
    """Privileged evaluator that owns exact target timestamps.

    Policies should never receive this object, its targets, or Judgement values.
    Later chart/perception code can expose only rendered/noisy observations while
    this evaluator retains exact truth for scoring.
    """

    def __init__(self, targets: list[TargetHit], *, hit_window_s: float = 0.150) -> None:
        if hit_window_s <= 0.0:
            raise ValueError("hit_window_s must be positive")
        self.targets = tuple(sorted(targets, key=lambda x: x.time_s))
        self.hit_window_s = hit_window_s
        self._used = [False] * len(self.targets)
        self._judgements: list[Judgement] = []

    def record(self, events: tuple[TimedKeyEvent, ...]) -> None:
        for event in events:
            if event.event is not KeyEvent.DOWN:
                continue

            best_index: int | None = None
            best_error = self.hit_window_s + 1.0
            for i, target in enumerate(self.targets):
                if self._used[i] or target.key != event.key:
                    continue
                error = abs(event.time_s - target.time_s)
                if error <= self.hit_window_s and error < best_error:
                    best_error = error
                    best_index = i

            if best_index is not None:
                target = self.targets[best_index]
                self._used[best_index] = True
                signed_error = event.time_s - target.time_s
                self._judgements.append(
                    Judgement(target.time_s, event.time_s, signed_error, target.key)
                )

    def finalize(self) -> EvaluationSummary:
        judged_targets = {j.target_time_s for j in self._judgements}
        for i, target in enumerate(self.targets):
            if not self._used[i] and target.time_s not in judged_targets:
                self._judgements.append(Judgement(target.time_s, None, None, target.key))

        errors_ms = [abs(j.error_s) * 1000.0 for j in self._judgements if j.error_s is not None]
        hits = len(errors_ms)
        misses = len(self.targets) - hits
        return EvaluationSummary(
            targets=len(self.targets),
            hits=hits,
            misses=misses,
            mean_abs_error_ms=(sum(errors_ms) / hits if hits else None),
            max_abs_error_ms=(max(errors_ms) if errors_ms else None),
        )

    @property
    def judgements(self) -> tuple[Judgement, ...]:
        return tuple(self._judgements)
