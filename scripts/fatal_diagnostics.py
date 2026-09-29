from __future__ import annotations

from dmdod.adofai_rules import TimingJudgement
from dmdod.keyboard import KeyEvent


class FatalTrackingMixin:
    """Evaluator-only tracker for the first ADOFAI-style fatal failure.

    The wrapped environment keeps its existing continuation/failure semantics.
    This mixin only records where the run would first have died from a miss or
    overload so diagnostic evaluation can report both views at once.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fatal_reason: str | None = None
        self.fatal_floor: int | None = None
        self.fatal_time_s: float | None = None
        self.fatal_survived_targets = 0

    def reset(self):
        self.fatal_reason = None
        self.fatal_floor = None
        self.fatal_time_s = None
        self.fatal_survived_targets = 0
        return super().reset()

    def _record_fatal(self, reason: str, target, time_s: float) -> None:
        if self.fatal_reason is not None:
            return
        self.fatal_reason = reason
        self.fatal_floor = None if target is None else int(target.floor_index)
        self.fatal_time_s = float(time_s)
        self.fatal_survived_targets = int(self._hits)

    def _score_event(self, event):
        if event.event is not KeyEvent.DOWN:
            return super()._score_event(event)

        target = self.privileged_next_target()
        before = len(self.hit_margins)
        reward = super()._score_event(event)
        if self.fatal_reason is None and len(self.hit_margins) > before:
            margin = self.hit_margins[-1]
            if margin is TimingJudgement.FAIL_OVERLOAD:
                self._record_fatal("overload", target, event.time_s)
            elif margin is TimingJudgement.TOO_LATE:
                self._record_fatal("miss", target, event.time_s)
        return reward

    def _expire_misses(self, now_s: float) -> float:
        target = self.privileged_next_target() if self.fatal_reason is None else None
        before = int(self._misses)
        reward = super()._expire_misses(now_s)
        if self.fatal_reason is None and self._misses > before and target is not None:
            fatal_time = target.episode_time_s + self._windows(target).pass_s
            self._record_fatal("miss", target, fatal_time)
        return reward


def fatal_summary(env) -> dict:
    clear = env.fatal_reason is None
    return {
        "clear": clear,
        "fatal_reason": env.fatal_reason,
        "fatal_floor": env.fatal_floor,
        "fatal_time_s": env.fatal_time_s,
        "survived_targets": len(env.segment.targets) if clear else env.fatal_survived_targets,
    }


def format_fatal_summary(env) -> str:
    info = fatal_summary(env)
    if info["clear"]:
        return f"ADOFAI-style: clear=True survived={info['survived_targets']}/{len(env.segment.targets)}"
    return (
        "ADOFAI-style: clear=False "
        f"fatal={info['fatal_reason']} floor={info['fatal_floor']} "
        f"t={info['fatal_time_s']:.3f}s "
        f"survived={info['survived_targets']}/{len(env.segment.targets)}"
    )
