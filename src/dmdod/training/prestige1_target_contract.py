from __future__ import annotations

"""Prestige I v2.3 evaluator-only target/physical timing contract.

No Student policy may receive fields from this module. Stage A/C legacy
label APIs are untouched. The signed late interval is *not* represented by
the old 0..200ms regression target: future training must design it explicitly.
"""

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

from dmdod.adofai_rules import TimingDifficulty, TimingJudgement, classify_timing, timing_windows

EPS = 1e-9


class TargetPhase(str, Enum):
    HIT = "hit"
    MISSED = "missed"
    FUTURE_OUTSIDE_PASS = "future_outside_pass"
    EARLY_PASS = "early_pass"
    ON_TIME = "on_time"
    LATE_PASS = "late_pass"
    EXPIRED_PENDING = "expired_pending"


@dataclass(frozen=True)
class TargetContract:
    """State of *one* chart target at an evaluator-only instant.

    remaining_s is signed (positive means future). Classification uses the
    same chart BPM/difficulty/pitch parameters as the physical simulator;
    legal_to_hit is false for both TooEarly and TooLate conditions.
    """

    target_index: int
    phase: str
    remaining_s: float
    pass_radius_s: float
    resolved: bool
    hit: bool
    missed: bool
    legal_to_hit: bool
    would_judge: str | None

    def to_json(self):
        return asdict(self)


def target_contract(targets: Sequence, used: Sequence[bool], missed: Sequence[bool],
                    index: int, now_s: float, *,
                    difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
                    timing_scale: float = 1.0, pitch: float = 1.0) -> TargetContract:
    if len(targets) != len(used) or len(targets) != len(missed):
        raise ValueError("Target resolution state length mismatch")
    if not (0 <= index < len(targets)):
        raise IndexError("Target index outside playable chart")
    if any(bool(u) and bool(m) for u, m in zip(used, missed)):
        raise ValueError("Hit and Miss flags are mutually exclusive")
    if not isfinite(float(now_s)):
        raise ValueError("Nonfinite episode time")
    target = targets[index]
    remain = float(target.episode_time_s) - float(now_s)
    windows = timing_windows(target.bpm, difficulty=difficulty,
                             timing_scale=timing_scale, pitch=pitch)
    pass_radius = float(windows.pass_s)
    if used[index]:
        phase, would, legal = TargetPhase.HIT, None, False
    elif missed[index]:
        phase, would, legal = TargetPhase.MISSED, None, False
    else:
        judgement = classify_timing(
            -remain, target.bpm, difficulty=difficulty,
            timing_scale=timing_scale, pitch=pitch)
        would = judgement.value
        legal = judgement not in (TimingJudgement.TOO_EARLY, TimingJudgement.TOO_LATE)
        if judgement is TimingJudgement.TOO_EARLY:
            phase = TargetPhase.FUTURE_OUTSIDE_PASS
        elif judgement is TimingJudgement.TOO_LATE:
            phase = TargetPhase.EXPIRED_PENDING
        elif remain > EPS:
            phase = TargetPhase.EARLY_PASS
        elif remain < -EPS:
            phase = TargetPhase.LATE_PASS
        else:
            phase = TargetPhase.ON_TIME
    return TargetContract(
        target_index=index, phase=phase.value, remaining_s=remain,
        pass_radius_s=pass_radius, resolved=bool(used[index] or missed[index]),
        hit=bool(used[index]), missed=bool(missed[index]),
        legal_to_hit=bool(legal), would_judge=would,
    )


def next_unresolved_contract(targets: Sequence, used: Sequence[bool],
                             missed: Sequence[bool], now_s: float, *,
                             difficulty: TimingDifficulty | str = TimingDifficulty.NORMAL,
                             timing_scale: float = 1.0, pitch: float = 1.0
                             ) -> TargetContract | None:
    """Match RealChartMotorEnv._next_target_index() exactly.

    This is a private evaluator/debug mapping, *not* something observable
    from the 263D HUD. It cannot be supplied to Motor at inference.
    """
    if len(targets) != len(used) or len(targets) != len(missed):
        raise ValueError("Target resolution state length mismatch")
    for i, (u, m) in enumerate(zip(used, missed)):
        if not (u or m):
            return target_contract(
                targets, used, missed, i, now_s, difficulty=difficulty,
                timing_scale=timing_scale, pitch=pitch)
    return None


def contract_from_env(env, now_s: float | None = None) -> TargetContract | None:
    """Bridge solely for read-only Expert diagnostic inspection."""
    if now_s is None:
        now_s = env.privileged_episode_time_s()
    record = next_unresolved_contract(
        env.segment.targets, env._used, env._missed, now_s,
        difficulty=env.difficulty,
        timing_scale=env.timing_scale, pitch=env.segment.pitch_ratio)
    selected = env._next_target_index()
    if (record.target_index if record is not None else None) != selected:
        raise RuntimeError("Evaluator target contract diverges from physical target")
    if record is not None:
        target = env.segment.targets[record.target_index]
        # Use exactly the simulator's _classify path as the authority.
        judgement = env._classify(float(now_s)-float(target.episode_time_s), target)
        if record.would_judge != judgement.value:
            raise RuntimeError("Target contract differs from physics judgement")
    return record
