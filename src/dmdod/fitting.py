from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Callable

from .benchmark import find_fastest_sustainable_rate
from .body import BodyConfig, FingerConfig, HandConfig


@dataclass(frozen=True)
class CalibrationTargets:
    short_single_hz: float = 8.9
    sustained_single_hz: float = 8.2
    same_hand_alternate_hz: float = 12.5
    cross_hand_alternate_hz: float = 16.0


@dataclass(frozen=True)
class CalibrationResult:
    config: BodyConfig
    loss: float
    short_single_hz: float
    sustained_single_hz: float
    alternate_hz: float
    alternate_target_hz: float
    profile: str


def _loss(short: float, sustained: float, alternate: float, targets: CalibrationTargets, alternate_target: float) -> float:
    return (
        ((short - targets.short_single_hz) / targets.short_single_hz) ** 2
        + ((sustained - targets.sustained_single_hz) / targets.sustained_single_hz) ** 2
        + ((alternate - alternate_target) / alternate_target) ** 2
    )


def _evaluate(config: BodyConfig, targets: CalibrationTargets, alternate_target: float, profile: str, *, final: bool) -> CalibrationResult:
    resolution = 1.0 if final else 4.0
    short_duration = 5.0 if final else 2.0
    long_duration = 20.0 if final else 5.0
    alt_duration = 10.0 if final else 3.0
    short = find_fastest_sustainable_rate(mode="single", duration_s=short_duration, resolution_ms=resolution, body_config=config).rate_hz
    sustained = find_fastest_sustainable_rate(mode="single", duration_s=long_duration, resolution_ms=resolution, body_config=config).rate_hz
    alternate = find_fastest_sustainable_rate(mode="alternate", duration_s=alt_duration, resolution_ms=resolution, body_config=config).rate_hz
    return CalibrationResult(config, _loss(short, sustained, alternate, targets, alternate_target), short, sustained, alternate, alternate_target, profile)


def _quick_short(config: BodyConfig, targets: CalibrationTargets) -> float:
    rate = find_fastest_sustainable_rate(mode="single", duration_s=1.5, resolution_ms=5.0, body_config=config).rate_hz
    return abs(rate - targets.short_single_hz) / targets.short_single_hz


def _scaled(config: BodyConfig, *, tau: float = 1.0, force: float = 1.0, damping: float = 1.0,
            fatigue: float = 1.0, recovery: float = 1.0, hand_capacity: float = 1.0,
            hand_fatigue: float = 1.0, hand_recovery: float = 1.0) -> BodyConfig:
    def finger(f: FingerConfig) -> FingerConfig:
        return replace(f, activation_tau_s=f.activation_tau_s * tau, max_force_n=f.max_force_n * force,
                       damping_n_s_m=f.damping_n_s_m * damping, fatigue_gain_s=f.fatigue_gain_s * fatigue,
                       fatigue_recovery_s=f.fatigue_recovery_s * recovery)
    h = config.hand
    hand = replace(h, capacity=max(0.2, h.capacity * hand_capacity),
                   fatigue_gain_s=h.fatigue_gain_s * hand_fatigue,
                   fatigue_recovery_s=h.fatigue_recovery_s * hand_recovery)
    return BodyConfig(left=finger(config.left), right=finger(config.right), hand=hand, same_hand=config.same_hand)


def fit_body_config(targets: CalibrationTargets = CalibrationTargets(), progress: Callable[[str], None] | None = None,
                    *, profile: str = "same-hand") -> CalibrationResult:
    if profile == "same-hand":
        alternate_target = targets.same_hand_alternate_hz
        same_hand = True
    elif profile == "cross-hand":
        alternate_target = targets.cross_hand_alternate_hz
        same_hand = False
    else:
        raise ValueError("profile must be 'same-hand' or 'cross-hand'")

    say = progress or (lambda _: None)
    base = replace(BodyConfig(), same_hand=same_hand)

    # Wider tau range: the previous fit hit its upper boundary at 0.175 s.
    stage1: list[tuple[float, BodyConfig]] = []
    mechanics = list(product((4.0, 6.0, 8.0, 10.0), (0.45, 0.65, 0.85), (1.0, 1.8)))
    for index, (tau, force, damping) in enumerate(mechanics, 1):
        config = _scaled(base, tau=tau, force=force, damping=damping)
        stage1.append((_quick_short(config, targets), config))
        if index % 4 == 0 or index == len(mechanics):
            say(f"stage 1/3: {index}/{len(mechanics)}")
    stage1.sort(key=lambda item: item[0])
    finalists = [config for _, config in stage1[:4]]

    stage2_configs: list[BodyConfig] = []
    for config in finalists:
        if same_hand:
            combinations = product((2.0, 5.0), (0.5, 1.0), (0.55, 0.75, 0.95), (1.0, 3.0))
            for fatigue, recovery, capacity, hand_fatigue in combinations:
                stage2_configs.append(_scaled(config, fatigue=fatigue, recovery=recovery,
                                              hand_capacity=capacity, hand_fatigue=hand_fatigue))
        else:
            for fatigue, recovery in product((2.0, 5.0, 9.0), (0.5, 1.0)):
                stage2_configs.append(_scaled(config, fatigue=fatigue, recovery=recovery))

    evaluated: list[CalibrationResult] = []
    for index, config in enumerate(stage2_configs, 1):
        evaluated.append(_evaluate(config, targets, alternate_target, profile, final=False))
        if index % 8 == 0 or index == len(stage2_configs):
            say(f"stage 2/3: {index}/{len(stage2_configs)}")
    current = min(evaluated, key=lambda result: result.loss).config

    names = ["tau", "force", "damping", "fatigue", "recovery"]
    if same_hand:
        names += ["hand_capacity", "hand_fatigue", "hand_recovery"]
    for round_index, scale in enumerate((1.18, 1.08), 1):
        neighborhood = [current]
        for name in names:
            neighborhood.append(_scaled(current, **{name: 1.0 / scale}))
            neighborhood.append(_scaled(current, **{name: scale}))
        results = [_evaluate(config, targets, alternate_target, profile, final=False) for config in neighborhood]
        current = min(results, key=lambda result: result.loss).config
        say(f"stage 3/3: refinement {round_index}/2")

    say("final validation: 1 s warmup + 5 s / 20 s / 10 s measurement")
    return _evaluate(current, targets, alternate_target, profile, final=True)
