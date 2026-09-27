from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Callable

from .benchmark import find_fastest_sustainable_rate
from .body import BodyConfig, FingerConfig


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
    return (((short-targets.short_single_hz)/targets.short_single_hz)**2
            + ((sustained-targets.sustained_single_hz)/targets.sustained_single_hz)**2
            + ((alternate-alternate_target)/alternate_target)**2)


def _evaluate(config: BodyConfig, targets: CalibrationTargets, alternate_target: float, profile: str, *, final: bool) -> CalibrationResult:
    resolution = 1.0 if final else 4.0
    short = find_fastest_sustainable_rate(mode="single", duration_s=5.0 if final else 2.0, resolution_ms=resolution, body_config=config).rate_hz
    sustained = find_fastest_sustainable_rate(mode="single", duration_s=20.0 if final else 5.0, resolution_ms=resolution, body_config=config).rate_hz
    alternate = find_fastest_sustainable_rate(mode="alternate", duration_s=10.0 if final else 3.0, resolution_ms=resolution, body_config=config).rate_hz
    return CalibrationResult(config, _loss(short, sustained, alternate, targets, alternate_target), short, sustained, alternate, alternate_target, profile)


def _single_rates(config: BodyConfig) -> tuple[float, float]:
    short = find_fastest_sustainable_rate(mode="single", duration_s=2.0, resolution_ms=4.0, body_config=config).rate_hz
    sustained = find_fastest_sustainable_rate(mode="single", duration_s=5.0, resolution_ms=4.0, body_config=config).rate_hz
    return short, sustained


def _scaled(config: BodyConfig, *, tau=1.0, force=1.0, damping=1.0, fatigue=1.0, recovery=1.0,
            fatigue_threshold=1.0, hand_capacity=1.0, hand_fatigue=1.0, hand_recovery=1.0,
            hand_threshold=1.0, switch_tau=1.0, coordination_floor=1.0) -> BodyConfig:
    def finger(f: FingerConfig) -> FingerConfig:
        return replace(f, activation_tau_s=f.activation_tau_s*tau, max_force_n=f.max_force_n*force,
                       damping_n_s_m=f.damping_n_s_m*damping, fatigue_gain_s=f.fatigue_gain_s*fatigue,
                       fatigue_recovery_s=f.fatigue_recovery_s*recovery,
                       fatigue_threshold=max(0.05, min(0.95, f.fatigue_threshold*fatigue_threshold)))
    h = config.hand
    hand = replace(h, capacity=max(0.2, h.capacity*hand_capacity), fatigue_gain_s=h.fatigue_gain_s*hand_fatigue,
                   fatigue_recovery_s=h.fatigue_recovery_s*hand_recovery,
                   fatigue_threshold=max(0.05, min(0.95, h.fatigue_threshold*hand_threshold)),
                   switch_tau_s=max(0.001, h.switch_tau_s*switch_tau),
                   coordination_floor=max(0.02, min(1.0, h.coordination_floor*coordination_floor)))
    return BodyConfig(left=finger(config.left), right=finger(config.right), hand=hand, same_hand=config.same_hand)


def fit_body_config(targets: CalibrationTargets = CalibrationTargets(), progress: Callable[[str], None] | None = None,
                    *, profile: str = "same-hand") -> CalibrationResult:
    if profile == "same-hand":
        alternate_target, same_hand = targets.same_hand_alternate_hz, True
    elif profile == "cross-hand":
        alternate_target, same_hand = targets.cross_hand_alternate_hz, False
    else:
        raise ValueError("profile must be 'same-hand' or 'cross-hand'")
    say = progress or (lambda _: None)
    base = replace(BodyConfig(), same_hand=same_hand)

    # Stage 1: mechanics. Only the short single-finger limit matters here.
    mechanics = list(product((4.0, 6.0, 8.0, 10.0), (0.45, 0.65, 0.85), (1.0, 1.8)))
    ranked = []
    for i, (tau, force, damping) in enumerate(mechanics, 1):
        c = _scaled(base, tau=tau, force=force, damping=damping)
        rate = find_fastest_sustainable_rate(mode="single", duration_s=1.5, resolution_ms=5.0, body_config=c).rate_hz
        ranked.append((abs(rate-targets.short_single_hz), c))
        if i % 4 == 0: say(f"stage 1/3 mechanics: {i}/{len(mechanics)}")
    ranked.sort(key=lambda x: x[0])
    mechanics_finalists = [c for _, c in ranked[:3]]

    # Stage 2: fatigue only. No coordination Cartesian product.
    fatigue_candidates = []
    combos = list(product((2.0, 5.0, 10.0), (0.35, 0.7, 1.0), (0.55, 0.8, 1.05)))
    total = len(mechanics_finalists) * len(combos)
    i = 0
    for base_c in mechanics_finalists:
        for gain, recovery, threshold in combos:
            i += 1
            c = _scaled(base_c, fatigue=gain, recovery=recovery, fatigue_threshold=threshold)
            short, sustained = _single_rates(c)
            score = ((short-targets.short_single_hz)/targets.short_single_hz)**2 + ((sustained-targets.sustained_single_hz)/targets.sustained_single_hz)**2
            fatigue_candidates.append((score, c))
            if i % 9 == 0 or i == total: say(f"stage 2/3 fatigue: {i}/{total}")
    fatigue_candidates.sort(key=lambda x: x[0])
    current = fatigue_candidates[0][1]

    # Stage 3: coordination only for same-hand. Cross-hand has no shared switch.
    if same_hand:
        coordination = list(product((0.45, 0.65, 0.85), (0.6, 1.0, 1.8, 3.0), (0.35, 0.6, 1.0)))
        coord_results = []
        for i, (capacity, switch, floor) in enumerate(coordination, 1):
            c = _scaled(current, hand_capacity=capacity, switch_tau=switch, coordination_floor=floor)
            rate = find_fastest_sustainable_rate(mode="alternate", duration_s=3.0, resolution_ms=4.0, body_config=c).rate_hz
            score = ((rate-alternate_target)/alternate_target)**2
            coord_results.append((score, c))
            if i % 9 == 0 or i == len(coordination): say(f"stage 3/3 coordination: {i}/{len(coordination)}")
        current = min(coord_results, key=lambda x: x[0])[1]
    else:
        say("stage 3/3 coordination: skipped for cross-hand profile")

    say("final validation: 1 s warmup + 5 s / 20 s / 10 s measurement")
    return _evaluate(current, targets, alternate_target, profile, final=True)
