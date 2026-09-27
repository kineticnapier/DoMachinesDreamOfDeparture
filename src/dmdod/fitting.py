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

    @property
    def single_drop_hz(self) -> float:
        return self.short_single_hz - self.sustained_single_hz


@dataclass(frozen=True)
class CalibrationResult:
    config: BodyConfig
    loss: float
    short_single_hz: float
    sustained_single_hz: float
    alternate_hz: float
    alternate_target_hz: float
    profile: str


def _single_loss(short: float, sustained: float, targets: CalibrationTargets) -> float:
    endpoint = (((short-targets.short_single_hz)/targets.short_single_hz)**2
                + ((sustained-targets.sustained_single_hz)/targets.sustained_single_hz)**2)
    actual_drop = short - sustained
    trend = ((actual_drop - targets.single_drop_hz) / targets.short_single_hz) ** 2
    reverse_penalty = (max(0.0, sustained - short) / targets.short_single_hz) ** 2
    return endpoint + 2.0 * trend + 4.0 * reverse_penalty


def _loss(short: float, sustained: float, alternate: float, targets: CalibrationTargets, alternate_target: float) -> float:
    return _single_loss(short, sustained, targets) + ((alternate-alternate_target)/alternate_target)**2


def _rate(config: BodyConfig, *, mode: str, duration_s: float, resolution_ms: float) -> float:
    return find_fastest_sustainable_rate(mode=mode, duration_s=duration_s,
                                         resolution_ms=resolution_ms, body_config=config).rate_hz


def _evaluate(config: BodyConfig, targets: CalibrationTargets, alternate_target: float, profile: str,
              *, resolution_ms: float = 1.0) -> CalibrationResult:
    short = _rate(config, mode="single", duration_s=5.0, resolution_ms=resolution_ms)
    sustained = _rate(config, mode="single", duration_s=20.0, resolution_ms=resolution_ms)
    alternate = _rate(config, mode="alternate", duration_s=10.0, resolution_ms=resolution_ms)
    return CalibrationResult(config, _loss(short, sustained, alternate, targets, alternate_target),
                             short, sustained, alternate, alternate_target, profile)


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
    search_resolution = 2.0

    mechanics = list(product((4.0, 6.0, 8.0, 10.0, 12.0), (0.35, 0.45, 0.65, 0.85), (1.0, 1.8)))
    ranked = []
    for i, (tau, force, damping) in enumerate(mechanics, 1):
        c = _scaled(base, tau=tau, force=force, damping=damping)
        rate = _rate(c, mode="single", duration_s=5.0, resolution_ms=search_resolution)
        ranked.append((abs(rate-targets.short_single_hz), c))
        if i % 8 == 0 or i == len(mechanics):
            say(f"stage 1/3 mechanics: {i}/{len(mechanics)}")
    ranked.sort(key=lambda x: x[0])
    mechanics_finalists = [c for _, c in ranked[:3]]

    fatigue_candidates = []
    # Include stronger/slower-recovering regimes now that the 5s->20s drop is
    # explicitly part of the objective rather than an incidental endpoint fit.
    combos = list(product((2.0, 5.0, 10.0, 18.0, 30.0), (0.08, 0.15, 0.35, 0.7), (0.35, 0.55, 0.8, 1.0)))
    total = len(mechanics_finalists) * len(combos)
    i = 0
    for base_c in mechanics_finalists:
        for gain, recovery, threshold in combos:
            i += 1
            c = _scaled(base_c, fatigue=gain, recovery=recovery, fatigue_threshold=threshold)
            short = _rate(c, mode="single", duration_s=5.0, resolution_ms=search_resolution)
            sustained = _rate(c, mode="single", duration_s=20.0, resolution_ms=search_resolution)
            fatigue_candidates.append((_single_loss(short, sustained, targets), c))
            if i % 20 == 0 or i == total:
                say(f"stage 2/3 fatigue+trend: {i}/{total}")
    current = min(fatigue_candidates, key=lambda x: x[0])[1]

    if same_hand:
        coordination = list(product((0.35, 0.5, 0.65, 0.8), (0.6, 1.0, 1.8, 3.0), (0.25, 0.5, 0.8, 1.0)))
        coord_results = []
        for i, (capacity, switch, floor) in enumerate(coordination, 1):
            c = _scaled(current, hand_capacity=capacity, switch_tau=switch, coordination_floor=floor)
            rate = _rate(c, mode="alternate", duration_s=10.0, resolution_ms=search_resolution)
            score = ((rate-alternate_target)/alternate_target)**2
            coord_results.append((score, c))
            if i % 16 == 0 or i == len(coordination):
                say(f"stage 3/3 coordination: {i}/{len(coordination)}")
        current = min(coord_results, key=lambda x: x[0])[1]
    else:
        say("stage 3/3 coordination: skipped for cross-hand profile")

    say("final validation: same 5 s / 20 s / 10 s tests at 1 ms search resolution")
    return _evaluate(current, targets, alternate_target, profile, resolution_ms=1.0)
