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
    alternate_hz: float = 12.5


@dataclass(frozen=True)
class CalibrationResult:
    config: BodyConfig
    loss: float
    short_single_hz: float
    sustained_single_hz: float
    alternate_hz: float


def _loss(short: float, sustained: float, alternate: float, targets: CalibrationTargets) -> float:
    return (
        ((short - targets.short_single_hz) / targets.short_single_hz) ** 2
        + ((sustained - targets.sustained_single_hz) / targets.sustained_single_hz) ** 2
        + ((alternate - targets.alternate_hz) / targets.alternate_hz) ** 2
    )


def _evaluate(config: BodyConfig, targets: CalibrationTargets, *, final: bool) -> CalibrationResult:
    resolution = 1.0 if final else 4.0
    short_duration = 5.0 if final else 2.0
    long_duration = 20.0 if final else 5.0
    alt_duration = 10.0 if final else 3.0

    short = find_fastest_sustainable_rate(
        mode="single", duration_s=short_duration, resolution_ms=resolution, body_config=config
    ).rate_hz
    sustained = find_fastest_sustainable_rate(
        mode="single", duration_s=long_duration, resolution_ms=resolution, body_config=config
    ).rate_hz
    alternate = find_fastest_sustainable_rate(
        mode="alternate", duration_s=alt_duration, resolution_ms=resolution, body_config=config
    ).rate_hz
    return CalibrationResult(config, _loss(short, sustained, alternate, targets), short, sustained, alternate)


def _quick_short(config: BodyConfig, targets: CalibrationTargets) -> float:
    rate = find_fastest_sustainable_rate(
        mode="single", duration_s=1.5, resolution_ms=5.0, body_config=config
    ).rate_hz
    return abs(rate - targets.short_single_hz) / targets.short_single_hz


def _scaled(
    config: BodyConfig,
    *,
    tau: float = 1.0,
    force: float = 1.0,
    damping: float = 1.0,
    fatigue: float = 1.0,
    recovery: float = 1.0,
    coupling: float = 1.0,
) -> BodyConfig:
    def finger(f: FingerConfig) -> FingerConfig:
        return replace(
            f,
            activation_tau_s=f.activation_tau_s * tau,
            max_force_n=f.max_force_n * force,
            damping_n_s_m=f.damping_n_s_m * damping,
            fatigue_gain_s=f.fatigue_gain_s * fatigue,
            fatigue_recovery_s=f.fatigue_recovery_s * recovery,
        )

    return BodyConfig(
        left=finger(config.left),
        right=finger(config.right),
        left_affected_by_right=min(0.95, config.left_affected_by_right * coupling),
        right_affected_by_left=min(0.95, config.right_affected_by_left * coupling),
    )


def fit_body_config(
    targets: CalibrationTargets = CalibrationTargets(),
    progress: Callable[[str], None] | None = None,
) -> CalibrationResult:
    """Fit a provisional body with a staged deterministic search.

    Stage 1 cheaply identifies plausible short-term mechanics.  Stage 2 only
    evaluates fatigue/coupling for the best few candidates.  Stage 3 performs a
    small local refinement, then one full-duration validation.  This avoids the
    previous tens-of-thousands of full simulation runs.
    """
    say = progress or (lambda _: None)
    base = BodyConfig()

    # Stage 1: short-term mechanics only.  36 candidates, one 1.5 s test each.
    stage1: list[tuple[float, BodyConfig]] = []
    mechanics = list(product((2.0, 3.0, 4.0, 5.0), (0.55, 0.75, 1.0), (1.0, 1.6, 2.3)))
    for index, (tau, force, damping) in enumerate(mechanics, 1):
        config = _scaled(base, tau=tau, force=force, damping=damping)
        stage1.append((_quick_short(config, targets), config))
        if index % 6 == 0 or index == len(mechanics):
            say(f"stage 1/3: {index}/{len(mechanics)}")
    stage1.sort(key=lambda item: item[0])
    finalists = [config for _, config in stage1[:4]]

    # Stage 2: only the four plausible bodies get fatigue/coupling combinations.
    stage2_configs: list[BodyConfig] = []
    for config in finalists:
        for fatigue, recovery, coupling in product((2.0, 5.0, 9.0), (0.5, 1.0), (1.5, 3.5, 6.0)):
            stage2_configs.append(
                _scaled(config, fatigue=fatigue, recovery=recovery, coupling=coupling)
            )

    evaluated: list[CalibrationResult] = []
    for index, config in enumerate(stage2_configs, 1):
        evaluated.append(_evaluate(config, targets, final=False))
        if index % 8 == 0 or index == len(stage2_configs):
            say(f"stage 2/3: {index}/{len(stage2_configs)}")
    evaluated.sort(key=lambda result: result.loss)
    current = evaluated[0].config

    # Stage 3: small coordinate refinement, no Cartesian explosion.
    names = ("tau", "force", "damping", "fatigue", "recovery", "coupling")
    for round_index, scale in enumerate((1.18, 1.08), 1):
        neighborhood = [current]
        for name in names:
            neighborhood.append(_scaled(current, **{name: 1.0 / scale}))
            neighborhood.append(_scaled(current, **{name: scale}))
        results = [_evaluate(config, targets, final=False) for config in neighborhood]
        current = min(results, key=lambda result: result.loss).config
        say(f"stage 3/3: refinement {round_index}/2")

    say("final validation: 5 s / 20 s / 10 s")
    return _evaluate(current, targets, final=True)
