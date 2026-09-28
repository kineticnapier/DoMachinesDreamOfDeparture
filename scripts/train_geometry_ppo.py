from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tqdm is required. Run: uv sync --extra dev --extra rl --inexact"
    ) from exc

try:
    from . import train_geometry_ppo_core as core
except ImportError:  # direct script execution
    import train_geometry_ppo_core as core


PROGRESS_ENABLED = True
ALLOW_PRECISION_SKIP = False

# Accuracy curriculum. These can be raised from the CLI later, all the way to PP=100%.
PRECISION_MIN_BPM_XACC = 90.0
PRECISION_OVERALL_XACC = 95.0
PRECISION_MIN_BPM_PP = 0.60
PRECISION_OVERALL_PP = 0.80
COMPLETION_MIN_BPM_HIT = 0.95
COMPLETION_MIN_BPM_FULL = 0.90

# v0.3.3 evaluation/sampling defaults.
QUICK_EVAL_EPISODES = 20
QUICK_RETENTION_EPISODES = 5
LATE_EVAL_EVERY_UPDATES = 4
WORST_BPM_FOCUS = 0.50
WORST_BPM_COUNT = 2

_ORIGINAL_SAVE_CHECKPOINT = core.save_checkpoint


@dataclass(frozen=True)
class AccuracyBpmSlice(core.BpmSlice):
    x_accuracy_percent: float = 0.0
    perfect_rate: float = 0.0


@dataclass(frozen=True)
class AccuracyProbe(core.Probe):
    x_accuracy_percent: float = 0.0
    perfect_rate: float = 0.0

    @property
    def min_bpm_x_accuracy_percent(self) -> float:
        values = [
            float(getattr(item, "x_accuracy_percent", self.x_accuracy_percent))
            for item in self.bpm_slices
        ]
        return min(values, default=self.x_accuracy_percent)

    @property
    def min_bpm_perfect_rate(self) -> float:
        values = [
            float(getattr(item, "perfect_rate", self.perfect_rate))
            for item in self.bpm_slices
        ]
        return min(values, default=self.perfect_rate)

    @property
    def min_bpm_hit_rate(self) -> float:
        return min((item.hit_rate for item in self.bpm_slices), default=self.hit_rate)

    @property
    def min_bpm_full_rate(self) -> float:
        return min((item.full_rate for item in self.bpm_slices), default=self.full_rate)


def _bar(iterable=None, **kwargs):
    kwargs.setdefault("disable", not PROGRESS_ENABLED)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("ascii", False)
    return tqdm(iterable, **kwargs)


def _write(message: str = "") -> None:
    if PROGRESS_ENABLED:
        tqdm.write(message)
    else:
        print(message)


def _fmt_ms(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def _xacc(probe: core.Probe) -> float:
    return float(getattr(probe, "x_accuracy_percent", 0.0))


def _pp(probe: core.Probe) -> float:
    return float(getattr(probe, "perfect_rate", 0.0))


def _min_xacc(probe: core.Probe) -> float:
    return float(getattr(probe, "min_bpm_x_accuracy_percent", _xacc(probe)))


def _min_pp(probe: core.Probe) -> float:
    return float(getattr(probe, "min_bpm_perfect_rate", _pp(probe)))


def _min_hit(probe: core.Probe) -> float:
    return float(
        getattr(
            probe,
            "min_bpm_hit_rate",
            min((item.hit_rate for item in probe.bpm_slices), default=probe.hit_rate),
        )
    )


def _min_full(probe: core.Probe) -> float:
    return float(
        getattr(
            probe,
            "min_bpm_full_rate",
            min((item.full_rate for item in probe.bpm_slices), default=probe.full_rate),
        )
    )


def deterministic_probe(
    model: core.RecurrentActorCritic,
    device: core.torch.device,
    *,
    phase: core.CurriculumPhase,
    control_dt: float,
    episodes: int,
    eval_bpm_points: int,
    seed_base: int,
) -> AccuracyProbe:
    """Deterministic probe with ADOFAI X-Accuracy and Perfect metrics."""

    jitter_s = phase.eval_phase_jitter_ms / 1000.0
    offsets = (
        (0.0,)
        if episodes == 1
        else tuple(-jitter_s + 2.0 * jitter_s * i / (episodes - 1) for i in range(episodes))
    )
    bpms = core.bpm_points(phase.bpm_min, phase.bpm_max, eval_bpm_points)
    buckets: dict[float, dict[str, object]] = {
        bpm: {
            "episodes": 0,
            "hits": 0,
            "targets": 0,
            "full": 0,
            "clean": 0,
            "errors": [],
            "xacc_points": 0.0,
            "perfects": 0,
        }
        for bpm in bpms
    }

    hits = targets = full = clean = overloads = too_early = perfects = 0
    xacc_points = 0.0
    errors: list[float] = []
    was_training = model.training
    model.eval()

    iterator = _bar(
        enumerate(offsets),
        total=len(offsets),
        desc="probe",
        unit="ep",
        leave=False,
        position=2,
    )
    for i, offset in iterator:
        bpm = bpms[i % len(bpms)]
        env = core.make_env(
            bpm=bpm,
            notes=phase.notes,
            start_s=max(0.050, core.curriculum_start_s(phase.notes) + offset),
            control_dt=control_dt,
            config=phase.vision,
            seed=seed_base + i * 1009,
        )
        observation = env.reset()
        state = model.initial_state(device)
        while True:
            action, state = model.deterministic_action(
                core.observation_tensor(observation, device), state
            )
            transition = env.step(action)
            observation = transition.observation
            if transition.done:
                break

        stats = env.stats
        is_full = stats.hits == stats.targets and not stats.overloaded
        is_clean = is_full and stats.too_early_presses == 0
        episode_errors = env.timing_errors_ms
        episode_xacc_points = stats.x_accuracy_percent * stats.targets / 100.0

        hits += stats.hits
        targets += stats.targets
        full += int(is_full)
        clean += int(is_clean)
        overloads += int(stats.overloaded)
        too_early += stats.too_early_presses
        perfects += stats.perfects
        xacc_points += episode_xacc_points
        errors.extend(episode_errors)

        bucket = buckets[bpm]
        bucket["episodes"] = int(bucket["episodes"]) + 1
        bucket["hits"] = int(bucket["hits"]) + stats.hits
        bucket["targets"] = int(bucket["targets"]) + stats.targets
        bucket["full"] = int(bucket["full"]) + int(is_full)
        bucket["clean"] = int(bucket["clean"]) + int(is_clean)
        bucket["xacc_points"] = float(bucket["xacc_points"]) + episode_xacc_points
        bucket["perfects"] = int(bucket["perfects"]) + stats.perfects
        bucket_errors = bucket["errors"]
        assert isinstance(bucket_errors, list)
        bucket_errors.extend(episode_errors)

    if was_training:
        model.train()

    slices = tuple(
        AccuracyBpmSlice(
            bpm=bpm,
            episodes=int(bucket["episodes"]),
            hits=int(bucket["hits"]),
            targets=int(bucket["targets"]),
            full=int(bucket["full"]),
            clean=int(bucket["clean"]),
            errors_ms=tuple(float(x) for x in bucket["errors"]),
            x_accuracy_percent=(
                100.0 * float(bucket["xacc_points"]) / max(1, int(bucket["targets"]))
            ),
            perfect_rate=int(bucket["perfects"]) / max(1, int(bucket["targets"])),
        )
        for bpm, bucket in buckets.items()
    )
    return AccuracyProbe(
        episodes=episodes,
        hits=hits,
        targets=targets,
        full=full,
        clean=clean,
        overloads=overloads,
        too_early=too_early,
        errors_ms=tuple(errors),
        bpm_slices=slices,
        x_accuracy_percent=100.0 * xacc_points / max(1, targets),
        perfect_rate=perfects / max(1, targets),
    )


def completion_passes(
    probe: core.Probe,
    retention: core.Retention,
    args,
    notes: int,
) -> bool:
    if probe.overloads or retention.overloads:
        return False
    if retention.min_hit_rate < args.retention_hit_rate:
        return False
    if retention.min_full_rate < args.retention_full_rate:
        return False
    if _min_hit(probe) < COMPLETION_MIN_BPM_HIT:
        return False
    if _min_full(probe) < COMPLETION_MIN_BPM_FULL:
        return False
    if notes == 1:
        return probe.hit_rate >= args.stage1_hit_rate and probe.full_rate >= args.stage1_full_rate
    return probe.hit_rate >= args.advance_hit_rate and probe.full_rate >= args.advance_full_rate


def precision_passes(probe: core.Probe, args, notes: int) -> bool:
    del args, notes
    return (
        _min_xacc(probe) >= PRECISION_MIN_BPM_XACC
        and _xacc(probe) >= PRECISION_OVERALL_XACC
        and _min_pp(probe) >= PRECISION_MIN_BPM_PP
        and _pp(probe) >= PRECISION_OVERALL_PP
    )


def passes(probe: core.Probe, retention: core.Retention, args, notes: int) -> bool:
    return completion_passes(probe, retention, args, notes) and precision_passes(
        probe, args, notes
    )


def rank_key(probe: core.Probe, retention: core.Retention) -> tuple[float, ...]:
    """Prefer stable clears first, then worst-BPM accuracy before averages."""

    mae = probe.mae_ms
    bias = probe.max_abs_bpm_bias_ms
    return (
        1.0 if retention.overloads == 0 else 0.0,
        -float(retention.overloads),
        retention.min_hit_rate,
        retention.min_full_rate,
        1.0 if probe.overloads == 0 else 0.0,
        -float(probe.overloads),
        _min_hit(probe),
        _min_full(probe),
        probe.hit_rate,
        probe.full_rate,
        _min_xacc(probe),
        _xacc(probe),
        _min_pp(probe),
        _pp(probe),
        -(bias if bias is not None else float("inf")),
        -(mae if mae is not None else float("inf")),
        probe.clean_rate,
    )


def _slice_xacc(item: core.BpmSlice) -> float:
    return float(getattr(item, "x_accuracy_percent", 0.0))


def _slice_pp(item: core.BpmSlice) -> float:
    return float(getattr(item, "perfect_rate", 0.0))


def _slice_deficit(item: core.BpmSlice) -> float:
    """Weighted gate deficit used only to decide where to collect more rollouts."""

    return (
        4.0 * max(0.0, COMPLETION_MIN_BPM_HIT - item.hit_rate)
        + 4.0 * max(0.0, COMPLETION_MIN_BPM_FULL - item.full_rate)
        + 2.0 * max(0.0, (PRECISION_MIN_BPM_XACC - _slice_xacc(item)) / 100.0)
        + 2.0 * max(0.0, PRECISION_MIN_BPM_PP - _slice_pp(item))
    )


def weak_bpms(probe: core.Probe, count: int | None = None) -> tuple[float, ...]:
    """Return the weakest BPM slices, preferring actual gate deficits."""

    if not probe.bpm_slices:
        return ()
    count = WORST_BPM_COUNT if count is None else count
    ranked = sorted(
        probe.bpm_slices,
        key=lambda item: (
            _slice_deficit(item),
            100.0 - _slice_xacc(item),
            1.0 - _slice_pp(item),
            -item.hit_rate,
            -item.full_rate,
        ),
        reverse=True,
    )
    return tuple(item.bpm for item in ranked[: max(1, count)])


def focused_training_bpm_schedule(
    phase: core.CurriculumPhase,
    *,
    episodes: int,
    points: int,
    rng: random.Random,
    focus_bpms: tuple[float, ...] = (),
    focus_fraction: float | None = None,
) -> list[float]:
    """Mix stratified coverage with repeated rollouts on the current weak BPMs."""

    fraction = WORST_BPM_FOCUS if focus_fraction is None else focus_fraction
    if not focus_bpms or fraction <= 0.0:
        return core.training_bpm_schedule(
            phase, episodes=episodes, points=points, rng=rng
        )

    focus_count = min(episodes, max(1, int(round(episodes * fraction))))
    base_count = episodes - focus_count
    schedule: list[float] = []
    if base_count:
        schedule.extend(
            core.training_bpm_schedule(
                phase, episodes=base_count, points=points, rng=rng
            )
        )
    schedule.extend(focus_bpms[i % len(focus_bpms)] for i in range(focus_count))
    rng.shuffle(schedule)
    return schedule


def _retention_probe(
    model: core.RecurrentActorCritic,
    device: core.torch.device,
    *,
    previous_notes: tuple[int, ...],
    args,
    episodes: int,
) -> core.Retention:
    if not previous_notes:
        return core.Retention(())
    tuned = copy.copy(args)
    tuned.retention_episodes = episodes
    return core.previous_note_probes(
        model, device, previous_notes=previous_notes, args=tuned
    )


def _evaluate(
    model: core.RecurrentActorCritic,
    device: core.torch.device,
    *,
    phase: core.CurriculumPhase,
    previous_notes: tuple[int, ...],
    args,
    episodes: int,
    retention_episodes: int,
    seed_base: int,
    label: str,
) -> tuple[AccuracyProbe, core.Retention]:
    probe = deterministic_probe(
        model,
        device,
        phase=phase,
        control_dt=args.control_dt,
        episodes=episodes,
        eval_bpm_points=args.eval_bpm_points,
        seed_base=seed_base,
    )
    retention = _retention_probe(
        model,
        device,
        previous_notes=previous_notes,
        args=args,
        episodes=retention_episodes,
    )
    _print_probe(label, probe)
    _print_gate(probe, retention, args, phase.notes)
    return probe, retention


def _eval_interval(args, phase_index: int) -> int:
    if phase_index < 5:
        return args.eval_every_updates
    return max(args.eval_every_updates, LATE_EVAL_EVERY_UPDATES)


def save_checkpoint(
    path: Path,
    model: core.RecurrentActorCritic,
    optimizer: core.torch.optim.Optimizer,
    *,
    args,
    phase_index: int,
    phase: core.CurriculumPhase,
    global_update: int,
    probe: core.Probe,
) -> None:
    """Keep v0.3 checkpoint compatibility while persisting accuracy metadata."""

    _ORIGINAL_SAVE_CHECKPOINT(
        path,
        model,
        optimizer,
        args=args,
        phase_index=phase_index,
        phase=phase,
        global_update=global_update,
        probe=probe,
    )
    saved = core.torch.load(path, map_location="cpu")
    saved["format_version"] = 12
    saved["trainer_ui_version"] = "0.3.3-adaptive-probe"
    saved["accuracy_gate"] = {
        "min_bpm_xacc": PRECISION_MIN_BPM_XACC,
        "overall_xacc": PRECISION_OVERALL_XACC,
        "min_bpm_perfect_rate": PRECISION_MIN_BPM_PP,
        "overall_perfect_rate": PRECISION_OVERALL_PP,
        "min_bpm_hit_rate": COMPLETION_MIN_BPM_HIT,
        "min_bpm_full_rate": COMPLETION_MIN_BPM_FULL,
    }
    saved["adaptive_probe"] = {
        "quick_eval_episodes": QUICK_EVAL_EPISODES,
        "quick_retention_episodes": QUICK_RETENTION_EPISODES,
        "late_eval_every_updates": LATE_EVAL_EVERY_UPDATES,
        "worst_bpm_focus": WORST_BPM_FOCUS,
        "worst_bpm_count": WORST_BPM_COUNT,
    }
    saved.setdefault("probe", {}).update(
        {
            "x_accuracy_percent": _xacc(probe),
            "perfect_rate": _pp(probe),
            "min_bpm_x_accuracy_percent": _min_xacc(probe),
            "min_bpm_perfect_rate": _min_pp(probe),
            "min_bpm_hit_rate": _min_hit(probe),
            "min_bpm_full_rate": _min_full(probe),
        }
    )
    core.torch.save(saved, path)


def _probe_summary(label: str, probe: core.Probe) -> str:
    return (
        f"{label:<9} H={probe.hit_rate:.3f} F={probe.full_rate:.3f} "
        f"X={_xacc(probe):6.2f}% PP={_pp(probe):5.1%} "
        f"minX={_min_xacc(probe):5.1f}% minPP={_min_pp(probe):4.0%} "
        f"E={_fmt_ms(probe.mean_error_ms, signed=True)}ms "
        f"MAE={_fmt_ms(probe.mae_ms)}ms O={probe.overloads}"
    )


def _print_probe(label: str, probe: core.Probe) -> None:
    _write(_probe_summary(label, probe))
    if not probe.bpm_slices:
        return
    pieces: list[str] = []
    for item in probe.bpm_slices:
        pieces.append(
            f"{item.bpm:g}:H{item.hit_rate:.2f} F{item.full_rate:.2f} "
            f"X{_slice_xacc(item):.1f} P{_slice_pp(item):.0%} "
            f"E{_fmt_ms(item.mean_error_ms, signed=True)}"
        )
    _write("  BPM  " + " | ".join(pieces))


def _print_gate(probe: core.Probe, retention: core.Retention, args, notes: int) -> None:
    completion = completion_passes(probe, retention, args, notes)
    precision = precision_passes(probe, args, notes)
    _write(
        "  gate "
        f"clear={'PASS' if completion else 'wait'} "
        f"acc={'PASS' if precision else 'wait'} | "
        f"minH={_min_hit(probe):.2f}/{COMPLETION_MIN_BPM_HIT:.2f} "
        f"minF={_min_full(probe):.2f}/{COMPLETION_MIN_BPM_FULL:.2f} "
        f"minX={_min_xacc(probe):.1f}/{PRECISION_MIN_BPM_XACC:.1f} "
        f"X={_xacc(probe):.1f}/{PRECISION_OVERALL_XACC:.1f} "
        f"minPP={_min_pp(probe):.0%}/{PRECISION_MIN_BPM_PP:.0%} "
        f"PP={_pp(probe):.0%}/{PRECISION_OVERALL_PP:.0%}"
    )


def _print_header(args, phases: tuple[core.CurriculumPhase, ...], mode: str) -> None:
    _write("=== DMDOD / Planet Geometry PPO v0.3.3 ===")
    _write(
        f"mode={mode}  device={args.device}  seed={args.seed}  checkpoint={args.checkpoint}"
    )
    _write(
        f"task={args.notes} notes  BPM={args.bpm_min:g}..{args.bpm_max:g}  "
        f"control={args.control_dt*1000:.1f}ms  vision={args.vision_hz:g}Hz/"
        f"{args.vision_latency_ms:g}ms"
    )
    _write(
        f"PPO={args.rollout_episodes} ep x {args.ppo_epochs} epochs  "
        f"clip={args.ppo_clip:.2f}  KL={args.target_kl:.3f}  lr={args.lr:g}"
    )
    _write(
        f"accuracy gate: minX>={PRECISION_MIN_BPM_XACC:g}% X>={PRECISION_OVERALL_XACC:g}% "
        f"minPP>={PRECISION_MIN_BPM_PP:.0%} PP>={PRECISION_OVERALL_PP:.0%}"
    )
    _write(
        f"clear gate: each BPM H>={COMPLETION_MIN_BPM_HIT:.2f} "
        f"F>={COMPLETION_MIN_BPM_FULL:.2f}; precision-skip={'on' if ALLOW_PRECISION_SKIP else 'off'}"
    )
    _write(
        f"screen={QUICK_EVAL_EPISODES}ep retention={QUICK_RETENTION_EPISODES}ep; "
        f"verify={args.eval_episodes}ep retention={args.retention_episodes}ep; "
        f"P5+ eval every {LATE_EVAL_EVERY_UPDATES} upd"
    )
    _write(
        f"weak-BPM focus={WORST_BPM_FOCUS:.0%} on worst {WORST_BPM_COUNT}; "
        f"rollback={args.rollback_drop:.2f}; phases={len(phases)}"
    )
    _write("visible=motor+planet geometry | hidden=time/BPM/target-angle/error/direction")
    _write()


def train_ui(args) -> None:
    core.torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = core.torch.device(args.device)
    phases = core.build_curriculum(args)
    model = core.RecurrentActorCritic(
        input_dim=core.GEOMETRY_INPUT_DIM,
        hidden_dim=64,
        initial_log_std=args.initial_log_std,
    ).to(device)
    optimizer = core.torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = Path(args.checkpoint)
    resume_phase = 0
    global_update = 0
    mode = "fresh"

    if args.resume and args.warm_start:
        raise SystemExit("--resume and --warm-start are mutually exclusive")
    if args.warm_start:
        source = Path(args.warm_start)
        if not source.exists():
            raise SystemExit(f"warm-start checkpoint not found: {source}")
        copied = core.warm_start(model, source, device)
        mode = f"warm-start:{source.name}"
        _write(f"warm-start copied: {copied}")
    if args.resume:
        if not checkpoint.exists():
            raise SystemExit(f"resume checkpoint not found: {checkpoint}")
        saved = core.torch.load(checkpoint, map_location=device)
        if saved.get("experiment") != "planet-geometry-ppo-v0.3":
            raise SystemExit("checkpoint is not geometry PPO v0.3")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        resume_phase = int(saved.get("curriculum_phase_index", 1)) - 1
        global_update = int(saved.get("global_update", 0))
        mode = f"resume:P{resume_phase + 1}"

    _print_header(args, phases, mode)
    phase_bar = _bar(
        total=len(phases), initial=resume_phase, desc="curriculum", unit="phase", position=0
    )

    try:
        for phase_pos in range(resume_phase, len(phases)):
            phase = phases[phase_pos]
            phase_index = phase_pos + 1
            previous_notes = (
                tuple(notes for notes in core.note_curriculum(phase.notes) if notes < phase.notes)
                if phase.notes > 1
                else ()
            )
            phase_bar.set_description_str(f"P{phase_index:02d}/{len(phases):02d} {phase.name}")
            _write(
                f"\n[P{phase_index:02d}/{len(phases):02d}] {phase.name}  "
                f"notes={phase.notes} BPM={phase.bpm_min:g}..{phase.bpm_max:g} "
                f"jitter=+/-{phase.train_phase_jitter_ms:g}ms"
            )

            quick_seed = args.seed * 1000000 + phase_index * 10000
            full_seed = quick_seed + 500000
            baseline, baseline_retention = _evaluate(
                model,
                device,
                phase=phase,
                previous_notes=previous_notes,
                args=args,
                episodes=QUICK_EVAL_EPISODES,
                retention_episodes=QUICK_RETENTION_EPISODES,
                seed_base=quick_seed,
                label="screen 00",
            )

            if passes(baseline, baseline_retention, args, phase.notes):
                verified, verified_retention = _evaluate(
                    model,
                    device,
                    phase=phase,
                    previous_notes=previous_notes,
                    args=args,
                    episodes=args.eval_episodes,
                    retention_episodes=args.retention_episodes,
                    seed_base=full_seed,
                    label="verify",
                )
                if passes(verified, verified_retention, args, phase.notes):
                    _write("  status: already passed (verified)")
                    save_checkpoint(
                        checkpoint,
                        model,
                        optimizer,
                        args=args,
                        phase_index=phase_index,
                        phase=phase,
                        global_update=global_update,
                        probe=verified,
                    )
                    phase_bar.update(1)
                    continue

            best_probe = baseline
            best_retention = baseline_retention
            best_key = rank_key(baseline, baseline_retention)
            best_model = copy.deepcopy(model.state_dict())
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            focus = weak_bpms(baseline)
            phase_passed = False
            eval_every = _eval_interval(args, phase_index)

            update_bar = _bar(
                range(1, args.updates_per_phase + 1),
                total=args.updates_per_phase,
                desc=f"P{phase_index:02d} updates",
                unit="upd",
                leave=False,
                position=1,
            )
            for phase_update in update_bar:
                global_update += 1
                rollouts: list[core.Rollout] = []
                bpm_schedule = focused_training_bpm_schedule(
                    phase,
                    episodes=args.rollout_episodes,
                    points=args.train_bpm_points,
                    rng=rng,
                    focus_bpms=focus,
                )
                rollout_iter = _bar(
                    range(args.rollout_episodes),
                    total=args.rollout_episodes,
                    desc="rollouts",
                    unit="ep",
                    leave=False,
                    position=2,
                )
                for rollout_index in rollout_iter:
                    rollout_phase = phase
                    bpm_override: float | None = bpm_schedule[rollout_index]
                    if previous_notes and rng.random() < args.previous_stage_replay:
                        rollout_phase = core.full_vision_phase(args, rng.choice(previous_notes))
                        bpm_override = None
                    rollouts.append(
                        core.collect_rollout(
                            model,
                            device,
                            phase=rollout_phase,
                            control_dt=args.control_dt,
                            gamma=args.gamma,
                            rng=rng,
                            bpm_override=bpm_override,
                        )
                    )

                _, _, _, approx_kl, epochs_done = core.ppo_update(
                    model, optimizer, rollouts, args
                )
                rollout_hit = sum(r.hits for r in rollouts) / max(
                    1, sum(r.targets for r in rollouts)
                )
                rollout_reward = sum(r.reward for r in rollouts) / len(rollouts)
                rollout_overload = sum(int(r.overloaded) for r in rollouts)
                focus_text = "/".join(f"{b:g}" for b in focus) or "-"
                update_bar.set_postfix_str(
                    f"H={rollout_hit:.3f} R={rollout_reward:+.2f} O={rollout_overload} "
                    f"focus={focus_text} KL={approx_kl:.4f} e={epochs_done}",
                    refresh=True,
                )

                if phase_update % eval_every != 0 and phase_update != args.updates_per_phase:
                    continue

                probe, retention = _evaluate(
                    model,
                    device,
                    phase=phase,
                    previous_notes=previous_notes,
                    args=args,
                    episodes=QUICK_EVAL_EPISODES,
                    retention_episodes=QUICK_RETENTION_EPISODES,
                    seed_base=quick_seed,
                    label=f"screen {phase_update:02d}",
                )
                focus = weak_bpms(probe)
                key = rank_key(probe, retention)

                if key > best_key:
                    best_key = key
                    best_probe = probe
                    best_retention = retention
                    best_model = copy.deepcopy(model.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    _write(
                        f"  best screen X={_xacc(probe):.2f}% PP={_pp(probe):.1%} "
                        f"minX={_min_xacc(probe):.1f}% minPP={_min_pp(probe):.0%}"
                    )

                catastrophic = (
                    probe.hit_rate < best_probe.hit_rate - args.rollback_drop
                    or probe.full_rate < best_probe.full_rate - args.rollback_drop
                    or _min_hit(probe) < _min_hit(best_probe) - args.rollback_drop
                    or _min_full(probe) < _min_full(best_probe) - args.rollback_drop
                    or (best_probe.overloads == 0 and probe.overloads > 0)
                )
                if catastrophic:
                    model.load_state_dict(best_model)
                    optimizer.load_state_dict(best_optimizer)
                    focus = weak_bpms(best_probe)
                    for group in optimizer.param_groups:
                        group["lr"] = max(
                            args.min_lr, float(group["lr"]) * args.rollback_lr_factor
                        )
                    _write(
                        f"  rollback -> H={best_probe.hit_rate:.3f} "
                        f"F={best_probe.full_rate:.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    )
                    continue

                if passes(probe, retention, args, phase.notes):
                    verified, verified_retention = _evaluate(
                        model,
                        device,
                        phase=phase,
                        previous_notes=previous_notes,
                        args=args,
                        episodes=args.eval_episodes,
                        retention_episodes=args.retention_episodes,
                        seed_base=full_seed,
                        label="verify",
                    )
                    if passes(verified, verified_retention, args, phase.notes):
                        save_checkpoint(
                            checkpoint,
                            model,
                            optimizer,
                            args=args,
                            phase_index=phase_index,
                            phase=phase,
                            global_update=global_update,
                            probe=verified,
                        )
                        phase_passed = True
                        _write("  status: PASS (full verification)")
                        break
                    _write("  verify failed; continue training")

            update_bar.close()

            if not phase_passed:
                model.load_state_dict(best_model)
                optimizer.load_state_dict(best_optimizer)
                verified, verified_retention = _evaluate(
                    model,
                    device,
                    phase=phase,
                    previous_notes=previous_notes,
                    args=args,
                    episodes=args.eval_episodes,
                    retention_episodes=args.retention_episodes,
                    seed_base=full_seed,
                    label="final",
                )
                save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    args=args,
                    phase_index=phase_index,
                    phase=phase,
                    global_update=global_update,
                    probe=verified,
                )

                if passes(verified, verified_retention, args, phase.notes):
                    _write("  status: PASS at final verification")
                    phase_bar.update(1)
                    continue
                if ALLOW_PRECISION_SKIP and completion_passes(
                    verified, verified_retention, args, phase.notes
                ):
                    _write(
                        "  status: precision skip "
                        f"X={_xacc(verified):.2f}% PP={_pp(verified):.1%} "
                        f"minX={_min_xacc(verified):.1f}% minPP={_min_pp(verified):.0%}"
                    )
                    phase_bar.update(1)
                    continue
                _write(
                    "  status: STOP accuracy target unmet "
                    f"H={verified.hit_rate:.3f} F={verified.full_rate:.3f} "
                    f"X={_xacc(verified):.2f}% PP={_pp(verified):.1%} "
                    f"minX={_min_xacc(verified):.1f}% minPP={_min_pp(verified):.0%}"
                )
                break

            phase_bar.update(1)
    finally:
        phase_bar.close()

    _write()
    _write(f"best checkpoint: {checkpoint}")


def _pop_float_arg(name: str, default: float) -> float:
    prefix = name + "="
    for i, value in enumerate(tuple(sys.argv)):
        if value.startswith(prefix):
            sys.argv.remove(value)
            return float(value[len(prefix) :])
        if value == name:
            if i + 1 >= len(sys.argv):
                raise SystemExit(f"{name} requires a value")
            result = float(sys.argv[i + 1])
            del sys.argv[i : i + 2]
            return result
    return default


def _pop_int_arg(name: str, default: int) -> int:
    return int(_pop_float_arg(name, float(default)))


def _read_ui_args() -> None:
    global PROGRESS_ENABLED, ALLOW_PRECISION_SKIP
    global PRECISION_MIN_BPM_XACC, PRECISION_OVERALL_XACC
    global PRECISION_MIN_BPM_PP, PRECISION_OVERALL_PP
    global COMPLETION_MIN_BPM_HIT, COMPLETION_MIN_BPM_FULL
    global QUICK_EVAL_EPISODES, QUICK_RETENTION_EPISODES, LATE_EVAL_EVERY_UPDATES
    global WORST_BPM_FOCUS, WORST_BPM_COUNT

    if "--no-progress" in sys.argv:
        sys.argv.remove("--no-progress")
        PROGRESS_ENABLED = False
    elif not sys.stderr.isatty():
        PROGRESS_ENABLED = False

    if "--allow-precision-skip" in sys.argv:
        sys.argv.remove("--allow-precision-skip")
        ALLOW_PRECISION_SKIP = True

    PRECISION_MIN_BPM_XACC = _pop_float_arg(
        "--precision-min-bpm-xacc", PRECISION_MIN_BPM_XACC
    )
    PRECISION_OVERALL_XACC = _pop_float_arg(
        "--precision-overall-xacc", PRECISION_OVERALL_XACC
    )
    PRECISION_MIN_BPM_PP = _pop_float_arg("--precision-min-bpm-pp", PRECISION_MIN_BPM_PP)
    PRECISION_OVERALL_PP = _pop_float_arg("--precision-overall-pp", PRECISION_OVERALL_PP)
    COMPLETION_MIN_BPM_HIT = _pop_float_arg(
        "--completion-min-bpm-hit", COMPLETION_MIN_BPM_HIT
    )
    COMPLETION_MIN_BPM_FULL = _pop_float_arg(
        "--completion-min-bpm-full", COMPLETION_MIN_BPM_FULL
    )
    QUICK_EVAL_EPISODES = _pop_int_arg("--quick-eval-episodes", QUICK_EVAL_EPISODES)
    QUICK_RETENTION_EPISODES = _pop_int_arg(
        "--quick-retention-episodes", QUICK_RETENTION_EPISODES
    )
    LATE_EVAL_EVERY_UPDATES = _pop_int_arg(
        "--late-eval-every-updates", LATE_EVAL_EVERY_UPDATES
    )
    WORST_BPM_FOCUS = _pop_float_arg("--worst-bpm-focus", WORST_BPM_FOCUS)
    WORST_BPM_COUNT = _pop_int_arg("--worst-bpm-count", WORST_BPM_COUNT)

    if not 0.0 <= PRECISION_MIN_BPM_XACC <= 100.0:
        raise SystemExit("--precision-min-bpm-xacc must be in [0, 100]")
    if not 0.0 <= PRECISION_OVERALL_XACC <= 100.0:
        raise SystemExit("--precision-overall-xacc must be in [0, 100]")
    for name, value in (
        ("--precision-min-bpm-pp", PRECISION_MIN_BPM_PP),
        ("--precision-overall-pp", PRECISION_OVERALL_PP),
        ("--completion-min-bpm-hit", COMPLETION_MIN_BPM_HIT),
        ("--completion-min-bpm-full", COMPLETION_MIN_BPM_FULL),
        ("--worst-bpm-focus", WORST_BPM_FOCUS),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0, 1]")
    for name, value in (
        ("--quick-eval-episodes", QUICK_EVAL_EPISODES),
        ("--quick-retention-episodes", QUICK_RETENTION_EPISODES),
        ("--late-eval-every-updates", LATE_EVAL_EVERY_UPDATES),
        ("--worst-bpm-count", WORST_BPM_COUNT),
    ):
        if value <= 0:
            raise SystemExit(f"{name} must be positive")


def main() -> None:
    _read_ui_args()

    # The core keeps optimizer/checkpoint/CLI compatibility. The frontend owns
    # accuracy metrics, gates, ranking, adaptive probing, sampling, and progress UI.
    core.deterministic_probe = deterministic_probe
    core.completion_passes = completion_passes
    core.precision_passes = precision_passes
    core.passes = passes
    core.rank_key = rank_key
    core.save_checkpoint = save_checkpoint
    core.train = train_ui
    core.main()


if __name__ == "__main__":
    main()
