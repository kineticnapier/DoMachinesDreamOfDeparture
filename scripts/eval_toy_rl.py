from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - user-facing dependency message
    raise SystemExit(
        'PyTorch is required. Run: uv sync --extra dev --extra rl --inexact'
    ) from exc

from dmdod.ablation import CueAblationMode, CueAblator
from dmdod.perception import VisualCueConfig
from dmdod.rhythm_env import EpisodeStats, RhythmMotorEnv, RhythmObservation, make_regular_targets
from dmdod.toy_policy import ActorCritic, observation_tensor


@dataclass(frozen=True)
class EvalRun:
    stats: EpisodeStats
    timing_errors_ms: tuple[float, ...]


@dataclass(frozen=True)
class EvalSummary:
    episodes: int
    total_hits: int
    total_targets: int
    total_misses: int
    total_too_early: int
    overloads: int
    full_hits: int
    zero_too_early: int
    clean_clears: int
    timing_errors_ms: tuple[float, ...]

    @property
    def hit_rate(self) -> float:
        return self.total_hits / max(self.total_targets, 1)

    @property
    def mean_too_early(self) -> float:
        return self.total_too_early / max(self.episodes, 1)

    @property
    def mean_signed_error_ms(self) -> float | None:
        if not self.timing_errors_ms:
            return None
        return sum(self.timing_errors_ms) / len(self.timing_errors_ms)

    @property
    def mean_abs_error_ms(self) -> float | None:
        if not self.timing_errors_ms:
            return None
        return sum(abs(error) for error in self.timing_errors_ms) / len(self.timing_errors_ms)

    @property
    def p95_abs_error_ms(self) -> float | None:
        return p95_abs(list(self.timing_errors_ms))


def p95_abs(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(abs(value) for value in values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def perception_config_from_checkpoint(checkpoint: dict[str, object]) -> VisualCueConfig:
    raw = checkpoint.get("perception_config")
    if not isinstance(raw, dict):
        return VisualCueConfig()
    return VisualCueConfig(
        latency_s=float(raw.get("latency_s", 0.050)),
        width_s=float(raw.get("width_s", 0.090)),
        horizon_s=float(raw.get("horizon_s", 0.450)),
        latency_jitter_s=float(raw.get("latency_jitter_s", 0.0)),
        sample_period_s=float(raw.get("sample_period_s", 0.0)),
        amplitude_noise_std=float(raw.get("amplitude_noise_std", 0.0)),
        dropout_probability=float(raw.get("dropout_probability", 0.0)),
    )


def _policy_observation(
    observation: RhythmObservation,
    ablator: CueAblator,
) -> RhythmObservation:
    return RhythmObservation(observation.motor, ablator.transform(observation.cue))


def run_episode(
    model: ActorCritic,
    device: torch.device,
    *,
    bpm: float,
    notes: int,
    pattern: str,
    same_hand: bool,
    control_dt: float,
    start_s: float,
    cue_config: VisualCueConfig,
    perception_seed: int,
    cue_mode: CueAblationMode | str = CueAblationMode.NORMAL,
    cue_delay_s: float = 0.100,
    cue_seed: int = 0,
) -> EvalRun:
    targets = make_regular_targets(
        bpm=bpm,
        count=notes,
        start_s=start_s,
        pattern=pattern,
    )
    env = RhythmMotorEnv(
        targets,
        bpm=bpm,
        same_hand=same_hand,
        control_dt_s=control_dt,
        cue_config=cue_config,
        perception_seed=perception_seed,
    )
    ablator = CueAblator(
        cue_mode,
        control_dt_s=control_dt,
        delay_s=cue_delay_s,
        seed=cue_seed,
    )
    observation = _policy_observation(env.reset(), ablator)

    while True:
        x = observation_tensor(observation, device)
        action = model.deterministic_action(x)
        transition = env.step(action)
        observation = _policy_observation(transition.observation, ablator)
        if transition.done:
            break

    return EvalRun(env.stats, env.timing_errors_ms)


def summarize_runs(runs: list[EvalRun]) -> EvalSummary:
    all_errors = tuple(error for run in runs for error in run.timing_errors_ms)
    return EvalSummary(
        episodes=len(runs),
        total_hits=sum(run.stats.hits for run in runs),
        total_targets=sum(run.stats.targets for run in runs),
        total_misses=sum(run.stats.misses for run in runs),
        total_too_early=sum(run.stats.too_early_presses for run in runs),
        overloads=sum(int(run.stats.overloaded) for run in runs),
        full_hits=sum(
            run.stats.hits == run.stats.targets and not run.stats.overloaded
            for run in runs
        ),
        zero_too_early=sum(run.stats.too_early_presses == 0 for run in runs),
        clean_clears=sum(
            run.stats.hits == run.stats.targets
            and run.stats.too_early_presses == 0
            and not run.stats.overloaded
            for run in runs
        ),
        timing_errors_ms=all_errors,
    )


def print_single(label: str, run: EvalRun) -> None:
    stats = run.stats
    mae = "--" if stats.mean_abs_error_ms is None else f"{stats.mean_abs_error_ms:.2f} ms"
    p95 = p95_abs(list(run.timing_errors_ms))
    p95_text = "--" if p95 is None else f"{p95:.2f} ms"
    print(label)
    print(f"  hits:       {stats.hits}/{stats.targets}")
    print(f"  misses:     {stats.misses}")
    print(f"  Perfect:    {stats.perfects}")
    print(f"  E/LPerfect: {stats.early_late_perfects}")
    print(f"  Early/Late: {stats.early_late_hits}")
    print(f"  Too Early:  {stats.too_early_presses}")
    print(f"  overload:   {stats.overload_counter}/6{'  OVERLOAD!' if stats.overloaded else ''}")
    print(f"  MAE:        {mae}")
    print(f"  P95:        {p95_text}")
    print(f"  reward:     {stats.total_reward:.3f}")


def _fmt_ms(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def print_summary_row(label: str, summary: EvalSummary) -> None:
    print(
        f"  {label:<14} "
        f"hit={summary.hit_rate:6.3f}  "
        f"full={summary.full_hits:3d}/{summary.episodes:<3d}  "
        f"clean={summary.clean_clears:3d}/{summary.episodes:<3d}  "
        f"ovl={summary.overloads:3d}/{summary.episodes:<3d}  "
        f"early={summary.mean_too_early:5.2f}  "
        f"meanErr={_fmt_ms(summary.mean_signed_error_ms, signed=True):>6}ms  "
        f"MAE={_fmt_ms(summary.mean_abs_error_ms):>5}ms  "
        f"P95={_fmt_ms(summary.p95_abs_error_ms):>5}ms"
    )


def evaluate_runs(
    model: ActorCritic,
    device: torch.device,
    *,
    starts_s: list[float],
    bpm: float,
    notes: int,
    pattern: str,
    same_hand: bool,
    control_dt: float,
    cue_config: VisualCueConfig,
    cue_mode: CueAblationMode,
    cue_delay_s: float,
    seed: int,
) -> list[EvalRun]:
    return [
        run_episode(
            model,
            device,
            bpm=bpm,
            notes=notes,
            pattern=pattern,
            same_hand=same_hand,
            control_dt=control_dt,
            start_s=start_s,
            cue_config=cue_config,
            perception_seed=seed + i * 4099,
            cue_mode=cue_mode,
            cue_delay_s=cue_delay_s,
            cue_seed=seed + i * 1009,
        )
        for i, start_s in enumerate(starts_s)
    ]


def make_starts(
    *,
    episodes: int,
    base_start_s: float,
    jitter_ms: float,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    jitter_s = jitter_ms / 1000.0
    return [
        max(
            0.050,
            base_start_s + (rng.uniform(-jitter_s, jitter_s) if jitter_s > 0.0 else 0.0),
        )
        for _ in range(episodes)
    ]


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    hidden_dim = int(checkpoint.get("hidden_dim", state["backbone.0.weight"].shape[0]))
    model = ActorCritic(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    bpm = float(checkpoint.get("bpm", 180.0))
    bpm_min = float(checkpoint.get("bpm_min", bpm))
    bpm_max = float(checkpoint.get("bpm_max", bpm))
    notes = int(checkpoint.get("notes", 16))
    target_notes = int(checkpoint.get("target_notes", notes))
    pattern = str(checkpoint.get("pattern", "left"))
    control_dt = float(checkpoint.get("control_dt", 0.010))
    base_start_s = float(checkpoint.get("start_s", 0.750))
    same_hand = pattern != "alternate" or bool(checkpoint.get("same_hand", False))
    cue_config = perception_config_from_checkpoint(checkpoint)

    probe_targets = make_regular_targets(
        bpm=bpm,
        count=max(1, notes),
        start_s=base_start_s,
        pattern=pattern,
    )
    probe_env = RhythmMotorEnv(
        probe_targets,
        bpm=bpm,
        same_hand=same_hand,
        control_dt_s=control_dt,
        cue_config=cue_config,
        perception_seed=args.seed,
    )
    windows = probe_env.timing_windows

    print("=== Deterministic Toy Policy Evaluation ===")
    print(f"checkpoint: {checkpoint_path}")
    print(
        f"task: {bpm:g} BPM nominal, {notes} notes, pattern={pattern}, "
        f"control_dt={control_dt*1000:.1f} ms, start={base_start_s*1000:.0f} ms"
    )
    if abs(bpm_max - bpm_min) > 1e-12:
        print(f"trained BPM domain: {bpm_min:g}..{bpm_max:g}")
    if bool(checkpoint.get("curriculum", False)):
        stage_index = checkpoint.get("curriculum_stage_index", "?")
        print(f"curriculum checkpoint: stage {stage_index}, {notes}/{target_notes} notes")
    sample_text = (
        "continuous"
        if cue_config.sample_period_s <= 0.0
        else f"{1.0 / cue_config.sample_period_s:.0f} Hz"
    )
    print(
        "sensor: "
        f"latency={cue_config.latency_s*1000:.1f}±{cue_config.latency_jitter_s*1000:.1f} ms, "
        f"sample={sample_text}, noise={cue_config.amplitude_noise_std:.3f}, "
        f"dropout={cue_config.dropout_probability*100:.1f}%"
    )
    print("action: tanh(actor mean), no motor exploration noise")
    print(
        f"Normal timing: Perfect ±{windows.perfect_s*1000:.2f} ms, "
        f"E/L Perfect ±{windows.early_late_perfect_s*1000:.2f} ms, "
        f"Pass ±{windows.pass_s*1000:.2f} ms"
    )
    print("OVERLOAD: Too Early +2, valid hit -1, fail at 6")
    if int(checkpoint.get("format_version", 0)) < 2:
        print("note: checkpoint predates ADOFAI timing/OVERLOAD training; this is an out-of-distribution rules test")
    print()

    exact = run_episode(
        model,
        device,
        bpm=bpm,
        notes=notes,
        pattern=pattern,
        same_hand=same_hand,
        control_dt=control_dt,
        start_s=base_start_s,
        cue_config=cue_config,
        perception_seed=args.seed,
    )
    print_single("Exact checkpoint schedule:", exact)

    starts_s = make_starts(
        episodes=args.episodes,
        base_start_s=base_start_s,
        jitter_ms=args.start_jitter_ms,
        seed=args.seed,
    )
    runs = evaluate_runs(
        model,
        device,
        starts_s=starts_s,
        bpm=bpm,
        notes=notes,
        pattern=pattern,
        same_hand=same_hand,
        control_dt=control_dt,
        cue_config=cue_config,
        cue_mode=CueAblationMode.NORMAL,
        cue_delay_s=args.cue_delay_ms / 1000.0,
        seed=args.seed,
    )
    summary = summarize_runs(runs)

    print()
    print(
        f"Phase/sensor robustness: {args.episodes} episodes, "
        f"start offset ±{args.start_jitter_ms:g} ms around {base_start_s*1000:.0f} ms"
    )
    print(f"  hit rate:       {summary.total_hits}/{summary.total_targets} = {summary.hit_rate:.4f}")
    print(f"  mean hits:      {summary.total_hits/args.episodes:.3f}/{notes}")
    print(f"  mean misses:    {summary.total_misses/args.episodes:.3f}")
    print(f"  mean Too Early: {summary.mean_too_early:.3f}")
    print(f"  OVERLOAD runs:  {summary.overloads}/{args.episodes}")
    print(f"  full-hit runs:  {summary.full_hits}/{args.episodes}")
    print(f"  zero Too Early: {summary.zero_too_early}/{args.episodes}")
    print(f"  clean clears:   {summary.clean_clears}/{args.episodes}")
    print(f"  mean signed error: {'--' if summary.mean_signed_error_ms is None else f'{summary.mean_signed_error_ms:+.2f} ms'}")
    print(f"  MAE all hits:   {'--' if summary.mean_abs_error_ms is None else f'{summary.mean_abs_error_ms:.2f} ms'}")
    print(f"  P95 abs error:  {'--' if summary.p95_abs_error_ms is None else f'{summary.p95_abs_error_ms:.2f} ms'}")

    if args.cue_ablation:
        delay_s = args.cue_delay_ms / 1000.0
        print()
        print(
            f"Cue ablation: same {args.episodes} phase/sensor seeds; "
            f"delay=+{args.cue_delay_ms:g} ms"
        )
        print_summary_row("normal", summary)
        for mode, label in (
            (CueAblationMode.ZERO, "zero"),
            (CueAblationMode.RANDOM, "random"),
            (CueAblationMode.DELAY, f"delay+{args.cue_delay_ms:g}"),
        ):
            transformed = summarize_runs(
                evaluate_runs(
                    model,
                    device,
                    starts_s=starts_s,
                    bpm=bpm,
                    notes=notes,
                    pattern=pattern,
                    same_hand=same_hand,
                    control_dt=control_dt,
                    cue_config=cue_config,
                    cue_mode=mode,
                    cue_delay_s=delay_s,
                    seed=args.seed,
                )
            )
            print_summary_row(label, transformed)

    if args.bpm_sweep:
        sweep_low = bpm_min if args.bpm_sweep_min is None else args.bpm_sweep_min
        sweep_high = bpm_max if args.bpm_sweep_max is None else args.bpm_sweep_max
        if sweep_low <= 0.0 or sweep_high <= 0.0 or sweep_low > sweep_high:
            raise SystemExit("invalid BPM sweep range")
        points = max(1, args.bpm_sweep_points)
        if points == 1 or abs(sweep_high - sweep_low) < 1e-12:
            bpms = (float(sweep_low),)
        else:
            bpms = tuple(
                sweep_low + (sweep_high - sweep_low) * i / (points - 1)
                for i in range(points)
            )

        print()
        print(
            f"BPM sweep: {args.bpm_sweep_episodes} episodes per point, "
            f"same sensor model, phase jitter ±{args.start_jitter_ms:g} ms"
        )
        for index, sweep_bpm in enumerate(bpms):
            sweep_starts = make_starts(
                episodes=args.bpm_sweep_episodes,
                base_start_s=base_start_s,
                jitter_ms=args.start_jitter_ms,
                seed=args.seed + index * 100003,
            )
            sweep_summary = summarize_runs(
                evaluate_runs(
                    model,
                    device,
                    starts_s=sweep_starts,
                    bpm=sweep_bpm,
                    notes=notes,
                    pattern=pattern,
                    same_hand=same_hand,
                    control_dt=control_dt,
                    cue_config=cue_config,
                    cue_mode=CueAblationMode.NORMAL,
                    cue_delay_s=args.cue_delay_ms / 1000.0,
                    seed=args.seed + index * 100003,
                )
            )
            print_summary_row(f"{sweep_bpm:g} BPM", sweep_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a toy motor policy without motor exploration noise.")
    parser.add_argument("--checkpoint", default="checkpoints/toy_policy.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--start-jitter-ms",
        type=float,
        default=100.0,
        help="uniformly vary chart start time around the checkpoint's training start",
    )
    parser.add_argument(
        "--cue-ablation",
        action="store_true",
        help="compare normal, zero, random, and delayed visual cues on the same seeds",
    )
    parser.add_argument(
        "--cue-delay-ms",
        type=float,
        default=100.0,
        help="extra policy-side visual delay used by the delay cue-ablation condition",
    )
    parser.add_argument("--bpm-sweep", action="store_true")
    parser.add_argument("--bpm-sweep-min", type=float, default=None)
    parser.add_argument("--bpm-sweep-max", type=float, default=None)
    parser.add_argument("--bpm-sweep-points", type=int, default=5)
    parser.add_argument("--bpm-sweep-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.episodes <= 0 or args.bpm_sweep_episodes <= 0:
        parser.error("episode counts must be positive")
    if args.start_jitter_ms < 0.0:
        parser.error("start-jitter-ms must be non-negative")
    if args.cue_delay_ms < 0.0:
        parser.error("cue-delay-ms must be non-negative")
    if args.bpm_sweep_points <= 0:
        parser.error("bpm-sweep-points must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
