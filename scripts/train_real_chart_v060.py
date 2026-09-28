from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch

import train_real_chart_v054 as v054
import train_real_chart_v055 as v055
import train_real_chart_v056 as v056
import train_real_chart_v057 as v057
import train_real_chart_v058 as v058
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import load_compiled_adofai
from dmdod.finger_agnostic_teacher import finger_agnostic_lead_action
from dmdod.privileged_teacher import calibrate_single_press_lead
from dmdod.real_chart_features import (
    DEFAULT_REAL_CHART_FEATURE_CONFIG,
    REAL_CHART_INPUT_DIM,
    encode_real_chart_observation,
)
from dmdod.recurrent_policy import RecurrentActorCritic


TRAINER_VERSION = "0.6.0-finger-agnostic-trust-region-dagger"
CHECKPOINT_FORMAT_VERSION = 8
DEFAULT_CHECKPOINT = "checkpoints/real_chart_v060_finger_agnostic.pt"


def _teacher_action(env, observation, lead_s: float, preferred_action=None):
    target = env.privileged_next_target()
    if target is None:
        return finger_agnostic_lead_action(
            now_s=env.privileged_episode_time_s(),
            target_time_s=float("inf"),
            lead_s=lead_s,
            motor=observation.motor,
            preferred_action=preferred_action,
        )
    return finger_agnostic_lead_action(
        now_s=env.privileged_episode_time_s(),
        target_time_s=target.episode_time_s,
        lead_s=lead_s,
        motor=observation.motor,
        preferred_action=preferred_action,
    )


def _collect_expert_sequence(
    segment,
    *,
    lead_s: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
):
    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    observations: list[tuple[float, ...]] = []
    actions: list[tuple[float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    for _ in range(max_steps):
        action = _teacher_action(env, observation, lead_s)
        observations.append(encode_real_chart_observation(observation))
        actions.append((action.left, action.right))
        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    else:
        raise RuntimeError("finger-agnostic teacher episode exceeded step budget")

    return (
        torch.tensor(observations, dtype=torch.float32, device=device),
        torch.tensor(actions, dtype=torch.float32, device=device),
        v054.StudentEvalResult(env.stats, env.physical_keydowns),
    )


def _collect_mixture_rollout(
    model: RecurrentActorCritic,
    segment,
    *,
    lead_s: float,
    teacher_fraction: float,
    same_hand: bool,
    control_dt_s: float,
    device: torch.device,
    source: str,
    seed: int,
    press_recovery_cap: int,
) -> v056.StableRollout:
    if not 0.0 <= teacher_fraction <= 1.0:
        raise ValueError("teacher_fraction must be in [0, 1]")

    env = v054.DiagnosticRealChartMotorEnv(
        segment,
        same_hand=same_hand,
        control_dt_s=control_dt_s,
        behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
        ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
    )
    observation = env.reset()
    state = model.initial_state(device)
    rng = random.Random(seed)
    observations: list[tuple[float, ...]] = []
    labels: list[tuple[float, float]] = []

    max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200
    with torch.no_grad():
        for _ in range(max_steps):
            encoded = encode_real_chart_observation(observation)
            x = torch.tensor(encoded, dtype=torch.float32, device=device)
            student, state = model.deterministic_action(x, state)
            teacher = _teacher_action(
                env,
                observation,
                lead_s,
                preferred_action=student,
            )

            observations.append(encoded)
            labels.append((teacher.left, teacher.right))
            applied = teacher if rng.random() < teacher_fraction else student
            step = env.step(applied)
            observation = step.observation
            if step.done:
                break
        else:
            raise RuntimeError("finger-agnostic DAgger rollout exceeded step budget")

    sequence = v055.DAggerSequence(
        observations=torch.tensor(observations, dtype=torch.float32, device=device),
        teacher_actions=torch.tensor(labels, dtype=torch.float32, device=device),
        source=source,
    )
    stable = v056._make_stable_sequence(
        sequence,
        press_recovery_cap=press_recovery_cap,
        expert=False,
    )
    return v056.StableRollout(
        sequence=stable,
        evaluation=v054.StudentEvalResult(env.stats, env.physical_keydowns),
        teacher_fraction=teacher_fraction,
    )


def _load_resume(
    model: RecurrentActorCritic,
    path: Path,
    *,
    hidden_dim: int,
    device: torch.device,
) -> dict:
    payload = torch.load(path, map_location=device)
    if int(payload.get("format_version", -1)) != CHECKPOINT_FORMAT_VERSION:
        raise SystemExit("v0.6.0 resumes only from a v0.6.0 finger-agnostic checkpoint")
    if int(payload.get("input_dim", -1)) != REAL_CHART_INPUT_DIM:
        raise SystemExit("checkpoint input dimension does not match current encoder")
    if int(payload.get("hidden_dim", -1)) != hidden_dim:
        raise SystemExit("checkpoint hidden size does not match --hidden")
    model.load_state_dict(payload["model_state"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Finger-agnostic real-chart trust-region DAgger. Notes are not assigned "
            "to left/right fingers; DAgger labels preserve a student's valid finger choice."
        )
    )
    parser.add_argument("chart")
    parser.add_argument("--train-start", type=float, default=0.0)
    parser.add_argument("--train-end", type=float, default=30.0)
    parser.add_argument("--sight-start", type=float, default=None)
    parser.add_argument("--sight-end", type=float, default=None)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--round-epochs", type=int, default=4)
    parser.add_argument("--bootstrap-epochs", type=int, default=28)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bootstrap-lr", type=float, default=3e-4)
    parser.add_argument("--chunk-steps", type=int, default=192)
    parser.add_argument("--control-dt", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--press-recovery-cap", type=int, default=v056.DEFAULT_PRESS_RECOVERY_CAP)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cross-hand", action="store_true")
    args = parser.parse_args()

    if args.train_end <= args.train_start:
        raise SystemExit("--train-end must be greater than --train-start")
    if args.rounds <= 0 or args.round_epochs <= 0 or args.bootstrap_epochs <= 0:
        raise SystemExit("round counts and epoch counts must be positive")
    if args.hidden <= 0 or args.chunk_steps <= 0 or args.press_recovery_cap <= 0:
        raise SystemExit("hidden/chunk-steps/press-recovery-cap must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    same_hand = not args.cross_hand
    compiled = load_compiled_adofai(args.chart)
    train_end = min(args.train_end, compiled.duration_s)
    train_segment = build_playable_segment(compiled, start_s=args.train_start, end_s=train_end)
    if not train_segment.targets:
        raise SystemExit("training segment contains no playable targets")

    sight_start = args.train_end if args.sight_start is None else args.sight_start
    sight_end = (
        sight_start + (args.train_end - args.train_start)
        if args.sight_end is None
        else args.sight_end
    )
    sight_end = min(sight_end, compiled.duration_s)
    sight_segment = build_playable_segment(compiled, start_s=sight_start, end_s=sight_end)
    if not sight_segment.targets:
        raise SystemExit("sight-read segment contains no playable targets")

    calibration = calibrate_single_press_lead(
        control_dt_s=args.control_dt,
        same_hand=same_hand,
    )
    print("=== DMDOD / Real Chart Student v0.6.0 Finger-Agnostic ===")
    print(
        f"chart={args.chart}\n"
        f"train={args.train_start:g}..{train_end:g}s targets={len(train_segment.targets)} | "
        f"sight={sight_start:g}..{sight_end:g}s targets={len(sight_segment.targets)}"
    )
    print(
        f"input={REAL_CHART_INPUT_DIM}D visible-only hidden={args.hidden} "
        f"lead={calibration.lead_s * 1000.0:.1f}ms control={args.control_dt * 1000.0:.1f}ms "
        f"rounds={args.rounds}x{args.round_epochs} proposal-lr={args.lr:g}"
    )
    print(
        "teacher=finger-agnostic (no ordinal parity) | "
        "student positive finger preference preserved during DAgger"
    )
    print(
        "trust alphas=" + "/".join(f"{alpha:g}" for alpha in v058.DEFAULT_TRUST_ALPHAS)
        + " | gameplay guard at every alpha | accuracy-first selection"
    )

    expert_x, expert_y, teacher_eval = _collect_expert_sequence(
        train_segment,
        lead_s=calibration.lead_s,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    expert_sequence = v056._make_stable_sequence(
        v055.DAggerSequence(expert_x, expert_y, "finger-agnostic-expert"),
        press_recovery_cap=args.press_recovery_cap,
        expert=True,
    )
    print(v054._format_eval("teacher train", teacher_eval))
    press, release, neutral = v054.base._action_frame_counts(expert_y)
    print(
        f"expert dataset frames={expert_sequence.frames} "
        f"labels(P/R/N)={press}/{release}/{neutral}"
    )

    model = RecurrentActorCritic(
        input_dim=REAL_CHART_INPUT_DIM,
        hidden_dim=args.hidden,
        initial_log_std=-1.20,
    ).to(device)
    checkpoint_path = Path(args.checkpoint)

    if args.resume:
        if not checkpoint_path.exists():
            raise SystemExit(f"checkpoint not found: {checkpoint_path}")
        _load_resume(model, checkpoint_path, hidden_dim=args.hidden, device=device)
        print(f"resume={checkpoint_path}")
    else:
        # Old v0.5.x checkpoints learned an alternating ordinal->finger teacher.
        # Start clean so that fixed fingering cannot leak into this experiment.
        print(f"bootstrap=fresh finger-agnostic expert BC epochs={args.bootstrap_epochs}")
        v054._train_bc(
            model,
            expert_x,
            expert_y,
            epochs=args.bootstrap_epochs,
            learning_rate=args.bootstrap_lr,
            chunk_steps=args.chunk_steps,
        )

    best_eval = v054._evaluate_student(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    best_state = copy.deepcopy(model.state_dict())
    print(v054._format_eval("baseline practiced", best_eval))

    replay: list[v056.StableSequence] = []
    round_history: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        model.load_state_dict(best_state)
        beta = v056._mixture_beta(round_index)
        rollout = _collect_mixture_rollout(
            model,
            train_segment,
            lead_s=calibration.lead_s,
            teacher_fraction=beta,
            same_hand=same_hand,
            control_dt_s=args.control_dt,
            device=device,
            source=f"finger-agnostic-round-{round_index}",
            seed=args.seed * 1000 + round_index,
            press_recovery_cap=args.press_recovery_cap,
        )
        v056._print_rollout(f"finger {round_index:02d} rollout", rollout)

        replay.append(rollout.sequence)
        replay = v056._trim_student_replay(replay, max_frames=expert_sequence.frames)
        replay_frames = v056._student_replay_frames(replay)
        expert_share = expert_sequence.frames / max(1, expert_sequence.frames + replay_frames)
        sequences = [expert_sequence, *replay]
        print(
            f"replay expert={expert_sequence.frames} student={replay_frames} "
            f"expert-share={expert_share * 100.0:.1f}% trajectories={len(sequences)}"
        )

        accepted_epochs = 0
        epoch_history: list[dict] = []
        for epoch_index in range(1, args.round_epochs + 1):
            model.load_state_dict(best_state)
            base_state = copy.deepcopy(best_state)
            optimizer = v057._new_optimizer(model, args.lr)
            loss = v057._train_one_epoch(
                model,
                sequences,
                optimizer=optimizer,
                chunk_steps=args.chunk_steps,
                reverse_order=not bool(epoch_index & 1),
            )
            proposal_state = copy.deepcopy(model.state_dict())

            choice, candidates, chosen_state = v058._evaluate_line_search(
                model,
                base_state=base_state,
                proposal_state=proposal_state,
                best_eval=best_eval,
                alphas=v058.DEFAULT_TRUST_ALPHAS,
                train_segment=train_segment,
                same_hand=same_hand,
                control_dt_s=args.control_dt,
                device=device,
                label_prefix=f"finger {round_index:02d} epoch {epoch_index:02d}",
            )

            if choice.accepted and choice.evaluation is not None and chosen_state is not None:
                accepted_epochs += 1
                best_eval = choice.evaluation
                best_state = copy.deepcopy(chosen_state)
                print(
                    f"finger {round_index:02d} epoch {epoch_index:02d}: "
                    f"ACCEPT alpha={choice.alpha:g} loss={loss:.6f} "
                    f"best=H{best_eval.stats.hits}/{best_eval.stats.targets} "
                    f"X{best_eval.stats.x_accuracy_percent:.2f}% "
                    f"PP{best_eval.stats.perfect_rate * 100.0:.1f}% "
                    f"early={best_eval.stats.too_early_presses}"
                )
            else:
                model.load_state_dict(best_state)
                print(
                    f"finger {round_index:02d} epoch {epoch_index:02d}: "
                    f"ROLLBACK all alphas loss={loss:.6f}"
                )

            epoch_history.append(
                {
                    "epoch": epoch_index,
                    "proposal_loss": loss,
                    "accepted": choice.accepted,
                    "alpha": choice.alpha,
                    "candidates": [
                        {
                            "alpha": candidate.alpha,
                            "hits": candidate.evaluation.stats.hits,
                            "xacc": candidate.evaluation.stats.x_accuracy_percent,
                            "pp": candidate.evaluation.stats.perfect_rate,
                            "early": candidate.evaluation.stats.too_early_presses,
                            "overloaded": candidate.evaluation.stats.overloaded,
                            "guard": candidate.decision.reason,
                        }
                        for candidate in candidates
                    ],
                }
            )

        print(
            f"finger {round_index:02d} summary: accepted-epochs={accepted_epochs}/"
            f"{args.round_epochs} best=H{best_eval.stats.hits}/{best_eval.stats.targets} "
            f"X{best_eval.stats.x_accuracy_percent:.2f}% "
            f"PP{best_eval.stats.perfect_rate * 100.0:.1f}% "
            f"early={best_eval.stats.too_early_presses}"
        )
        round_history.append(
            {
                "round": round_index,
                "teacher_fraction": beta,
                "rollout_frames": rollout.sequence.frames,
                "replay_frames": replay_frames,
                "expert_share": expert_share,
                "accepted_epochs": accepted_epochs,
                "epochs": epoch_history,
            }
        )

    model.load_state_dict(best_state)
    startup = v054.base._startup_probe(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    practiced = v054._evaluate_student(
        model,
        train_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    sight = v054._evaluate_student(
        model,
        sight_segment,
        same_hand=same_hand,
        control_dt_s=args.control_dt,
        device=device,
    )
    print(
        f"startup probe: {'PASS' if startup.safe else 'FAIL'} "
        f"early={startup.early_presses} overload={startup.overloaded} "
        f"max_push={startup.max_positive_action:+.3f} until={startup.until_s:.3f}s"
    )
    print(v054._format_eval("best student practiced", practiced))
    print(v054._format_eval("best student sight-read", sight))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "trainer_version": TRAINER_VERSION,
            "input_dim": REAL_CHART_INPUT_DIM,
            "hidden_dim": args.hidden,
            "model_state": best_state,
            "chart": str(args.chart),
            "train_start": args.train_start,
            "train_end": train_end,
            "feature_config": {
                "behind_floors": DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
                "ahead_floors": DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
            },
            "finger_agnostic": {
                "old_v05_bootstrap_used": False,
                "student_preference_preserved": True,
                "trust_alphas": list(v058.DEFAULT_TRUST_ALPHAS),
                "mixture_betas": list(v056.DEFAULT_MIXTURE_BETAS),
                "press_recovery_cap": args.press_recovery_cap,
                "round_history": round_history,
            },
        },
        checkpoint_path,
    )
    print(f"checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
