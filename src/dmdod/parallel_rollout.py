from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParallelRolloutSpec:
    index: int
    notes: int
    bpm: float
    start_s: float
    control_dt: float
    vision: dict[str, float]
    env_seed: int
    policy_seed: int
    gamma: float


def _vision_config(values: dict[str, float]):
    from .planet_perception import PlanetVisionConfig

    return PlanetVisionConfig(
        latency_s=float(values["latency_s"]),
        latency_jitter_s=float(values["latency_jitter_s"]),
        sample_period_s=float(values["sample_period_s"]),
        position_noise_std=float(values["position_noise_std"]),
        dropout_probability=float(values["dropout_probability"]),
    )


def collect_rollout_batch(
    model_state: dict[str, Any],
    hidden_dim: int,
    specs: list[ParallelRolloutSpec],
    *,
    too_early_penalty: float = 1.0,
) -> list[dict[str, Any]]:
    """Collect a batch of stochastic rollouts in one CPU worker.

    The worker reconstructs one policy from the supplied frozen state, then
    reuses it for every episode in ``specs``.  Returning plain Python lists
    keeps Windows ``spawn`` transport independent of torch shared-memory file
    handles.  PPO gradients are computed only in the parent process.
    """

    import torch

    from .motion_geometry_env import MotionGeometryEnv
    from .predictive_recurrent_policy import PredictiveRecurrentActorCritic
    from .rhythm_env import RewardConfig, make_regular_targets
    from .toy_policy import MOTION_GEOMETRY_INPUT_DIM, discounted_returns, observation_tensor

    # Several workers each running a tiny network are faster when every worker
    # stays single-threaded instead of nesting BLAS/OpenMP pools.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = torch.device("cpu")
    model = PredictiveRecurrentActorCritic(
        input_dim=MOTION_GEOMETRY_INPUT_DIM,
        hidden_dim=int(hidden_dim),
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    results: list[dict[str, Any]] = []
    for spec in specs:
        torch.manual_seed(int(spec.policy_seed))
        targets = make_regular_targets(
            bpm=float(spec.bpm),
            count=int(spec.notes),
            start_s=float(spec.start_s),
            pattern="left",
        )
        env = MotionGeometryEnv(
            targets,
            bpm=float(spec.bpm),
            same_hand=True,
            control_dt_s=float(spec.control_dt),
            vision_config=_vision_config(spec.vision),
            perception_seed=int(spec.env_seed),
            reward_config=RewardConfig(too_early_penalty=float(too_early_penalty)),
        )
        observation = env.reset()
        state = model.initial_state(device)
        observations = []
        latents = []
        old_log_probs = []
        old_values = []
        rewards: list[float] = []

        with torch.no_grad():
            while True:
                x = observation_tensor(observation, device)
                action, latent, log_prob, value, _, state = model.sample_action_latent(x, state)
                transition = env.step(action)
                observations.append(x.detach())
                latents.append(latent.detach())
                old_log_probs.append(log_prob.detach())
                old_values.append(value.detach())
                rewards.append(float(transition.reward))
                observation = transition.observation
                if transition.done:
                    break

        returns = discounted_returns(rewards, float(spec.gamma), device)
        old_values_t = torch.stack(old_values)
        advantages = returns - old_values_t
        stats = env.stats
        results.append(
            {
                "index": int(spec.index),
                "observations": torch.stack(observations).cpu().tolist(),
                "latents": torch.stack(latents).cpu().tolist(),
                "old_log_probs": torch.stack(old_log_probs).cpu().tolist(),
                "old_values": old_values_t.cpu().tolist(),
                "returns": returns.cpu().tolist(),
                "advantages": advantages.cpu().tolist(),
                "hits": int(stats.hits),
                "targets": int(stats.targets),
                "too_early": int(stats.too_early_presses),
                "overloaded": bool(stats.overloaded),
                "reward": float(stats.total_reward),
            }
        )

    return results
