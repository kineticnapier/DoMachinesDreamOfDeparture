from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover - optional RL dependency
    raise ImportError(
        "PyTorch is required for dmdod.toy_policy. "
        "Install the rl extra before importing this module."
    ) from exc

from .geometry_rhythm_env import GeometryRhythmObservation
from .motion_geometry_env import MotionGeometryObservation
from .motor_env import MotorAction
from .pattern_geometry_env import PatternGeometryObservation
from .rhythm_env import RhythmObservation


GAUSSIAN_INPUT_DIM = 8
GEOMETRY_INPUT_DIM = 10
MOTION_GEOMETRY_INPUT_DIM = 14
PATTERN_GEOMETRY_INPUT_DIM = 22


def _motor_values(
    observation: (
        RhythmObservation
        | GeometryRhythmObservation
        | MotionGeometryObservation
        | PatternGeometryObservation
    ),
) -> list[float]:
    m = observation.motor
    # Fixed values here are unit conversions / broad physical scales, not chart
    # timing information. Exact simulator/chart time remains unavailable.
    return [
        m.left_position_m / 0.006,
        m.right_position_m / 0.006,
        m.left_velocity_m_s / 1.0,
        m.right_velocity_m_s / 1.0,
        1.0 if m.left_pressed else 0.0,
        1.0 if m.right_pressed else 0.0,
    ]


def observation_tensor(
    observation: (
        RhythmObservation
        | GeometryRhythmObservation
        | MotionGeometryObservation
        | PatternGeometryObservation
    ),
    device: torch.device,
) -> torch.Tensor:
    """Convert only agent-visible observations into the policy input tensor."""

    values = _motor_values(observation)
    if isinstance(observation, PatternGeometryObservation):
        g = observation.geometry
        p = observation.pattern
        values.extend([g.orbit_x, g.orbit_y, g.next_x, g.next_y])
        values.extend(
            [
                p.delta_orbit_x,
                p.delta_orbit_y,
                p.delta_next_x,
                p.delta_next_y,
                p.shared_timing_correction,
                p.shared_action_left,
                p.shared_action_right,
                p.shared_confidence,
                p.chart_timing_correction,
                p.chart_action_left,
                p.chart_action_right,
                p.chart_confidence,
            ]
        )
    elif isinstance(observation, MotionGeometryObservation):
        g = observation.geometry
        m = observation.motion
        values.extend([g.orbit_x, g.orbit_y, g.next_x, g.next_y])
        values.extend(
            [
                m.delta_orbit_x,
                m.delta_orbit_y,
                m.delta_next_x,
                m.delta_next_y,
            ]
        )
    elif isinstance(observation, GeometryRhythmObservation):
        g = observation.geometry
        values.extend([g.orbit_x, g.orbit_y, g.next_x, g.next_y])
    else:
        values.extend([observation.cue.left, observation.cue.right])
    return torch.tensor(values, dtype=torch.float32, device=device)


class ActorCritic(nn.Module):
    """Small continuous-action actor-critic used by the first RL tasks."""

    def __init__(
        self,
        input_dim: int = GAUSSIAN_INPUT_DIM,
        hidden_dim: int = 64,
        initial_log_std: float = -1.20,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        # The first version started at log_std=-0.35 (sigma ~= 0.70), which
        # produced so much random motion that many episodes hit OVERLOAD before
        # the first cue. Start at sigma ~= 0.30 instead; log_std remains
        # learnable and entropy regularization may still increase exploration.
        self.log_std = nn.Parameter(torch.full((2,), float(initial_log_std)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mean = self.actor_mean(h)
        value = self.critic(h).squeeze(-1)
        std = self.log_std.exp().clamp(0.08, 1.5)
        return mean, std, value

    def sample_action(self, x: torch.Tensor) -> tuple[MotorAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a stochastic action for policy-gradient training."""

        mean, std, value = self(x)
        dist = Normal(mean, std)
        latent = dist.sample()
        squashed = torch.tanh(latent)
        # Change-of-variables correction for tanh. latent/action are sampled
        # without a reparameterized path; policy gradients flow through log_prob.
        log_prob = dist.log_prob(latent).sum() - torch.log(1.0 - squashed.square() + 1e-6).sum()
        entropy = dist.entropy().sum()
        action = MotorAction(float(squashed[0].item()), float(squashed[1].item()))
        return action, log_prob, value, entropy

    @torch.no_grad()
    def deterministic_action(self, x: torch.Tensor) -> MotorAction:
        """Use tanh(actor mean) with no sampling/exploration noise."""

        mean, _, _ = self(x)
        squashed = torch.tanh(mean)
        return MotorAction(float(squashed[0].item()), float(squashed[1].item()))


def discounted_returns(rewards: list[float], gamma: float, device: torch.device) -> torch.Tensor:
    running = 0.0
    result: list[float] = []
    for reward in reversed(rewards):
        running = reward + gamma * running
        result.append(running)
    result.reverse()
    return torch.tensor(result, dtype=torch.float32, device=device)
