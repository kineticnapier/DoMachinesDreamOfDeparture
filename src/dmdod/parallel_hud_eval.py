from __future__ import annotations

"""Process-safe HUD policy evaluation helpers.

Each worker evaluates one fixed policy state on a batch of chart segments.  The
training process uses this to evaluate trust-region alphas in parallel without
changing any guard, reward, or checkpoint semantics.
"""

from typing import Iterable

import torch

from .real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG
from .real_chart_hud import DiagnosticHudRealChartMotorEnv
from .real_chart_hud_features import HUD_REAL_CHART_INPUT_DIM, encode_hud_real_chart_observation
from .recurrent_policy import RecurrentActorCritic


def _single_thread_torch() -> None:
    """Prevent six worker processes from each spawning a full BLAS thread pool."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only allows changing inter-op threads before parallel work has
        # started. A freshly spawned worker normally takes the fast path above.
        pass


def evaluate_hud_state_on_segments(
    state_dict: dict[str, torch.Tensor],
    hidden_dim: int,
    segments: Iterable,
    same_hand: bool,
    control_dt_s: float,
) -> list[tuple[object, int]]:
    """Evaluate one deterministic HUD policy on several segments in one process.

    The return value is intentionally made only of picklable simulator stats and
    integers. The parent reconstructs its historical ``StudentEvalResult``
    wrapper, keeping this module independent of trainer scripts.
    """

    _single_thread_torch()
    device = torch.device("cpu")
    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=int(hidden_dim),
        initial_log_std=-1.20,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    results: list[tuple[object, int]] = []
    for segment in segments:
        env = DiagnosticHudRealChartMotorEnv(
            segment,
            same_hand=bool(same_hand),
            control_dt_s=float(control_dt_s),
            behind_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.behind_floors,
            ahead_floors=DEFAULT_REAL_CHART_FEATURE_CONFIG.ahead_floors,
        )
        observation = env.reset()
        state = model.initial_state(device)
        max_steps = int((segment.duration_s + 2.0) / control_dt_s) + 200

        with torch.no_grad():
            for _ in range(max_steps):
                x = torch.tensor(
                    encode_hud_real_chart_observation(observation),
                    dtype=torch.float32,
                    device=device,
                )
                action, state = model.deterministic_action(x, state)
                step = env.step(action)
                observation = step.observation
                if step.done:
                    break
            else:
                raise RuntimeError("parallel HUD policy evaluation exceeded step budget")

        results.append((env.stats, int(env.physical_keydowns)))

    return results
