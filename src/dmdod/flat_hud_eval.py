from __future__ import annotations

"""Process-safe single-segment HUD policy evaluation.

This is the fine-grained counterpart to ``parallel_hud_eval``.  One worker task
contains exactly one (policy state, chart segment) pair, allowing the trainer to
flatten six trust-region alphas times train/validation/anchor segments into one
shared process queue without changing gameplay or guard semantics.
"""

import torch

from .real_chart_features import DEFAULT_REAL_CHART_FEATURE_CONFIG
from .real_chart_hud import DiagnosticHudRealChartMotorEnv
from .real_chart_hud_features import HUD_REAL_CHART_INPUT_DIM, encode_hud_real_chart_observation
from .recurrent_policy import RecurrentActorCritic


def _single_thread_torch() -> None:
    """Keep each process to one PyTorch thread; parallelism lives across tasks."""

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def evaluate_hud_state_on_segment(
    state_dict: dict[str, torch.Tensor],
    hidden_dim: int,
    segment,
    same_hand: bool,
    control_dt_s: float,
) -> tuple[object, int]:
    """Evaluate one deterministic HUD policy state on exactly one segment."""

    _single_thread_torch()
    device = torch.device("cpu")
    model = RecurrentActorCritic(
        input_dim=HUD_REAL_CHART_INPUT_DIM,
        hidden_dim=int(hidden_dim),
        initial_log_std=-1.20,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

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
            raise RuntimeError("flat parallel HUD policy evaluation exceeded step budget")

    return env.stats, int(env.physical_keydowns)
