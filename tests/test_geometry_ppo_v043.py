from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_geometry_ppo_v043 as trainer  # noqa: E402


def _phase(name: str, low: float, high: float, *, clean: bool = True):
    vision = (
        trainer.core.clean_vision_config()
        if clean
        else trainer.core.PlanetVisionConfig(
            latency_s=0.050,
            latency_jitter_s=0.0,
            sample_period_s=1.0 / 60.0,
            position_noise_std=0.0,
            dropout_probability=0.0,
        )
    )
    return trainer.core.CurriculumPhase(
        name,
        1,
        low,
        high,
        80.0,
        80.0,
        vision,
    )


def test_phase_bc_is_limited_to_clean_p3_through_p6():
    assert trainer.phase_bc_eligible(_phase("bpm-145-240-clean", 145.0, 240.0))
    assert trainer.phase_bc_eligible(_phase("bpm-135-260-clean", 135.0, 260.0))
    assert trainer.phase_bc_eligible(_phase("bpm-125-280-clean", 125.0, 280.0))
    assert trainer.phase_bc_eligible(_phase("full-bpm-clean", 120.0, 300.0))

    assert not trainer.phase_bc_eligible(_phase("bpm-159-222-clean", 159.0, 222.0))
    assert not trainer.phase_bc_eligible(
        _phase("latency-sampling", 120.0, 300.0, clean=False)
    )


def test_phase_demonstrations_cover_both_edges_and_keep_student_input_14d():
    phase = _phase("bpm-135-260-clean", 135.0, 260.0)
    teacher, _ = trainer._teacher_for(0.010)
    demos, bpms = trainer.collect_phase_demonstrations(
        teacher,
        phase=phase,
        episodes=8,
        control_dt=0.010,
        device=torch.device("cpu"),
        seed=7,
    )

    assert len(demos) == 8
    assert 135.0 in bpms
    assert 260.0 in bpms
    assert all(135.0 <= bpm <= 260.0 for bpm in bpms)
    assert all(demo.observations.shape[1] == 14 for demo in demos)
    assert all(demo.actions.shape == (demo.observations.shape[0], 2) for demo in demos)


def test_optimizer_reset_after_bc_discards_adam_moments_and_restores_cli_lr():
    model = trainer.v038.PredictiveRecurrentActorCritic(input_dim=14, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    assert optimizer.state

    trainer._reset_optimizer_after_bc(optimizer, SimpleNamespace(lr=1e-4))

    assert not optimizer.state
    assert all(group["lr"] == 1e-4 for group in optimizer.param_groups)
