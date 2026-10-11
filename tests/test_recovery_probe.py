from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.training.config import load_training_config
from dmdod.training.real_chart import NamedSegment, evaluate_role_continuous
from dmdod.training.recovery_probe import (
    format_recovery_probe, run_teacher_takeover_probe,
)


class IdlePolicy:
    key_count = 4

    def eval(self):
        return self

    def initial_state(self, device):
        return torch.zeros(1, device=device)

    def forward_step(self, x, state):
        return torch.zeros(4, device=x.device), None, None, state


def _named():
    chart = parse_adofai_text(
        '{"angleData":[0,90,0,0],"settings":{"bpm":120,"pitch":100,'
        '"countdownTicks":0,"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    segment = build_playable_segment(
        compiled, start_s=0.0, end_s=compiled.duration_s,
    )
    return NamedSegment("validation", "tiny", "sha", 0.0, segment.duration_s, segment)


def _run():
    model = IdlePolicy()
    named = _named()
    traces = []
    results = evaluate_role_continuous(
        model, [named], label="baseline",
        control_dt_s=0.01, physics_dt_s=0.001,
        device=torch.device("cpu"), verbose=False,
        trace_collector=traces,
    )
    report = run_teacher_takeover_probe(
        model, named, alpha=0.125, lead_s=0.044,
        control_dt_s=0.01, physics_dt_s=0.001,
        device=torch.device("cpu"), expected_trace=traces[0],
        takeover_times_s=(0.01, 0.02),
    )
    return report, traces[0], named


def test_teacher_takeover_replays_student_and_does_not_mutate_checkpoint():
    report, trace, _ = _run()
    assert report.student_hits == trace.termination.hits
    assert report.student_keydowns == trace.termination.keydowns
    assert report.student_end_s == pytest.approx(trace.termination.time_s)
    assert len(report.takeovers) == 2
    assert report.skipped_times_s == ()
    assert [r.actual_time_s for r in report.takeovers] == pytest.approx([0.01, 0.02])
    assert all(r.teacher_end_s >= r.actual_time_s for r in report.takeovers)
    assert "diagnostic only" in "\n".join(format_recovery_probe(report))
    assert "fresh reservations" in "\n".join(format_recovery_probe(report))


def test_teacher_takeover_rejects_nonreproducible_student():
    _, reference, named = _run()
    bad = replace(
        reference,
        termination=replace(reference.termination, hits=reference.termination.hits + 1),
    )
    with pytest.raises(RuntimeError, match="reproduction FAIL"):
        run_teacher_takeover_probe(
            IdlePolicy(), named, alpha=0.125, lead_s=0.044,
            control_dt_s=0.01, physics_dt_s=0.001,
            device=torch.device("cpu"), expected_trace=bad,
            takeover_times_s=(0.01,),
        )


@pytest.mark.parametrize("times", [
    (), (0.0,), (0.015,), (0.02, 0.01), (float("nan"),),
])
def test_teacher_takeover_validates_snapshot_schedule(times):
    _, reference, named = _run()
    with pytest.raises(ValueError, match="takeover times"):
        run_teacher_takeover_probe(
            IdlePolicy(), named, alpha=0.125, lead_s=0.044,
            control_dt_s=0.01, physics_dt_s=0.001,
            device=torch.device("cpu"), expected_trace=reference,
            takeover_times_s=times,
        )


def test_recovery_smoke_uses_separate_checkpoint_and_default_disabled():
    cfg = load_training_config("configs/training/human_visible_recovery_probe_smoke.toml")
    assert cfg.human_visible.alpha_sweep_enabled
    assert cfg.human_visible.alpha_sweep_recovery_probe_enabled
    assert cfg.human_visible.alpha_sweep_alphas is None
    assert cfg.run.output.endswith("real_chart_humanvisible_recovery_probe_smoke.pt")
    original = load_training_config("configs/training/human_visible_alpha_sweep_smoke.toml")
    assert not original.human_visible.alpha_sweep_recovery_probe_enabled
