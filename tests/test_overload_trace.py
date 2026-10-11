from __future__ import annotations

from dataclasses import fields

import pytest
import torch

from dmdod.adofai_chart import parse_adofai_text
from dmdod.adofai_playable import build_playable_segment
from dmdod.adofai_timing import compile_adofai
from dmdod.keyboard import KeyEvent
from dmdod.motor_env import TimedKeyEvent
from dmdod.n_key_motor import NKeyAction
from dmdod.n_key_real_chart import (
    DiagnosticHudNKeyRealChartMotorEnv,
    NKeyActionFrameTrace,
    NKeyPhysicalKeyDownTrace,
    EpisodeTerminationTrace,
    NKeyOverloadTrace,
    TooEarlyKeyDownTrace,
)
from dmdod.training.human_visible_curriculum import (
    ValidationAnchorSnapshot,
    catastrophic_trace_anchor_indices,
    first_keydown_count_divergence,
    format_action_output_diagnostics,
    format_overload_trace_comparison,
    select_best_trace_reference,
    validation_rank,
)
from dmdod.training.real_chart import NamedSegment, evaluate_role_continuous


def _segment():
    chart = parse_adofai_text(
        '{"angleData":[0,90,0,0],"settings":{"bpm":120,"pitch":100,'
        '"countdownTicks":0,"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    return build_playable_segment(
        compiled, start_s=0.0, end_s=compiled.duration_s
    )


def _trace(*, reason="Overload", end=0.5, events=(), keydowns=()):
    return NKeyOverloadTrace(
        termination=EpisodeTerminationTrace(
            time_s=end, reason=reason, hits=30, too_early=len(events),
            keydowns=len(keydowns), next_target_index=2, overload_value=5.0,
        ),
        keydown_times_s=tuple(keydowns),
        too_early_events=tuple(events),
    )


def _event(t, *, key="left_1", failed=False):
    return TooEarlyKeyDownTrace(
        time_s=t, key=key, target_index=3, error_ms=-120.0,
        overload_before=3.0, overload_after=5.0, fail_overload=failed,
    )


def test_too_early_trace_gauge_fail_event_and_overload_termination():
    env = DiagnosticHudNKeyRealChartMotorEnv(
        _segment(), key_count=4, capture_overload_trace=True,
    )
    env.reset()
    for index in range(20):
        env._score_event(
            TimedKeyEvent(0.001 + index * 0.0001, "left_1", KeyEvent.DOWN)
        )
        if env._overload.overloaded:
            break
    assert env._overload.overloaded
    result = env.step(NKeyAction((0.0, 0.0, 0.0, 0.0)))
    assert result.done
    trace = env.overload_trace
    assert trace is not None
    assert trace.termination.reason == "Overload"
    assert trace.termination.time_s == pytest.approx(trace.too_early_events[-1].time_s)
    assert trace.termination.keydowns == env.physical_keydowns
    assert trace.termination.next_target_index == env._next_target_index()
    assert trace.termination.overload_value == pytest.approx(env._overload.value)
    assert trace.too_early_events[-1].fail_overload
    assert all(
        x.overload_after >= x.overload_before for x in trace.too_early_events
    )
    assert len(trace.keydown_times_s) == trace.termination.keydowns
    assert trace.too_early_events[0].target_index == 0
    assert trace.too_early_events[0].error_ms < 0.0


def test_other_termination_reasons_are_distinguishable():
    action = NKeyAction((0.0, 0.0, 0.0, 0.0))
    missed = DiagnosticHudNKeyRealChartMotorEnv(
        _segment(), key_count=4, capture_overload_trace=True,
    )
    missed.reset()
    missed._failed_on_miss = True
    missed.step(action)
    assert missed.overload_trace.termination.reason == "Miss failure"

    resolved = DiagnosticHudNKeyRealChartMotorEnv(
        _segment(), key_count=4, capture_overload_trace=True,
    )
    resolved.reset()
    resolved._used[:] = [True] * len(resolved._used)
    resolved.step(action)
    assert resolved.overload_trace.termination.reason == "All targets resolved"
    assert resolved.overload_trace.termination.next_target_index is None

    timed = DiagnosticHudNKeyRealChartMotorEnv(
        _segment(), key_count=4, capture_overload_trace=True,
    )
    timed.reset()
    timed._episode_end_s = 0.0
    timed.step(action)
    assert timed.overload_trace.termination.reason == "Time limit"


def test_trace_off_returns_none_and_does_not_change_visible_schema():
    env = DiagnosticHudNKeyRealChartMotorEnv(_segment(), key_count=4)
    obs = env.reset()
    assert env.overload_trace is None
    assert "overload_value" not in {field.name for field in fields(obs)}


class _IdlePolicy:
    key_count = 4

    def initial_state(self, device):
        return torch.zeros(1)

    def eval(self):
        return self

    def forward_step(self, x, state):
        return torch.zeros(self.key_count), None, None, state


def test_optional_trace_collector_preserves_eval_return_and_stats(capsys):
    segment = _segment()
    named = NamedSegment(
        "validation", "chart", "sha1", 0.0, segment.duration_s, segment
    )
    model = _IdlePolicy()
    kwargs = dict(
        label="test", control_dt_s=0.01, physics_dt_s=0.001,
        device=torch.device("cpu"), verbose=False,
    )
    plain = evaluate_role_continuous(model, [named], **kwargs)
    traces = []
    traced = evaluate_role_continuous(
        model, [named], trace_collector=traces, **kwargs
    )
    assert plain == traced
    assert len(traces) == 1
    assert traces[0].action_frames
    assert traces[0].action_frames[0].action_values == (0.0,) * 4
    assert len(traces[0].action_frames[0].positions_m) == 4
    assert traces[0].termination.keydowns == traced[0][1]
    assert traces[0].termination.reason in {
        "All targets resolved", "Time limit", "Miss failure"
    }


def test_too_early_report_shows_last_ten_including_fail_event():
    events = tuple(_event(i / 100, failed=i == 11) for i in range(12))
    lines = format_overload_trace_comparison(
        _trace(reason="Time limit", end=2.0),
        _trace(reason="Overload", end=1.0, events=events),
        anchor_label="#09 chart",
    )
    report = "\n".join(lines)
    assert "dt=-1.000s" in report
    assert "Time limit" in report and "Overload" in report
    assert "FAIL_OVERLOAD" in report
    assert "candidate TooEarly: total=12 last=10" in report
    assert "t=0.000s key=" not in report
    assert "t=0.110s key=" in report
    assert "next_target=2 gauge=5.000" in report


def test_key_agnostic_keydown_divergence_ignores_key_relabel():
    before = (0.1, 0.15, 0.2, 0.4)
    after = tuple(before)
    assert first_keydown_count_divergence(before, after) is None
    report = "\n".join(format_overload_trace_comparison(
        _trace(keydowns=before, events=(_event(0.1, key="left_1"),)),
        _trace(keydowns=after, events=(_event(0.1, key="right_4"),)),
        anchor_label="#01",
    ))
    assert "divergence: none" in report
    assert first_keydown_count_divergence(before, ()) is not None


def test_safe_loss_reports_affected_anchor_and_largest_hits_drop():
    def a(index, hits, overloaded):
        return ValidationAnchorSnapshot(
            index, f"chart{index}", "sha", 0.0, 1.0,
            hits, 0, overloaded, hits,
        )
    prior = (a(1, 100, False), a(2, 50, False), a(3, 100, True))
    current = (a(1, 40, True), a(2, 49, False), a(3, 101, False))
    assert catastrophic_trace_anchor_indices(prior, current) == (0, 1)


def test_best_trace_reference_updates_without_changing_rank():
    previous = (_trace(reason="Time limit", end=3.0),)
    candidate = (_trace(reason="Overload", end=1.0),)
    assert select_best_trace_reference(
        previous, candidate, selected_best=False
    ) is previous
    assert select_best_trace_reference(
        previous, candidate, selected_best=True
    ) is candidate
    best = {
        "safe": 12, "hits": 1786, "x": 50.0,
        "pp": 27.0, "mae_ms": 30.0, "early": 20,
    }
    worse = dict(best, safe=9, hits=1400)
    assert validation_rank(worse) < validation_rank(best)


def test_action_frame_uses_actual_command_and_post_step_motor_state():
    env = DiagnosticHudNKeyRealChartMotorEnv(
        _segment(), key_count=4, capture_overload_trace=True,
    )
    env.reset()
    action = NKeyAction((0.6, -0.2, 0.0, 0.1))
    env.step(action)
    frame = env._trace_action_frames[0]
    motor = env.motor.observe()
    assert frame.time_s == pytest.approx(0.0)
    assert frame.action_values == action.values
    assert frame.positions_m == tuple(motor.positions_m)
    assert frame.velocities_m_s == tuple(motor.velocities_m_s)
    assert frame.pressed_flags == tuple(motor.pressed_flags)
    assert frame.next_target_index == 0
    assert frame.next_target_time_s == pytest.approx(_segment().targets[0].episode_time_s)
    env._score_event(TimedKeyEvent(0.02, "left_1", KeyEvent.DOWN))
    down = env._trace_physical_keydowns[-1]
    assert down.time_s == pytest.approx(0.02)
    assert down.key == "left_1"
    assert down.target_index == 0
    assert down.target_time_s == pytest.approx(frame.next_target_time_s)


def test_action_output_comparison_reports_actions_motors_and_physical_events():
    def frame(t, action):
        return NKeyActionFrameTrace(
            time_s=t, action_values=action,
            positions_m=(0.001, 0.002, 0.003, 0.004),
            velocities_m_s=(0.01, 0.02, 0.03, 0.04),
            pressed_flags=(False, False, False, False),
            next_target_index=3, next_target_time_s=0.3,
        )
    first = _trace(reason="All targets resolved", end=1.0)
    second = _trace(reason="Overload", end=0.11)
    first = NKeyOverloadTrace(
        first.termination, (), (),
        action_frames=(frame(0.0, (0.0, 0.0, 0.0, 0.0)),
                       frame(0.1, (0.1, 0.0, 0.0, 0.0))),
        physical_keydowns=(NKeyPhysicalKeyDownTrace(0.101, "left_2", 3, 0.3),),
    )
    second = NKeyOverloadTrace(
        second.termination, (), (),
        action_frames=(frame(0.0, (0.0, 0.0, 0.0, 0.0)),
                       frame(0.1, (0.8, 0.0, 0.0, 0.0))),
        physical_keydowns=(NKeyPhysicalKeyDownTrace(0.102, "left_2", 3, 0.3),),
    )
    report = "\n".join(format_action_output_diagnostics(
        first, second, divergence=(0.0, 1, 4),
    ))
    assert "first KeyDown-count divergence" in report
    assert "candidate FailOverload lead-up" in report
    assert "tanh(mu)=+0.100/+0.800" in report
    assert "position=1.00/1.00mm" in report
    assert "next-target=#3@0.300s" in report
    assert "physical KeyDowns" in report
    assert "key=left_2" in report


def test_action_comparison_handles_old_traces_and_invalid_sample_count():
    assert format_action_output_diagnostics(
        _trace(), _trace(), divergence=None
    ) == ()
    with pytest.raises(ValueError, match="positive"):
        format_action_output_diagnostics(
            _trace(), _trace(), divergence=None, max_frames_per_window=0
        )
