from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.adofai_rules import classify_timing, timing_windows
from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_timing_teacher as timing
from dmdod.training import prestige1_timing_comparison as comparison


class MiniPhysics:
    """Real 1ms frozen 8-Key actuator, minimal true target classifier."""

    def __init__(self, target_times):
        self.motor = NKeyMotorEnv(8, control_dt_s=.010, physics_dt_s=.001)
        self.motor.reset()
        self.key_names = self.motor.key_names
        self.segment = SimpleNamespace(
            targets=[SimpleNamespace(episode_time_s=t, bpm=120.)
                     for t in target_times],
        )
        self._used = [False] * len(target_times)
        self._missed = [False] * len(target_times)

    def _classify(self, error_s, target):
        return classify_timing(error_s, target.bpm)

    def _windows(self, target):
        return timing_windows(target.bpm)

    def privileged_episode_time_s(self):
        return self.motor.time_s


def press(env, *keys):
    return NKeyAction(tuple(1. if key in keys else 0.
                            for key in env.key_names))


def test_first_frame_counterfactual_delays_actual_keydown_without_mutation():
    env = MiniPhysics([.100])
    base = press(env, "left_1")
    neutral = NKeyAction((0.,) * 8)
    state = env.motor.observe()
    baseline = timing.forecast_first_frame(env, base, base, lead_s=.12)
    deferred = timing.forecast_first_frame(env, neutral, base, lead_s=.12)
    assert len(baseline.hit_errors_ms) == len(deferred.hit_errors_ms) == 1
    assert baseline.too_early == deferred.too_early == 0
    assert baseline.misses == deferred.misses == 0
    assert baseline.hit_errors_ms[0][0] == deferred.hit_errors_ms[0][0] == 0
    assert baseline.hit_errors_ms[0][1] < deferred.hit_errors_ms[0][1] < 0.
    assert timing.eligible_timing_defer(baseline, deferred) is not None
    assert env.motor.time_s == 0.
    assert env.motor.observe() == state
    assert env._used == [False] and env._missed == [False]


def test_timing_controller_uses_neutral_and_leaves_v24_safety_in_place():
    env = MiniPhysics([.100])
    original = press(env, "left_1")
    controller = timing.TimingCenteredActuatorTeacher(lead_s=.12)
    corrected = controller(env, original)
    assert corrected.values[env.key_names.index("left_1")] == 0.
    assert all(value != -1. for value in corrected.values)
    assert controller.statistics()["timing"]["neutral_defer_commands"] == 1
    assert controller.statistics()["timing"]["reverse_defer_commands"] == 0
    assert controller.statistics()["v24_shield"]["changed_frames"] == 0
    assert controller.statistics()["timing_action_transitions"]["frames"] == 1
    assert env.motor.time_s == 0.

    dangerous = MiniPhysics([.400])
    second = timing.TimingCenteredActuatorTeacher(lead_s=.12)
    safeguarded = second(dangerous, press(dangerous, "left_1"))
    assert safeguarded.values[dangerous.key_names.index("left_1")] == -1.
    assert second.statistics()["v24_shield"]["changed_frames"] == 1
    assert second.statistics()["timing"]["neutral_defer_commands"] == 0
    assert dangerous.motor.time_s == 0.


def test_forecast_rejects_nonfinite_lead_and_wrong_actions():
    env = MiniPhysics([.1])
    original = press(env, "left_1")
    for lead in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            timing.forecast_first_frame(env, original, original, lead_s=lead)
    with pytest.raises(ValueError):
        timing.forecast_first_frame(
            env, NKeyAction((1.,)), original, lead_s=.12)
    with pytest.raises(ValueError):
        timing.TimingCenteredActuatorTeacher(lead_s=0.)


def test_defer_requires_unchanged_hit_set_and_no_physics_regression():
    reference = timing.ScheduledForecast(
        (), ((1, -40.), (2, -30.)), 0, 0, 0, 0)
    improving = timing.ScheduledForecast(
        (), ((1, -20.), (2, -30.)), 0, 0, 0, 0)
    assert timing.eligible_timing_defer(reference, improving) == 20.
    lost_hit = timing.ScheduledForecast(
        (), ((1, -20.),), 0, 0, 0, 0)
    extra_early = timing.ScheduledForecast(
        (), ((1, -20.), (2, -30.)), 1, 0, 0, 0)
    extra_miss = timing.ScheduledForecast(
        (), ((1, -20.), (2, -30.)), 0, 1, 0, 0)
    extra_stray = timing.ScheduledForecast(
        (), ((1, -20.), (2, -30.)), 0, 0, 0, 1)
    worsens_other = timing.ScheduledForecast(
        (), ((1, 0.), (2, -32.)), 0, 0, 0, 0)
    no_gain = timing.ScheduledForecast(
        (), ((1, -40.), (2, -30.)), 0, 0, 0, 0)
    for alt in (lost_hit, extra_early, extra_miss, extra_stray,
                worsens_other, no_gain):
        assert timing.eligible_timing_defer(reference, alt) is None
    assert timing.eligible_timing_defer(
        timing.ScheduledForecast((), (), 0, 0, 0, 0), improving) is None


def test_paired_target_analysis_accounts_for_survival_without_position_shift():
    old = [
        {"target_index": 0, "error_ms": -60.},
        {"target_index": 3, "error_ms": -20.},
    ]
    new = [
        {"target_index": 0, "error_ms": -20.},
        {"target_index": 2, "error_ms": -40.},
    ]
    stats = comparison._paired_analysis(old, new)
    assert stats["paired_hits"] == 1
    assert stats["baseline_only_targets"] == 1
    assert stats["candidate_only_targets"] == 1
    assert stats["improved"] == 1
    assert stats["paired_baseline_mae_ms"] == 60.
    assert stats["paired_candidate_mae_ms"] == 20.


def test_physical_gate_fails_closed_on_one_missing_hit():
    baseline = dict(hits=850, keydowns=850)
    safe = dict(
        safe=20, unsafe_anchors=[], hits=850,
        keydowns=850, misses=0, too_early=0, unjudged=0)
    assert comparison._safety_gate(safe, baseline)
    assert not comparison._safety_gate({**safe, "hits": 849}, baseline)
    assert not comparison._safety_gate({**safe, "misses": 1}, baseline)
    assert not comparison._safety_gate({**safe, "too_early": 1}, baseline)
    assert not comparison._safety_gate({**safe, "safe": 19}, baseline)


def test_train_report_is_non_replacing_and_no_checkpoint_access_when_present(
        tmp_path, monkeypatch):
    config = p.PrestigeConfig(
        dataset="absent", output_dir=str(tmp_path),
        core_path="absent", old_best_path="absent", device="cpu")
    filename = tmp_path / "diagnostics" / "prestige1" / comparison.REPORT
    filename.parent.mkdir(parents=True)
    filename.write_text("protected", encoding="utf-8")
    monkeypatch.setattr(
        p, "_protect_old_best",
        lambda *_: (_ for _ in ()).throw(AssertionError("must refuse early")))
    with pytest.raises(FileExistsError):
        comparison.run_train_timing_comparison(config)
    assert filename.read_text(encoding="utf-8") == "protected"
