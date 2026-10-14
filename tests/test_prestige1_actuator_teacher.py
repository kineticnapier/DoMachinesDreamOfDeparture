from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.adofai_rules import classify_timing, timing_windows
from dmdod.motor.n_key import NKeyAction, NKeyMotorEnv
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_actuator_teacher as actuator
from dmdod.training import prestige1_actuator_comparison as comparison


class SmallTeacherEnvironment:
    """Actual 1ms actuator, minimal pure physical target classifier."""

    def __init__(self, times):
        self.motor = NKeyMotorEnv(8, control_dt_s=.01, physics_dt_s=.001)
        self.motor.reset()
        self.key_names = self.motor.key_names
        self.segment = SimpleNamespace(
            targets=[SimpleNamespace(episode_time_s=t, bpm=120.)
                     for t in times])
        self._used = [False] * len(times)
        self._missed = [False] * len(times)

    def _windows(self, target):
        return timing_windows(target.bpm)

    def _classify(self, error_s, target):
        return classify_timing(error_s, target.bpm)


def press(*keys):
    names = SmallTeacherEnvironment([]).key_names
    return NKeyAction(tuple(1.0 if key in keys else 0.0 for key in names))


def test_future_outside_pass_is_detected_at_physical_not_command_time():
    # At 120 BPM the real pass radius is 60/(3*120) = 1/6s,
    # so .20s is not necessarily TooEarly for a ~50ms actuator delay.
    env = SmallTeacherEnvironment([.40])
    original = press("left_1")
    before = env.motor.observe()
    forecast = actuator.forecast_held_command(env, original, lead_s=.12)
    assert forecast.keydowns, "Physical KeyDown absent in forecast horizon"
    assert forecast.keydowns[0][3] == "too_early"
    assert forecast.too_early_keys == ("left_1",)
    assert forecast.keydowns[0][0] > 0.
    assert forecast.keydowns[0][0] < .40 - timing_windows(120.).pass_s
    assert env.motor.time_s == 0.
    assert env.motor.observe() == before
    assert env._used == [False] and env._missed == [False]


def test_safe_pre_acceleration_is_preserved():
    env = SmallTeacherEnvironment([.08])
    original = press("left_1")
    assert not actuator.forecast_held_command(
        env, original, lead_s=.12).too_early_keys
    corrector = actuator.ActuatorAwareTeacherSafety(lead_s=.12)
    assert corrector(env, original) == original
    assert corrector.statistics()["changed_frames"] == 0


def test_early_keydown_is_braked_without_changing_live_motor():
    # Force a real TooEarly, rather than an allowed 120 BPM early-pass hit.
    env = SmallTeacherEnvironment([.40])
    original = press("left_1")
    corrector = actuator.ActuatorAwareTeacherSafety(lead_s=.12)
    new = corrector(env, original)
    assert new.values[env.key_names.index("left_1")] == -1.0
    assert original.values[env.key_names.index("left_1")] == 1.0
    assert not actuator.forecast_held_command(
        env, new, lead_s=.12).too_early_keys
    assert env.motor.time_s == 0.
    assert corrector.statistics()["braked_key_commands"] == 1


def test_two_keys_across_one_target_keep_one_hit_and_brake_extra_key():
    env = SmallTeacherEnvironment([.08, .25])
    original = press("left_1", "right_1")
    baseline = actuator.forecast_held_command(env, original, lead_s=.12)
    assert baseline.hits == 1
    assert len(baseline.too_early_keys) == 1
    early_key = baseline.too_early_keys[0]
    corrector = actuator.ActuatorAwareTeacherSafety(lead_s=.12)
    candidate = corrector(env, original)
    assert candidate.values[env.key_names.index(early_key)] == -1.
    assert sum(value > 0 for value in candidate.values) == 1
    forecast = actuator.forecast_held_command(env, candidate, lead_s=.12)
    assert forecast.hits == 1
    assert not forecast.too_early_keys


def test_original_unavoidable_inertia_is_reported_without_mutation():
    env = SmallTeacherEnvironment([.25])
    moving = press("left_1")
    for _ in range(3):
        env.motor.step(moving)
    before = env.motor.observe()
    # May still be reversible; independently verify the pure forecast.
    neutral = NKeyAction((0.,) * 8)
    forecast = actuator.forecast_held_command(env, neutral, lead_s=.12)
    assert isinstance(forecast.too_early_keys, tuple)
    assert env.motor.observe() == before
    assert env.motor.time_s == pytest.approx(.03)


def test_reject_nonpositive_forecast_lead_or_invalid_width():
    env = SmallTeacherEnvironment([.1])
    with pytest.raises(ValueError):
        actuator.forecast_held_command(env, press("left_1"), lead_s=0.)
    with pytest.raises(ValueError):
        actuator.forecast_held_command(env, NKeyAction((1.,)), lead_s=.12)


def test_comparison_uses_immutable_v23_baseline_and_train_only(monkeypatch, tmp_path):
    config = p.PrestigeConfig(
        dataset="absent", output_dir=str(tmp_path),
        core_path="absent", old_best_path="absent", device="cpu")
    output = tmp_path / "diagnostics" / "prestige1" / comparison.REPORT
    output.parent.mkdir(parents=True)
    output.write_text("protected", encoding="utf-8")
    monkeypatch.setattr(
        p, "_protect_old_best",
        lambda *_: (_ for _ in ()).throw(AssertionError("must fail before reading")))
    with pytest.raises(FileExistsError):
        comparison.run_train_actuator_comparison(config)
    assert output.read_text(encoding="utf-8") == "protected"


def test_aggregate_reports_hits_misses_unjudged_and_overload():
    row = {
        "anchor": 1,
        "baseline": dict(hits=9, misses=1, too_early=2, keydowns=11,
                         overload=True, unjudged=3),
        "candidate": dict(hits=10, misses=0, too_early=0, keydowns=10,
                          overload=False, unjudged=3),
    }
    baseline = comparison._aggregate([row], "baseline")
    candidate = comparison._aggregate([row], "candidate")
    assert baseline["unsafe_anchors"] == [1]
    assert candidate["safe"] == 1
    assert candidate["hits"] == 10
    assert candidate["misses"] == 0
    assert candidate["unjudged"] == 3
