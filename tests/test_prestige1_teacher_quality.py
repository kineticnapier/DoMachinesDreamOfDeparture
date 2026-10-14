from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from dmdod.adofai_rules import classify_timing
from dmdod.motor.n_key import NKeyAction
from dmdod.training import prestige1 as p
from dmdod.training import prestige1_teacher_quality as quality


def hit(error, judgement):
    return {"error_ms": error, "judgement": judgement}


def test_percentile_is_defined_for_small_samples_and_nonfinite_rejected():
    assert quality.percentile([], .95) is None
    assert quality.percentile([7.], .95) == 7.
    assert quality.percentile([0., 10.], .95) == pytest.approx(9.5)
    assert quality.percentile([0., 10., 20.], .50) == 10.
    with pytest.raises(ValueError):
        quality.percentile([1.], 1.5)
    with pytest.raises(ValueError):
        quality.percentile([math.inf], .5)


def test_hit_only_metrics_separate_signed_bias_mae_and_perfect_family():
    events = [
        hit(-5., "perfect"), hit(-10., "early_perfect"),
        hit(40., "very_late"),
    ]
    stats = quality.summarize_hit_quality(events)
    assert stats["hits"] == 3
    assert stats["mean_signed_error_ms"] == pytest.approx(25. / 3.)
    assert stats["mae_ms"] == pytest.approx(55. / 3.)
    assert stats["median_abs_error_ms"] == 10.
    assert stats["p95_abs_error_ms"] == pytest.approx(37.)
    assert stats["pure_perfect_rate"] == pytest.approx(1./3.)
    assert stats["perfect_family_rate"] == pytest.approx(2./3.)
    assert stats["judgement_counts"]["very_late"] == 1
    assert quality.summarize_hit_quality([])["mae_ms"] is None
    with pytest.raises(ValueError):
        quality.summarize_hit_quality([hit(-90., "too_early")])


def test_physical_judgement_is_scored_at_unresolved_target_not_nearest_time():
    notes = [SimpleNamespace(episode_time_s=1., bpm=120.),
             SimpleNamespace(episode_time_s=2., bpm=120.)]
    env = SimpleNamespace(
        segment=SimpleNamespace(targets=notes),
        _classify=lambda error_s, target: classify_timing(error_s, target.bpm))
    # The first is a Hit; the next is 200ms early and must be TooEarly.
    # The third has no pending note; it must not be counted as a Hit.
    events = [
        dict(time_s=1.0, target_index=0, target_time_s=1.0, key="left_1"),
        dict(time_s=1.8, target_index=1, target_time_s=2.0, key="right_1"),
        dict(time_s=2.5, target_index=None, target_time_s=None, key="left_2"),
    ]
    expected = dict(hits=1, too_early=1, keydowns=3)
    reconstructed = quality.classify_physical_keydowns(events, env, expected)
    assert reconstructed["quality"]["hits"] == 1
    assert reconstructed["too_early_keydowns"] == 1
    assert reconstructed["no_target_keydowns"] == 1
    assert reconstructed["hit_events"][0]["target_index"] == 0
    with pytest.raises(RuntimeError, match="reconstruction"):
        quality.classify_physical_keydowns(
            events, env, dict(hits=2, too_early=0, keydowns=3))
    corrupted = [dict(events[0], target_time_s=.999), *events[1:]]
    with pytest.raises(RuntimeError, match="target time"):
        quality.classify_physical_keydowns(corrupted, env, expected)


def test_transition_summary_counts_direct_and_across_zero_reversals():
    t = quality.CommandTransitions(2)
    for action in ((1., 0.), (0., 1.), (-1., 1.), (1., -1.)):
        t.observe(action)
    stats = t.summary()
    assert stats["frames"] == 4
    assert stats["sign_transitions"] == 5
    assert stats["direct_direction_reversals"] == 2
    assert stats["nonzero_direction_reversals"] == 3
    assert stats["positive_command_endings"] == 2
    with pytest.raises(ValueError):
        t.observe((1.,))


def test_audit_keeps_both_actions_at_same_state_and_counts_disagreement(monkeypatch):
    from dmdod.training import prestige1_alignment_audit as alignment
    comparison = SimpleNamespace(
        kind="legacy_skips_unresolved", legacy_index=3, unresolved_index=1,
        legacy_points_to_resolved=False, unresolved_overdue=False,
        legacy_mask=True, unresolved_mask=True,
        legacy_labels=(.1, 1., 0.),
        unresolved_labels=(.2, 1., 0.))
    monkeypatch.setattr(alignment, "compare_targets", lambda *args: comparison)
    env = SimpleNamespace(
        key_names=("left_4", "left_3", "left_2", "left_1",
                   "right_1", "right_2", "right_3", "right_4"),
        segment=SimpleNamespace(
            targets=[], chart_time_from_episode=lambda t: t),
        _used=[], _missed=[],
        privileged_episode_time_s=lambda: .02,
        _floor_index_at_chart_time=lambda t: 0,
        _next_target_index=lambda: 1,
    )

    class FakeSafety:
        frames = 0
        changed_frames = 0
        def __call__(self, env, original):
            self.frames += 1
            self.changed_frames += 1
            assert original.values[0] == 1.
            return NKeyAction((-1.,) + original.values[1:])
        def statistics(self):
            return {"control_frames": self.frames,
                    "changed_frames": self.changed_frames}

    recorder = quality.TeacherQualityRecorder(.12)
    recorder.filter = FakeSafety()
    original = NKeyAction((1.,) + (0.,) * 7)
    corrected = recorder(env, original)
    assert corrected.values[0] == -1.
    assert original.values[0] == 1.
    assert recorder.action_frames[0]["original_action_8k"] == [1.] + [0.] * 7
    assert recorder.action_frames[0]["corrected_action_8k"] == [-1.] + [0.] * 7
    assert recorder.action_frames[0]["modified_keys"] == ["left_4"]
    summary = recorder.summary()
    assert summary["counts"]["mismatch_frames"] == 1
    assert summary["counts"]["legacy_skips_unresolved"] == 1
    assert summary["counts"]["near_time_disagreement_frames"] == 1
    assert summary["counts"]["original_positive_mismatch_frames"] == 1
    assert summary["counts"]["corrected_positive_mismatch_frames"] == 0
    assert summary["counts"]["positive_to_brake"] == 1
    assert summary["actuator_interventions"]["control_frames"] == 1


def test_v24_provenance_refuses_validation_or_changed_gate():
    gate = dict(passed=False, gate_data_role="train_heldout")
    stage_a = dict(gate=gate)
    config = SimpleNamespace(signature=lambda: "abc")
    manifest = [dict(anchor=i) for i in range(1, 21)]
    old = dict(aggregate={"safe": 16}, train_manifest=manifest)
    previous = {
        "format": quality.v24.FORMAT, "stage": "C-v2.4",
        "role": "Train20 only", "no_training": True, "read_only": True,
        "validation_evaluated": False, "final_evaluated": False,
        "validation_used_for_selection": False, "checkpoint_modified": False,
        "student_checkpoint_loaded": False, "stage_a_gate_overridden": False,
        "original_stage_a_gate": gate, "config_sha256": "abc",
        "train_manifest": manifest, "per_anchor": [{}] * 20,
        "protected_inputs_unchanged": True, "baseline": old["aggregate"],
    }
    quality._check_v24_report(previous, old, stage_a, config, manifest)
    for altered in (
        {**previous, "validation_evaluated": True},
        {**previous, "stage_a_gate_overridden": True},
        {**previous, "original_stage_a_gate": {"passed": True}},
    ):
        with pytest.raises(RuntimeError, match="provenance"):
            quality._check_v24_report(altered, old, stage_a, config, manifest)
    with pytest.raises(RuntimeError, match="FAIL"):
        quality._check_v24_report(previous, old, {
            "gate": dict(passed=True, gate_data_role="train_heldout")
        }, config, manifest)


def test_existing_report_is_never_replaced_even_if_inputs_missing(tmp_path, monkeypatch):
    config = p.PrestigeConfig(
        dataset="unavailable", output_dir=str(tmp_path),
        core_path="unavailable", old_best_path="unavailable", device="cpu")
    report = tmp_path / "diagnostics" / "prestige1" / quality.REPORT
    report.parent.mkdir(parents=True)
    report.write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        p, "_protect_old_best", lambda *_: (
            _ for _ in ()).throw(AssertionError("should fail before touching files")))
    with pytest.raises(FileExistsError):
        quality.run_train_teacher_quality(config)
    assert report.read_text(encoding="utf-8") == "original"
