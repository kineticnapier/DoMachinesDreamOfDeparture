from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.adofai_rules import TimingJudgement, timing_windows
from dmdod.training import prestige1_target_contract as target
from dmdod.training import prestige1_teacher_safety as safety
from dmdod.training import prestige1 as p


def note(t, bpm=120.0):
    return SimpleNamespace(episode_time_s=t, bpm=bpm)


def state(t, now, *, hit=False, missed=False, bpm=120.0):
    return target.target_contract([note(t, bpm)], [hit], [missed], 0, now)


def test_target_contract_early_and_late_physical_pass_windows():
    radius = timing_windows(120.0).pass_s
    t = 1.0
    early = state(t, t-radius/2)
    assert early.phase == target.TargetPhase.EARLY_PASS.value
    assert early.remaining_s == pytest.approx(radius/2)
    assert early.legal_to_hit
    assert early.would_judge not in ("too_early", "too_late")
    late = state(t, t+radius/2)
    assert late.phase == target.TargetPhase.LATE_PASS.value
    assert late.remaining_s == pytest.approx(-radius/2)
    assert late.legal_to_hit
    assert late.would_judge != TimingJudgement.TOO_LATE.value


def test_target_contract_too_early_or_expired_pending():
    radius = timing_windows(120.0).pass_s
    t = 1.0
    future = state(t, t-1.5*radius)
    assert future.phase == target.TargetPhase.FUTURE_OUTSIDE_PASS.value
    assert future.would_judge == TimingJudgement.TOO_EARLY.value
    assert not future.legal_to_hit
    pending = state(t, t+1.5*radius)
    assert pending.phase == target.TargetPhase.EXPIRED_PENDING.value
    assert pending.would_judge == TimingJudgement.TOO_LATE.value
    assert not pending.legal_to_hit
    assert not pending.resolved
    finalized = state(t, t+1.5*radius, missed=True)
    assert finalized.phase == target.TargetPhase.MISSED.value
    assert finalized.resolved and not finalized.legal_to_hit
    assert finalized.would_judge is None


def test_target_contract_on_time_exact_boundary_and_resolved():
    radius = timing_windows(120.0).pass_s
    assert state(1.0, 1.0).phase == target.TargetPhase.ON_TIME.value
    assert state(1.0, 1.0).legal_to_hit
    # Actual classifier, not float heuristics, defines legal boundary.
    for now in (1.0-radius, 1.0+radius):
        record = state(1.0, now)
        judgement = record.would_judge
        assert record.legal_to_hit == (
            judgement not in ("too_early", "too_late"))
    hit = state(1.0, .95, hit=True)
    assert hit.phase == target.TargetPhase.HIT.value
    assert hit.resolved and hit.hit and not hit.legal_to_hit


def test_next_unresolved_skips_early_scored_target():
    data = [note(.12), note(.21)]
    result = target.next_unresolved_contract(
        data, [True, False], [False, False], .10)
    assert result.target_index == 1
    assert result.remaining_s == pytest.approx(.11)
    assert result.phase == target.TargetPhase.EARLY_PASS.value


def test_next_unresolved_includes_overdue_while_not_expired():
    data = [note(.1), note(.5)]
    result = target.next_unresolved_contract(
        data, [False]*2, [False]*2, .15)
    assert result.target_index == 0
    assert result.phase == target.TargetPhase.LATE_PASS.value
    assert result.legal_to_hit
    missed = target.next_unresolved_contract(
        data, [False]*2, [True, False], .15)
    assert missed.target_index == 1
    assert missed.phase == target.TargetPhase.FUTURE_OUTSIDE_PASS.value


def test_no_target_when_all_resolved_and_validation_errors():
    data = [note(.1), note(.5)]
    assert target.next_unresolved_contract(
        data, [True, False], [False, True], .2) is None
    with pytest.raises(ValueError):
        target.next_unresolved_contract(data, [False], [False], .2)
    with pytest.raises(ValueError):
        target.target_contract(data, [True, False], [True, False], 0, .2)
    with pytest.raises(IndexError):
        target.target_contract(data, [False]*2, [False]*2, 2, .2)


def test_contract_matches_real_physics_classifier_without_exporting_to_student():
    class FakeEnvironment:
        segment = SimpleNamespace(
            targets=[note(.1), note(.2)], pitch_ratio=1.)
        difficulty = "normal"
        timing_scale = 1.
        _used = [True, False]
        _missed = [False, False]
        def privileged_episode_time_s(self):
            return .23
        def _next_target_index(self):
            return 1
        def _classify(self, error, target_note):
            from dmdod.adofai_rules import classify_timing
            return classify_timing(error, target_note.bpm)
    e = FakeEnvironment()
    record = target.contract_from_env(e)
    assert record.target_index == 1
    assert record.phase == target.TargetPhase.LATE_PASS.value
    assert record.legal_to_hit
    assert "Student" not in repr(record)


def test_safety_replay_detects_mismatch_against_v22():
    current = {"anchor": 9, "chart": "A", "stats": {
        "hits": 9, "misses": 1, "targets": 48, "too_early": 7,
        "overload": True, "keydowns": 16, "unjudged": 38}}
    old = {"anchor": 9, "chart": "A", "physics": dict(current["stats"])}
    safety._same_as_v22(current, old)
    old["physics"]["keydowns"] += 1
    with pytest.raises(RuntimeError, match="replay divergence"):
        safety._same_as_v22(current, old)


def test_safety_refuses_existing_report_before_reading_paths(monkeypatch, tmp_path):
    config = p.PrestigeConfig(
        dataset="bad", output_dir=str(tmp_path),
        core_path="nonexistent", old_best_path="nonexistent", device="cpu")
    root = safety.transfer.output_dir(config)
    root.mkdir(parents=True)
    (root / safety.REPORT).write_text("original", encoding="utf-8")
    monkeypatch.setattr(
        p, "_protect_old_best",
        lambda *args: (_ for _ in ()).throw(AssertionError("should not read Best")))
    with pytest.raises(FileExistsError):
        safety.run_train_teacher_safety(config)
    assert (root / safety.REPORT).read_text(encoding="utf-8") == "original"
