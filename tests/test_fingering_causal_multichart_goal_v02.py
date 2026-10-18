from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_fingering_causal_multichart_goal_v02 as goal


def _chart(tmp_path):
    folder = tmp_path / "Validation"
    folder.mkdir(exist_ok=True)
    chart = folder / "one.adofai"
    chart.write_text("dummy chart", encoding="utf-8")
    return chart


def _protocol():
    return {
        "checkpoint_absolute_path": "fake",
        "checkpoint_sha256": "a" * 64,
        "reference_replay_sha256": "b" * 64,
        "code_hashes": {"causal": "c" * 64},
        "settings": {"fixed": True},
    }


def _manifest(tmp_path):
    chart = _chart(tmp_path)
    return {
        "format": goal.FORMAT,
        "protocol": _protocol(),
        "selected": [{
            "path": str(chart), "name": chart.name,
            "split": "validation", "sha256": goal.hash_file(chart)
        }],
        "selection": {"selection_basis": "fixture"}
    }


def _run(hits=3):
    return {
        "stats": {
            "hits": hits, "targets": 3, "misses": 3 - hits,
            "x_accuracy_percent": 100.0 if hits == 3 else 80.0,
            "timing_mae_ms": 4.0, "too_early": 0,
            "overloaded": False, "physical_keydowns": 3,
        },
        "decisions": [{"event_index": 0, "chosen": "right_1"}],
        "keydowns": [{"axis": "right_1", "contact": "right_1", "time_ms": 1.0}],
        "actual_downs": [(1.0, "right_1")],
        "interventions": {"actual_early_axis_steps": 1},
        "termination_reason": "chart_complete",
        "elapsed_wall_s": .5,
    }


def test_goal_progress_handles_incomplete_policy_pair(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    chart = manifest["selected"][0]
    path = goal.checkpoint_path(
        output, 1, chart["sha256"], "free", "baseline"
    )
    goal.write_json_atomic(path, {
        "format": goal.FORMAT + "-arm",
        "chart_sha256": chart["sha256"],
        "checkpoint_sha256": manifest["protocol"]["checkpoint_sha256"],
        "policy": "free", "arm": "baseline",
        "measured_wall_s": 2.5,
        "result": _run(),
    }, create_only=True)
    state = goal.snapshot(manifest, output, goal_seconds=120.0)
    assert state["completed_arms"] == 1
    assert state["total_arms"] == 4
    assert state["completed_charts"] == 0
    assert state["spent_completed_arms_s"] == 2.5
    assert state["projected_total_s"] == 10.0
    assert state["goal_status"] == "projected_within_target"
    assert state["charts"][0]["policies"]["free"]["pending_arms"] == ["causal_lead"]
    assert state["aggregate_completed_pairs_only"]["free"]["completed_charts"] == 0


def test_parallel_progress_uses_measured_batch_wall_not_arm_sum(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    chart = manifest["selected"][0]
    for policy, _alpha in goal.POLICIES:
        for arm in goal.ARMS:
            path = goal.checkpoint_path(
                output, 1, chart["sha256"], policy, arm
            )
            goal.write_json_atomic(path, {
                "format": goal.FORMAT + "-arm",
                "chart_sha256": chart["sha256"],
                "checkpoint_sha256": manifest["protocol"]["checkpoint_sha256"],
                "policy": policy,
                "arm": arm,
                "measured_wall_s": 10.0,
                "result": _run(),
            }, create_only=True)
    state = goal.snapshot(
        manifest, output, goal_seconds=25.0,
        active_invocation_wall_s=20.0, workers=2,
    )
    assert state["spent_completed_arms_s"] == 40.0
    assert state["active_invocation_wall_s"] == 20.0
    assert state["projected_total_s"] == 20.0
    assert state["goal_projection_basis"] == "active_invocation_wall_clock"
    assert state["goal_status"] == "projected_within_target"


def test_goal_is_reporting_only_and_does_not_stop_evaluation(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    executed = []

    monkeypatch.setattr(goal.causal.runner, "load_model", lambda p: object())
    monkeypatch.setattr(goal, "load_adofai", lambda p: object())
    monkeypatch.setattr(goal, "compile_adofai", lambda c: c)
    monkeypatch.setattr(goal, "build_playable_segment", lambda c: c)

    def simulate(_segment, _model, policy, alpha, arm, *, wall_timeout_s):
        executed.append((policy, arm))
        return _run()
    monkeypatch.setattr(goal.causal, "execute", simulate)

    result = goal.run(
        manifest, output, wall_timeout_s=3.0,
        goal_seconds=0.000001, checkpoint=tmp_path / "fake.pt"
    )
    assert len(executed) == 4
    assert result["completed_arms"] == 4
    assert result["completed_charts"] == 1
    assert result["goal_status"] == "projected_over_target"
    assert result["aggregate_completed_pairs_only"]["free"]["completed_charts"] == 1
    assert (output / "progress.json").is_file()


def test_resume_skips_finished_arms(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    calls = []
    monkeypatch.setattr(goal.causal.runner, "load_model", lambda p: object())
    monkeypatch.setattr(goal, "load_adofai", lambda p: object())
    monkeypatch.setattr(goal, "compile_adofai", lambda c: c)
    monkeypatch.setattr(goal, "build_playable_segment", lambda c: c)

    def interrupted(*args, **kwargs):
        calls.append((args[2], args[4]))
        if len(calls) > 1:
            raise KeyboardInterrupt("test stop")
        return _run()

    monkeypatch.setattr(goal.causal, "execute", interrupted)
    with pytest.raises(KeyboardInterrupt):
        goal.run(manifest, output, wall_timeout_s=1,
                 goal_seconds=10, checkpoint=tmp_path / "fake.pt")
    assert goal.snapshot(manifest, output, 10)["completed_arms"] == 1

    def continuing(*args, **kwargs):
        calls.append((args[2], args[4]))
        return _run()
    monkeypatch.setattr(goal.causal, "execute", continuing)
    state = goal.run(manifest, output, wall_timeout_s=1,
                     goal_seconds=10, checkpoint=tmp_path / "fake.pt")
    assert len(calls) == 5  # 2 attempted before interruption + 3 resumed
    assert state["completed_arms"] == 4
    assert state["completed_charts"] == 1


def test_start_and_resume_keep_original_chart_selection(tmp_path):
    manifest = _manifest(tmp_path)
    chart = Path(manifest["selected"][0]["path"])
    output = tmp_path / "out"
    selected = [chart]
    first = goal.start_or_resume(output, resume=False, root=tmp_path,
                                 charts=selected, max_charts=1,
                                 protocol=_protocol())
    same = goal.start_or_resume(output, resume=True,
                                root=tmp_path / "nonexistent",
                                charts=[], max_charts=99,
                                protocol=_protocol())
    assert same["selected"] == first["selected"]
    with pytest.raises(FileExistsError):
        goal.start_or_resume(output, resume=False, root=tmp_path,
                             charts=[], max_charts=1, protocol=_protocol())
    with pytest.raises(ValueError, match="resume refused"):
        goal.start_or_resume(output, resume=True, root=tmp_path,
                             charts=[], max_charts=99,
                             protocol={**_protocol(), "checkpoint_sha256": "x" * 64})


def test_resume_rejects_modified_chart(tmp_path):
    manifest = _manifest(tmp_path)
    chart = Path(manifest["selected"][0]["path"])
    output = tmp_path / "out"
    goal.start_or_resume(output, resume=False, root=tmp_path, charts=[chart],
                         max_charts=1, protocol=_protocol())
    chart.write_text("changed chart", encoding="utf-8")
    with pytest.raises(RuntimeError, match="chart content drift"):
        goal.start_or_resume(output, resume=True, root=tmp_path,
                             charts=[], max_charts=1, protocol=_protocol())


def test_checkpoint_never_overwrites_existing_arm(tmp_path):
    file = tmp_path / "arm.json"
    goal.write_json_atomic(file, {"version": 1}, create_only=True)
    with pytest.raises(FileExistsError):
        goal.write_json_atomic(file, {"version": 2}, create_only=True)
    assert json.loads(file.read_text(encoding="utf-8")) == {"version": 1}


def test_run_lock_clears_on_keyboard_interrupt(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with goal.exclusive_run_lock(tmp_path):
            assert (tmp_path / ".running.lock").exists()
            raise KeyboardInterrupt()
    assert not (tmp_path / ".running.lock").exists()
