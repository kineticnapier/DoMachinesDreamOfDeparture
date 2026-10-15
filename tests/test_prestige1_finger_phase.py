from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_global_annealing as physics
from dmdod.training import prestige1_finger_phase as fp
from dmdod.training import prestige1_finger_comparison as report


def tape(*rows):
    return tuple(
        (0., 0., 0., float(left), float(right), 0., 0., 0.)
        for left, right in rows)


def fake_result(mae=20., *, safe=True):
    hits = int(safe)
    return physics.PhysicalReplay(
        env=SimpleNamespace(key_names=(
            "left_4", "left_3", "left_2", "left_1",
            "right_1", "right_2", "right_3", "right_4")),
        checkpoints={0: object()},
        stats={"hits": hits, "targets": 1, "misses": int(not safe),
               "keydowns": hits, "too_early": 0, "unjudged": 0,
               "overload": False},
        hit_events=[{"target_index": 0, "error_ms": -mae}] if safe else [],
        timing={"hits": hits, "mae_ms": mae if safe else None},
        termination={"reason": "All targets resolved"},
        safe=safe, objective_ms=mae if safe else None,
    )


def test_independent_finger_phase_preserves_all_nonzero_commands():
    original = tape(
        (0, 0), (0, 0), (1, 0), (1, 1),
        (-1, 1), (-1, -1), (0, -1), (0, 0),
    )
    offsets = (0, 0, 0, -1, 2, 0, 0, 0)
    modified = fp.translate_finger_waveforms(original, offsets)
    assert len(modified) == len(original) + 2
    assert [row[3] for row in modified] == [
        0, 1, 1, -1, -1, 0, 0, 0, 0, 0]
    assert [row[4] for row in modified] == [
        0, 0, 0, 0, 0, 1, 1, -1, -1, 0]
    assert fp.translate_finger_waveforms(original, (0,) * 8) == original


def test_invalid_finger_offsets_fail_without_truncating_commands():
    original = tape((1, 0), (-1, 0), (0, 0))
    with pytest.raises(ValueError, match="truncated"):
        fp.translate_finger_waveforms(original, (0, 0, 0, -1, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="vector"):
        fp.translate_finger_waveforms(original, (0,) * 7)
    with pytest.raises(ValueError, match="vector"):
        fp.translate_finger_waveforms(original, (0, 0, 0, 3, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="vector"):
        fp.translate_finger_waveforms(original, (0, 0, 0, True, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="8-Key"):
        fp.translate_finger_waveforms(((1.,),), (0,) * 8)
    with pytest.raises(ValueError, match="nonempty"):
        fp.translate_finger_waveforms((), (0,) * 8)


def test_safe_only_coordinate_phases_select_per_key_improvements(monkeypatch):
    original = tape(
        (0, 0), (0, 0), (1, 0), (-1, 1),
        (0, -1), (0, 0), (0, 0), (0, 0),
    )
    # One candidate finger phase is unsafe, even when its MAE would
    # otherwise look attractive. The feasible optimum must win.
    calls = []
    def fake_replay(named, actions):
        commands = tuple(tuple(frame) for frame in actions)
        calls.append(commands)
        if commands == original:
            return fake_result(30.)
        left_index = next(i for i, row in enumerate(commands) if row[3] == 1.)
        right_index = next(i for i, row in enumerate(commands) if row[4] == 1.)
        if left_index == 3:
            return fake_result(2., safe=False)
        score = 30. - 5. * int(left_index == 1) - 6. * int(right_index == 4)
        return fake_result(score)

    monkeypatch.setattr(physics, "replay_commands", fake_replay)
    selected = fp.select_finger_phases(None, original)
    assert selected.best.safe
    assert selected.original.objective_ms == 30.
    assert selected.best.objective_ms == 19.
    assert selected.offsets[3] == -1
    assert selected.offsets[4] == 1
    assert selected.active_fingers == 2
    assert selected.improved_keys == 2
    assert selected.unsafe_miss >= 1
    assert calls[-1] == selected.commands, "independent re-run of selected plan"


def test_no_used_finger_is_ever_changed(monkeypatch):
    original = tape((0, 0), (0, 0))
    monkeypatch.setattr(physics, "replay_commands",
                        lambda *_: fake_result(11.))
    selected = fp.select_finger_phases(None, original)
    assert selected.offsets == (0,) * 8
    assert selected.active_fingers == 0
    assert selected.checked == 0
    assert selected.best.objective_ms == 11.
    assert selected.commands == original


def test_unsafe_finger_candidates_leave_original_tape_untouched(monkeypatch):
    original = tape((0, 0), (0, 0), (1, 0), (-1, 0))
    def fake_replay(named, commands):
        return fake_result(20., safe=tuple(commands) == original)
    monkeypatch.setattr(physics, "replay_commands", fake_replay)
    selected = fp.select_finger_phases(None, original)
    assert selected.offsets == (0,) * 8
    assert selected.best.safe
    assert selected.commands == original
    assert selected.unsafe_miss > 0


def test_local_sa_start_identity_is_mandatory(monkeypatch):
    original = tape((1, 0), (-1, 0))
    safe = fake_result(20.)
    phase = fp.FingerPhaseSelection(
        safe, safe, original, (0,) * 8,
        checked=0, physically_safe=0, unsafe_miss=0,
        unsafe_too_early=0, unsafe_overload=0,
        invalid_boundary=0, improved_keys=0, active_fingers=1,
    )
    monkeypatch.setattr(fp, "select_finger_phases", lambda *a: phase)
    monkeypatch.setattr(fp.local, "search_chart",
                        lambda *a, **kw: {
                            "baseline": fake_result(12.),
                            "best": fake_result(10.),
                            "best_commands": original,
                        })
    with pytest.raises(RuntimeError, match="diverged"):
        fp.optimize_finger_phases_and_pulses(None, original, seed=1)


def test_local_sa_can_only_finish_safe_nonregressing(monkeypatch):
    original = tape((1, 0), (-1, 0))
    safe = fake_result(20.)
    phase = fp.FingerPhaseSelection(
        safe, safe, original, (0,) * 8,
        checked=0, physically_safe=0, unsafe_miss=0,
        unsafe_too_early=0, unsafe_overload=0,
        invalid_boundary=0, improved_keys=0, active_fingers=1,
    )
    monkeypatch.setattr(fp, "select_finger_phases", lambda *a: phase)
    monkeypatch.setattr(fp.local, "search_chart",
                        lambda *a, **kw: {
                            "baseline": safe,
                            "best": fake_result(safe=False),
                            "best_commands": original,
                        })
    with pytest.raises(RuntimeError, match="SAFE|safe"):
        fp.optimize_finger_phases_and_pulses(None, original, seed=1)


def test_preexisting_v30_outputs_are_never_replaced(tmp_path, monkeypatch):
    cfg = p.PrestigeConfig(
        dataset="missing", output_dir=str(tmp_path),
        core_path="missing", old_best_path="missing", device="cpu")
    root = tmp_path / "diagnostics" / "prestige1"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        p, "_protect_old_best",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("Output must be refused before reading checkpoints")),
    )
    for name in (report.REPORT, report.PLANS):
        file = root / name
        file.write_text("preserved", encoding="utf-8")
        with pytest.raises(FileExistsError):
            report.run_train_finger_phase_annealing(cfg)
        assert file.read_text(encoding="utf-8") == "preserved"
        file.unlink()
