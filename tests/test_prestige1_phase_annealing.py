from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_global_annealing as physics
from dmdod.training import prestige1_phase_annealing as phase
from dmdod.training import prestige1_phase_comparison as report


def tape(*rows):
    return tuple(
        (0., 0., 0., float(row[0]), float(row[1]), 0., 0., 0.)
        for row in rows)


def fake_result(*, safe=True, mae=20.):
    hits = int(safe)
    return physics.PhysicalReplay(
        env=SimpleNamespace(key_names=["left_4", "left_3", "left_2",
                                       "left_1", "right_1", "right_2",
                                       "right_3", "right_4"]),
        checkpoints={0: object()},
        stats={"hits": hits, "targets": 1, "misses": int(not safe),
               "keydowns": hits, "too_early": 0, "unjudged": 0,
               "overload": False},
        hit_events=[{"target_index": 0, "error_ms": -mae}] if safe else [],
        timing={"hits": hits, "mae_ms": mae if safe else None},
        termination={"reason": "All targets resolved"},
        safe=safe, objective_ms=mae if safe else None,
    )


def test_global_phase_keeps_every_original_waveform_command():
    original = tape((1, 0), (1, -1), (-1, 1), (0, -1))
    neutral = (0.,) * 8
    assert phase.global_phase_tape(original, 0) == original
    shifted = phase.global_phase_tape(original, 2)
    assert shifted == (neutral, neutral) + original
    assert len(shifted) == len(original) + 2
    assert shifted[3][4] == -1., "preexisting motor brake is preserved"
    assert shifted[4][3] == -1., "press release duration is preserved"
    with pytest.raises(ValueError, match="phase"):
        phase.global_phase_tape(original, -1)
    with pytest.raises(ValueError, match="phase"):
        phase.global_phase_tape(original, 9)
    with pytest.raises(ValueError, match="8-Key"):
        phase.global_phase_tape(((1.,),), 0)
    with pytest.raises(ValueError, match="nonempty"):
        phase.global_phase_tape((), 0)


def test_global_phase_chooses_minimum_mae_among_physically_safe(monkeypatch):
    original = tape((1, 0), (1, 0), (-1, 0))
    neutral = (0.,) * 8
    scores = {0: 30., 1: 20., 2: None, 3: 24., 4: 12.}
    checked = []

    def fake_replay(named, candidate):
        shift = len(candidate) - len(original)
        assert candidate == (neutral,) * shift + original
        checked.append(shift)
        return fake_result(safe=scores[shift] is not None,
                           mae=scores[shift] or 999.)

    monkeypatch.setattr(physics, "replay_commands", fake_replay)
    selected = phase.select_global_phase(None, original, max_ticks=4)
    assert checked == list(range(5))
    assert selected.candidates == 5
    assert selected.safe_candidates == 4
    assert selected.unsafe_miss_candidates == 1
    assert selected.original.objective_ms == 30.
    assert selected.ticks == 4
    assert selected.phase_best.objective_ms == 12.
    assert selected.best_tape == (neutral,) * 4 + original


def test_global_phase_retains_original_when_every_delay_is_unsafe(monkeypatch):
    original = tape((1, 0), (-1, 0))

    def fake_replay(named, candidate):
        return fake_result(safe=candidate == original, mae=26.)

    monkeypatch.setattr(physics, "replay_commands", fake_replay)
    result = phase.select_global_phase(None, original, max_ticks=8)
    assert result.ticks == 0
    assert result.safe_candidates == 1
    assert result.unsafe_miss_candidates == 8
    assert result.best_tape == original


def test_phase_then_pulse_rejects_mismatched_local_start(monkeypatch):
    original = tape((1, 0), (-1, 0))
    initial = fake_result(mae=20.)
    selected = phase.PhaseSelection(
        original, initial, original, 0, 9, 1, 8, 0, 0)
    monkeypatch.setattr(phase, "select_global_phase", lambda *a, **kw: selected)
    local_initial = fake_result(mae=17.)
    local_best = fake_result(mae=10.)
    monkeypatch.setattr(
        phase.local, "search_chart",
        lambda *a, **kw: {
            "baseline": local_initial, "best": local_best,
            "best_commands": original,
        },
    )
    with pytest.raises(RuntimeError, match="diverged"):
        phase.optimize_phase_and_pulses(None, original, seed=1)


def test_phase_then_pulse_rejects_safety_regression(monkeypatch):
    original = tape((1, 0), (-1, 0))
    baseline = fake_result(mae=20.)
    selected = phase.PhaseSelection(
        baseline, baseline, original, 0, 9, 1, 8, 0, 0)
    monkeypatch.setattr(phase, "select_global_phase", lambda *a, **kw: selected)
    monkeypatch.setattr(
        phase.local, "search_chart",
        lambda *a, **kw: {
            "baseline": baseline, "best": fake_result(safe=False),
            "best_commands": original,
        },
    )
    with pytest.raises(RuntimeError, match="SAFE"):
        phase.optimize_phase_and_pulses(None, original, seed=1)


def test_phase_then_pulse_accepts_only_nonregressing_safe_output(monkeypatch):
    original = tape((1, 0), (-1, 0))
    initial = fake_result(mae=30.)
    aligned = fake_result(mae=20.)
    best = fake_result(mae=10.)
    selected = phase.PhaseSelection(
        initial, aligned, (tape((0, 0))[0],) + original,
        1, 9, 9, 0, 0, 0)
    monkeypatch.setattr(phase, "select_global_phase", lambda *a, **kw: selected)
    monkeypatch.setattr(
        phase.local, "search_chart",
        lambda *a, **kw: {
            "baseline": aligned, "best": best,
            "best_commands": selected.best_tape,
            "pulse_windows": (), "best_shifts": (),
            "counts": {}, "seed": 123,
        },
    )
    result = phase.optimize_phase_and_pulses(None, original, seed=123)
    assert result["best"].objective_ms == 10.
    assert result["phase_ticks"] == 1
    assert result["best_commands"] == selected.best_tape


def test_existing_v29_report_or_plan_is_never_overwritten(tmp_path, monkeypatch):
    cfg = p.PrestigeConfig(
        dataset="missing", output_dir=str(tmp_path),
        core_path="missing", old_best_path="missing", device="cpu")
    dest = tmp_path / "diagnostics" / "prestige1"
    dest.mkdir(parents=True)
    monkeypatch.setattr(
        p, "_protect_old_best",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("Pre-existing output must be checked first")),
    )
    for name in (report.REPORT, report.PLANS):
        path = dest / name
        path.write_text("protected", encoding="utf-8")
        with pytest.raises(FileExistsError):
            report.run_train_phase_aligned_annealing(cfg)
        assert path.read_text(encoding="utf-8") == "protected"
        path.unlink()
