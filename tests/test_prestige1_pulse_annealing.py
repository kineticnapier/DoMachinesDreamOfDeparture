from __future__ import annotations

from random import Random
from types import SimpleNamespace

import pytest

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_global_annealing as physical
from dmdod.training import prestige1_pulse_annealing as pulse
from dmdod.training import prestige1_pulse_comparison as report


def tape_from_two_fingers(*rows):
    return tuple(
        (0., 0., 0., float(left), float(right), 0., 0., 0.)
        for left, right in rows)


def test_full_pulse_shifts_press_and_release_together():
    original = tape_from_two_fingers(
        (0, 0), (1, 0), (1, 1), (-1, 1), (-1, -1),
        (0, -1), (0, 0), (0, 0),
        (1, 0), (1, 0), (-1, 0), (0, 0),
    )
    windows = pulse.pulse_windows(original)
    assert [(p.key_index, p.start, p.positive_end, p.tail_end,
             p.max_shift) for p in windows] == [
        (3, 1, 3, 5, 3),
        (4, 2, 4, 6, 6),
        (3, 8, 10, 11, 1),
    ]
    assert pulse.shifted_tape(original, windows, (0, 0, 0)) == original
    moved = pulse.shifted_tape(original, windows, (2, 1, 0))
    # Left positive run keeps its original *duration* of 2 frames.
    assert [row[3] for row in moved] == [
        0, 0, 0, 1, 1, -1, -1, 0, 1, 1, -1, 0]
    # Right release also moves, not just its key-down command.
    assert [row[4] for row in moved] == [
        0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0]
    assert pulse.changed_first(original, moved) == 1
    assert pulse.changed_first(original, original) is None


def test_shift_is_structurally_bounded_and_fails_closed():
    original = tape_from_two_fingers(
        (1, 0), (1, 0), (-1, 0), (0, 0), (1, 0),
        (-1, 0), (0, 0),
    )
    windows = pulse.pulse_windows(original)
    assert [(w.start, w.max_shift) for w in windows] == [(0, 1), (4, 1)]
    with pytest.raises(ValueError, match="shift"):
        pulse.shifted_tape(original, windows, (2, 0))
    with pytest.raises(ValueError, match="length"):
        pulse.shifted_tape(original, windows, (0,))
    with pytest.raises(ValueError, match="8-key"):
        pulse.pulse_windows(((1.,),))
    with pytest.raises(ValueError, match="Empty"):
        pulse.pulse_windows(())
    with pytest.raises(ValueError, match="durations"):
        pulse.changed_first(original, original[:-1])


def test_sequential_three_pulse_mutation_is_seed_reproducible():
    original = tape_from_two_fingers(*(
        [(1, 0), (1, 0), (-1, 0), (0, 0), (0, 0)] * 5))
    windows = pulse.pulse_windows(original)
    assert len(windows) == 5
    state = tuple(0 for _ in windows)
    r1 = Random(123)
    r2 = Random(123)
    results = []
    for _ in range(40):
        a = pulse.mutate(state, windows, r1)
        b = pulse.mutate(state, windows, r2)
        assert a == b and a is not None
        assert a != state
        assert all(0 <= v <= w.max_shift for v, w in zip(a, windows))
        results.append(a)
    assert len(set(results)) > 1


def test_safe_only_temperature_acceptance():
    assert pulse.accepts(50, 40, 0.01, Random(0))
    assert pulse.accepts(50, 50, 0.01, Random(0))
    class Zero:
        def random(self):
            return 0.
    class One:
        def random(self):
            return 1.
    assert pulse.accepts(50, 51, 3., Zero())
    assert not pulse.accepts(50, 51, 3., One())
    assert pulse.temperature(0, 96) == pulse.START_TEMPERATURE_MS
    assert pulse.temperature(95, 96) == pytest.approx(
        pulse.END_TEMPERATURE_MS)
    with pytest.raises(ValueError):
        pulse.accepts(50., 51., 0., Random(0))
    with pytest.raises(ValueError):
        pulse.accepts(float("inf"), 51., 3., Random(0))


def test_repair_uses_evaluator_miss_only_and_never_proposes_current():
    windows = (
        pulse.Pulse(3, 10, 12, 14, 3),
        pulse.Pulse(4, 36, 38, 40, 3),
        pulse.Pulse(3, 43, 45, 48, 3),
    )
    env = SimpleNamespace(
        _missed=[False, True],
        segment=SimpleNamespace(
            targets=[
                SimpleNamespace(episode_time_s=.1),
                SimpleNamespace(episode_time_s=.48),
            ]),
    )
    trial = SimpleNamespace(env=env)
    original = (0, 0, 0)
    unsafe = (0, 1, 0)
    repairs = pulse.repair_neighbors(unsafe, original, windows, trial)
    assert repairs and len(repairs) <= pulse.REPAIR_TRIALS_PER_UNSAFE
    assert all(candidate != original for candidate in repairs)
    assert all(sum(a != b for a, b in zip(candidate, unsafe)) == 1
               for candidate in repairs)
    env._missed = [False, False]
    assert pulse.repair_neighbors(unsafe, original, windows, trial) == ()


def test_unsafe_repair_does_not_enter_sa_chain_if_no_safe_candidate(monkeypatch):
    original = tape_from_two_fingers(
        (1, 0), (1, 0), (-1, 0), (-1, 0),
        (0, 0), (0, 0), (0, 0), (0, 0))
    windows = pulse.pulse_windows(original)
    assert len(windows) == 1

    def fake_replay(named, actions, *, parent=None, first_changed_frame=0):
        feasible = actions == original
        stats = dict(
            hits=int(feasible), misses=int(not feasible), targets=1,
            unjudged=0, too_early=0,
            keydowns=int(feasible), overload=False)
        env = SimpleNamespace(
            _missed=[not feasible],
            segment=SimpleNamespace(
                targets=[SimpleNamespace(episode_time_s=.20)]))
        return physical.PhysicalReplay(
            env=env, checkpoints={0: object()}, stats=stats,
            hit_events=[{"error_ms": -50.}] if feasible else [],
            timing={"hits": int(feasible),
                    "mae_ms": 50. if feasible else None},
            termination={"reason": "All targets resolved"},
            safe=feasible, objective_ms=50. if feasible else None)

    monkeypatch.setattr(physical, "replay_commands", fake_replay)
    result = pulse.search_chart(None, original, seed=123, proposals=10)
    assert result["best_commands"] == original
    assert result["best"].safe
    assert result["best"].objective_ms == 50.
    assert result["counts"]["rejected_unsafe_main"] > 0
    assert result["counts"]["best_updates"] == 0


def test_physical_repair_can_cross_infeasible_step_without_keeping_miss(monkeypatch):
    original = tape_from_two_fingers(
        (1, 0), (1, 0), (-1, 0), (-1, 0),
        (0, 0), (0, 0), (0, 0), (0, 0))
    windows = pulse.pulse_windows(original)
    assert len(windows) == 1
    shifted_one = pulse.shifted_tape(original, windows, (1,))
    shifted_two = pulse.shifted_tape(original, windows, (2,))
    calls = []

    def fake_replay(named, actions, *, parent=None, first_changed_frame=0):
        calls.append((actions, parent))
        feasible = actions in (original, shifted_two)
        score = 30. if actions == shifted_two else 50. if feasible else None
        stats = dict(
            hits=int(feasible), misses=int(not feasible), targets=1,
            unjudged=0, too_early=0, keydowns=int(feasible), overload=False)
        return physical.PhysicalReplay(
            env=SimpleNamespace(
                _missed=[not feasible],
                segment=SimpleNamespace(
                    targets=[SimpleNamespace(episode_time_s=.20)])),
            checkpoints={0: object()}, stats=stats,
            hit_events=[{"error_ms": -score}] if feasible else [],
            timing={"hits": int(feasible), "mae_ms": score},
            termination={"reason": "All targets resolved"},
            safe=feasible, objective_ms=score)

    monkeypatch.setattr(physical, "replay_commands", fake_replay)
    monkeypatch.setattr(pulse, "mutate", lambda *args: (1,))
    result = pulse.search_chart(None, original, seed=1, proposals=1)
    assert result["best_commands"] == shifted_two
    assert result["best"].safe
    assert result["best"].objective_ms == 30.
    assert result["counts"]["rejected_unsafe_main"] == 1
    assert result["counts"]["repair_attempts"] >= 1
    assert result["counts"]["repair_selected"] == 1
    assert calls[-1][1] is None, "best candidate needs full replay from frame 0"
    assert shifted_one not in (result["best_commands"], original)


def test_refuse_ever_overwriting_existing_v28_outputs(tmp_path, monkeypatch):
    cfg = p.PrestigeConfig(
        dataset="missing", output_dir=str(tmp_path),
        core_path="missing", old_best_path="missing", device="cpu")
    root = tmp_path / "diagnostics" / "prestige1"
    root.mkdir(parents=True)
    monkeypatch.setattr(p, "_protect_old_best", lambda *_: (
        (_ for _ in ()).throw(AssertionError("Must refuse before checkpoint reads"))))
    for filename in (report.REPORT, report.PLANS):
        file = root / filename
        file.write_text("immutable", encoding="utf-8")
        with pytest.raises(FileExistsError):
            report.run_train_full_pulse_annealing(cfg)
        assert file.read_text(encoding="utf-8") == "immutable"
        file.unlink()
