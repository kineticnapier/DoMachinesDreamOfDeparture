from __future__ import annotations

from math import isclose
from random import Random
from types import SimpleNamespace

import pytest

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_global_annealing as sa
from dmdod.training import prestige1_global_comparison as report


def actions(*lanes):
    """10ms commands, only left_1 (index 3) and right_1 (index 4)."""
    return tuple(
        tuple(float(value) if k in (3, 4) else 0. for k, value in
              enumerate((0., 0., 0., row[0], row[1], 0., 0., 0.)))
        for row in lanes
    )


def test_press_run_boundaries_and_safe_onset_delay():
    original = actions(
        (0, 0), (1, 0), (1, 1), (1, 1), (-1, 1),
        (0, -1), (1, 0), (1, 0), (0, 0),
    )
    runs = sa.press_runs(original)
    assert [(run.key_index, run.start, run.end, run.max_delay)
            for run in runs] == [
        (3, 1, 4, 2),
        (4, 2, 5, 2),
        (3, 6, 8, 1),
    ]
    zero = (0, 0, 0)
    assert sa.realize_schedule(original, runs, zero) == original
    changed = sa.realize_schedule(original, runs, (2, 1, 1))
    assert changed[1][3] == changed[2][3] == 0.
    assert changed[3][3] == 1.
    assert changed[2][4] == 0.
    assert changed[3][4] == 1.
    assert changed[6][3] == 0. and changed[7][3] == 1.
    assert changed[4][3] == -1., "original release remains untouched"
    assert sa.earliest_difference(runs, zero, (2, 1, 1)) == 1
    assert sa.earliest_difference(runs, (2, 1, 1), (1, 1, 1)) == 2
    with pytest.raises(ValueError, match="bounds"):
        sa.realize_schedule(original, runs, (3, 0, 0))
    with pytest.raises(ValueError, match="width"):
        sa.realize_schedule(original, runs, (0,))
    with pytest.raises(ValueError, match="No timing changes"):
        sa.earliest_difference(runs, zero, zero)


def test_a_zero_or_one_tick_press_run_cannot_disappear_from_schedule():
    original = actions((1, 0), (-1, 0), (1, 0), (1, 0), (-1, 0))
    runs = sa.press_runs(original)
    assert len(runs) == 1
    assert (runs[0].start, runs[0].end, runs[0].max_delay) == (2, 4, 1)


def test_invalid_motor_action_tape_is_rejected():
    with pytest.raises(ValueError, match="Empty"):
        sa.press_runs(())
    with pytest.raises(ValueError, match="alphabet"):
        sa.press_runs(((.3,) * 8,))
    with pytest.raises(ValueError, match="8-key"):
        sa.press_runs(((1.,) * 4,))


def test_sa_accepts_improvement_and_can_accept_feasible_uphill_move():
    assert sa.accept_sa(50., 49., .01, Random(1))
    assert sa.accept_sa(50., 50., .01, Random(1))
    class AlwaysZero:
        def random(self):
            return 0.
    class AlwaysOne:
        def random(self):
            return 1.
    assert sa.accept_sa(50., 51., 2., AlwaysZero())
    assert not sa.accept_sa(50., 51., 2., AlwaysOne())
    with pytest.raises(ValueError):
        sa.accept_sa(0., 1., 0., Random(1))
    with pytest.raises(ValueError):
        sa.accept_sa(float("nan"), 1., 1., Random(1))
    assert sa._temperature(0, 96) == sa.INITIAL_TEMPERATURE_MS
    assert isclose(sa._temperature(95, 96), sa.FINAL_TEMPERATURE_MS)


def test_annealing_mutations_stay_bounded_and_seeded():
    original = actions(*([(1, 1)] * 6 + [(0, 0)] + [(1, 0)] * 4))
    runs = sa.press_runs(original)
    zero = tuple(0 for _ in runs)
    a = Random(2440)
    b = Random(2440)
    for _ in range(70):
        next_a = sa.mutate_schedule(zero, runs, a)
        next_b = sa.mutate_schedule(zero, runs, b)
        assert next_a == next_b
        assert all(0 <= delay <= run.max_delay
                   for run, delay in zip(runs, next_a))
        assert next_a != zero


def test_physical_safety_requires_every_note_and_no_extra_keydowns():
    base = {
        "hits": 3, "misses": 0, "targets": 3, "unjudged": 0,
        "too_early": 0, "keydowns": 3, "overload": False,
    }
    assert sa.safe_physics(base, targets=3)
    for change in (
        {"hits": 2, "misses": 1},
        {"overload": True},
        {"too_early": 1},
        {"keydowns": 4},
        {"unjudged": 1},
    ):
        assert not sa.safe_physics({**base, **change}, targets=3)


def test_fixed_train20_report_gate_does_not_auto_authorize_student():
    reference = {
        "physics": dict(hits=850, keydowns=850),
        "timing": dict(hits=850, mae_ms=58.96),
    }
    valid = dict(safe=20, unsafe_anchors=[], hits=850, keydowns=850,
                 misses=0, too_early=0, unjudged=0)
    timing = dict(hits=850, mae_ms=12.)
    gate = report._report_gate(valid, timing, reference)
    assert gate["full_train20_safe850"]
    assert gate["timing_mae_below_v24"]
    assert not gate["student_training_permitted"]
    assert not gate["automatically_adopted"]
    assert not report._report_gate(
        {**valid, "hits": 849}, timing, reference)["full_train20_safe850"]


def test_v27_refuses_replacing_existing_report_or_action_plan(tmp_path, monkeypatch):
    cfg = p.PrestigeConfig(
        dataset="missing", output_dir=str(tmp_path),
        core_path="missing", old_best_path="missing", device="cpu")
    target = tmp_path / "diagnostics" / "prestige1"
    target.mkdir(parents=True)
    monkeypatch.setattr(p, "_protect_old_best",
                        lambda *_: (_ for _ in ()).throw(
                            AssertionError("Should reject before checkpoint reads")))
    for filename in (report.REPORT, report.PLANS):
        protected = target / filename
        protected.write_text("untouched", encoding="utf-8")
        with pytest.raises(FileExistsError):
            report.run_train_global_annealing(cfg)
        assert protected.read_text(encoding="utf-8") == "untouched"
        protected.unlink()


def test_real_1ms_motor_replay_and_prefix_snapshot_match():
    """Small actual ADOFAI segment; no training or Student feature access."""
    from dmdod.adofai_chart import parse_adofai_text
    from dmdod.adofai_playable import build_playable_segment
    from dmdod.adofai_timing import compile_adofai
    from dmdod.motor.capacity import calibrate_n_key_press_lead

    chart = parse_adofai_text(
        '{"angleData":[0,0,0],'
        '"settings":{"bpm":120,"pitch":100,"countdownTicks":0,'
        '"separateCountdownTime":false},"actions":[]}'
    )
    compiled = compile_adofai(chart)
    segment = build_playable_segment(
        compiled, start_s=0., end_s=compiled.duration_s)
    assert len(segment.targets) >= 2
    named = SimpleNamespace(segment=segment)
    latency = dict(calibrate_n_key_press_lead(8).key_latencies_s)
    end_frame = int(segment.duration_s / .01) + 70
    tape = [[0.] * 8 for _ in range(end_frame)]
    for index, target in enumerate(segment.targets):
        key = 3 if index % 2 == 0 else 4
        inner_key = ("left_1" if key == 3 else "right_1")
        start = max(0, round(
            (target.episode_time_s - latency[inner_key]) / .01))
        for i in range(start, min(end_frame, start + 9)):
            tape[i][key] = 1.
        for i in range(start + 9, min(end_frame, start + 16)):
            tape[i][key] = -1.
    original = tuple(tuple(frame) for frame in tape)
    full = sa.replay_commands(named, original)
    assert full.safe, full.stats
    assert 0 in full.checkpoints
    assert sa.SNAPSHOT_STRIDE in full.checkpoints
    frame = sa.SNAPSHOT_STRIDE + 1
    changed = list(original)
    new_frame = list(changed[frame])
    new_frame[0] = 1.
    changed[frame] = tuple(new_frame)
    changed = tuple(changed)
    from_prefix = sa.replay_commands(
        named, changed, parent=full, first_changed_frame=frame)
    from_start = sa.replay_commands(named, changed)
    assert from_prefix.stats == from_start.stats
    assert from_prefix.hit_events == from_start.hit_events
    assert from_prefix.termination == from_start.termination
    assert from_prefix.timing == from_start.timing
