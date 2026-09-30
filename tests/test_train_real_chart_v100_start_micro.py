from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v100_start_micro as v100


def _runtime(times, *, midspins=None, duration=20.0, name="chart"):
    midspins = set(midspins or [])
    floors = [
        SimpleNamespace(index=0, target_time_s=0.0, midspin=False),
        *[
            SimpleNamespace(
                index=index,
                target_time_s=float(value),
                midspin=index in midspins,
            )
            for index, value in enumerate(times, 1)
        ],
    ]
    return SimpleNamespace(
        compiled=SimpleNamespace(floors=floors),
        duration_s=float(duration),
        spec=SimpleNamespace(name=name),
    )


def test_start_micro_end_uses_nth_playable_target_and_skips_midspin():
    runtime = _runtime([1.0, 1.2, 1.5, 1.8, 2.1], midspins={2})
    assert v100._start_micro_end_s(runtime, 4) == 2.1


def test_start_micro_end_handles_short_chart():
    runtime = _runtime([0.7, 1.0], duration=1.4)
    assert v100._start_micro_end_s(runtime, 4) == 1.0


def test_augmented_anchors_keep_existing_and_add_one_micro_per_chart(monkeypatch):
    charts = [
        _runtime([1.0, 1.3, 1.6, 1.9, 2.2], name="a"),
        _runtime([0.8, 1.1, 1.4, 1.7, 2.0], name="b"),
    ]
    base = [SimpleNamespace(key=("base", 1)), SimpleNamespace(key=("base", 2))]
    created = []

    monkeypatch.setattr(
        v100,
        "_BASE_BUILD_ANCHOR_SEGMENTS",
        lambda train_charts, *, window_s, anchors_per_chart: list(base),
    )

    def fake_named(chart, role, start, end):
        named = SimpleNamespace(
            key=(role, chart.spec.name, start, end),
            role=role,
            chart_name=chart.spec.name,
            start_s=start,
            end_s=end,
        )
        created.append(named)
        return named

    monkeypatch.setattr(v100.v080, "_named_segment", fake_named)
    monkeypatch.setattr(v100, "_START_MICRO_TARGETS", 4)

    result = v100._build_anchor_segments_with_start_micro(
        charts,
        window_s=30.0,
        anchors_per_chart=2,
    )

    assert result[:2] == base
    assert len(result) == 4
    assert [item.role for item in created] == ["start-micro-4", "start-micro-4"]
    assert [item.start_s for item in created] == [0.0, 0.0]
    assert [item.end_s for item in created] == [1.9, 1.7]


def test_install_start_micro_updates_turbo_builder_and_checkpoint_identity(monkeypatch):
    old_builder = v100.turbo._ORIGINAL_BUILD_ANCHOR_SEGMENTS
    old_version = v100.v080.TRAINER_VERSION
    old_checkpoint = v100.v080.DEFAULT_CHECKPOINT
    try:
        v100.install_start_micro(target_count=5)
        assert v100._START_MICRO_TARGETS == 5
        assert v100.turbo._ORIGINAL_BUILD_ANCHOR_SEGMENTS is v100._build_anchor_segments_with_start_micro
        assert v100.v080.TRAINER_VERSION == v100.TRAINER_VERSION
        assert v100.v080.DEFAULT_CHECKPOINT == v100.DEFAULT_CHECKPOINT
    finally:
        monkeypatch.setattr(v100.turbo, "_ORIGINAL_BUILD_ANCHOR_SEGMENTS", old_builder)
        monkeypatch.setattr(v100.v080, "TRAINER_VERSION", old_version)
        monkeypatch.setattr(v100.v080, "DEFAULT_CHECKPOINT", old_checkpoint)
