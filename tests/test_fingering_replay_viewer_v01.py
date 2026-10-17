from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_fingering_replay_viewer_v01 as replay

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


def _payload() -> dict:
    trace = [[0.0, [0.0] * 12, [0.0] * 12, [0] * 12,
              ["off", "off"], [0, 0]]]
    metrics = {"targets": 1, "hits": 1, "misses": 0, "too_early": 0,
               "overloaded": False, "x_accuracy_percent": 100.0,
               "timing_mae_ms": 2.0, "physical_keydowns": 1}
    return {
        "format": replay.FORMAT, "chart": "x </script>.adofai", "chart_sha256": "a" * 64,
        "difficulty": "P20", "checkpoint": "model.pt", "checkpoint_sha256": "b" * 64,
        "timebase": {"episode_duration_s": 1.0, "segment_start_chart_s": 2.0,
                     "segment_end_chart_s": 3.0, "pitch_percent": 100.0,
                     "pitch_ratio": 1.0, "offset_ms": 0.0, "countdown_ticks": 0,
                     "initial_bpm": 120.0, "separate_countdown_time": False,
                     "separate_countdown_chart_s": 0.0, "formula": {}},
        "floors": [{"index": 1, "x": 0.0, "y": 0.0, "episode_s": 0.0,
                    "exit_episode_s": 1.0, "chart_s": 2.0, "bpm": 120.0,
                    "midspin": False, "is_ccw": False, "num_planets": 2,
                    "pause_chart_s": 0.0, "markers": ["SetSpeed"]}],
        "events": [{"event_index": 0, "episode_s": .5, "chart_s": 2.5,
                    "audio_s": 2.5, "floor_indices": [1], "chord_size": 1,
                    "teacher": None, "network": None}],
        "policies": {"teacher": {"trace": trace, "keydowns": [], "saved_metrics": metrics},
                     "network": {"trace": trace, "keydowns": [], "saved_metrics": metrics}},
        "rendering_limitations": [],
    }


def test_parse_allocation_preserves_physical_mode():
    fingers = replay.parse_allocation("left_5/flat + right_2")
    assert [(finger.axis, finger.mode) for finger in fingers] == [
        ("left_5", "flat"), ("right_2", "single")]


def test_metric_match_rejects_replay_drift():
    metrics = _payload()["policies"]["network"]["saved_metrics"]
    replay._assert_metric_match("Network", dict(metrics), metrics)
    changed = dict(metrics, hits=0)
    try:
        replay._assert_metric_match("Network", changed, metrics)
    except RuntimeError as error:
        assert "replay drift" in str(error)
    else:
        raise AssertionError("metric drift was accepted")


def test_html_is_self_contained_and_has_required_controls():
    html = replay.render_html(_payload())
    assert "requestAnimationFrame" in html
    assert 'id="seek"' in html and 'id="policy"' in html
    assert 'id="prev"' in html and 'id="next"' in html and 'id="jump"' in html
    assert 'id="prevBad"' in html and 'id="nextBad"' in html
    assert ".25" in html and ".5" in html and "Teacher" in html and "Network" in html
    assert "https://" not in html and "<script src=" not in html
    assert "</script>.adofai" not in html
    assert "診断用補間" in html and "Midspin" in html


def test_compiled_floor_export_preserves_midspin_pause_speed_and_timebase():
    compiled = compile_adofai(parse_adofai_text('''{
      "angleData":[0,999,90,180,270],
      "settings":{"bpm":120,"pitch":80,"countdownTicks":2,"separateCountdownTime":true},
      "actions":[
        {"floor":1,"eventType":"Pause","duration":1},
        {"floor":2,"eventType":"SetSpeed","speedType":"Multiplier","bpmMultiplier":2},
        {"floor":3,"eventType":"Twirl"}
      ]
    }'''))
    segment = build_playable_segment(compiled)
    floors = replay._floor_rows(compiled, segment)
    assert any(floor["midspin"] for floor in floors)
    assert any("Pause" in floor["markers"] for floor in floors)
    assert any("SetSpeed" in floor["markers"] for floor in floors)
    assert any("Twirl" in floor["markers"] for floor in floors)
    assert any(floor["pause_chart_s"] > 0 for floor in floors)
    assert len({floor["bpm"] for floor in floors}) > 1
    assert [floor["episode_s"] for floor in floors] == sorted(
        floor["episode_s"] for floor in floors
    )
    for value in (0.0, segment.duration_s * .5, segment.duration_s):
        chart_time = segment.chart_time_from_episode(value)
        assert segment.episode_time_from_chart(chart_time) == value
