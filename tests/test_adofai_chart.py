from dataclasses import fields

import pytest

from dmdod.adofai_chart import parse_adofai_bytes, parse_adofai_text
from dmdod.adofai_geometry import build_floor_geometry
from dmdod.adofai_timing import compile_adofai


def _chart(actions: str = "[]", *, settings: str = '"bpm": 120, "countdownTicks": 0'):
    return parse_adofai_text(
        f"""
        {{
          "angleData": [0, 0, 0],
          "settings": {{{settings}}},
          "actions": {actions}
        }}
        """
    )


def test_parser_accepts_loose_json_numeric_strings_and_utf16():
    text = """
    {
      // generated chart
      "angleData": ["0", 90,],
      "settings": {"bpm": "120", "countdownTicks": "0",},
      "actions": [
        {"floor": "1", "eventType": "Twirl",},
      ],
    }
    """
    chart = parse_adofai_text(text)
    assert chart.angles == (0.0, 90.0)
    assert chart.initial_bpm == 120.0
    assert chart.countdown_ticks == 0
    assert chart.actions[0].floor == 1
    assert chart.actions[0].event_type == "Twirl"

    utf16 = b"\xff\xfe" + text.encode("utf-16-le")
    assert parse_adofai_bytes(utf16).angles == chart.angles


def test_parser_accepts_raw_control_characters_inside_strings():
    # Real-world ADOFAI files can contain technically-invalid JSON strings,
    # especially raw tabs/newlines copied into metadata.  Stock/community tools
    # are tolerant of these, so the simulator importer must be as well.
    text = '{"angleData":[0],"settings":{"bpm":120,"songFilename":"line1\nline2\t.ogg"},"actions":[]}'
    chart = parse_adofai_text(text)
    assert chart.song_filename == "line1\nline2\t.ogg"


def test_geometry_matches_extremeeditor_pathbuilder_convention():
    geometry = build_floor_geometry(_chart())
    assert len(geometry) == 4
    assert (geometry[0].x, geometry[0].y) == pytest.approx((0.0, 0.0))
    assert (geometry[1].x, geometry[1].y) == pytest.approx((1.5, 0.0), abs=1e-9)
    assert (geometry[2].x, geometry[2].y) == pytest.approx((3.0, 0.0), abs=1e-9)
    assert (geometry[3].x, geometry[3].y) == pytest.approx((4.5, 0.0), abs=1e-9)


def test_setspeed_changes_floor_entry_times():
    chart = _chart(
        """[
          {"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": 240}
        ]"""
    )
    compiled = compile_adofai(chart)
    assert compiled.floors[1].target_time_s == pytest.approx(0.5, abs=1e-5)
    assert compiled.floors[2].target_time_s == pytest.approx(0.75, abs=1e-5)
    assert compiled.floors[2].bpm == pytest.approx(240.0, abs=1e-4)


def test_midfloor_setspeed_angle_offset_uses_weighted_floor_time():
    chart = _chart(
        """[
          {"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": 240, "angleOffset": 90}
        ]"""
    )
    compiled = compile_adofai(chart)
    # Floor 1 is a 180-degree turn: 90 degrees at 120 BPM (.25 s), then
    # 90 degrees at 240 BPM (.125 s).
    assert compiled.floors[2].target_time_s == pytest.approx(0.875, abs=2e-5)
    assert compiled.floors[1].exit_time_s - compiled.floors[1].target_time_s == pytest.approx(0.375, abs=2e-5)


def test_pause_adds_beats_before_rotation():
    chart = _chart(
        """[
          {"floor": 1, "eventType": "Pause", "duration": 2}
        ]"""
    )
    compiled = compile_adofai(chart)
    assert compiled.floors[1].pause_s == pytest.approx(1.0)
    assert compiled.floors[2].target_time_s == pytest.approx(2.0, abs=1e-5)


def test_twirl_and_three_planets_are_carried_into_compiled_floor():
    chart = _chart(
        """[
          {"floor": 1, "eventType": "Twirl"},
          {"floor": 1, "eventType": "MultiPlanet", "planets": "ThreePlanets"}
        ]"""
    )
    compiled = compile_adofai(chart)
    assert compiled.floors[1].is_ccw is True
    assert compiled.floors[1].num_planets == 3
    assert compiled.floors[1].event_markers == ("Twirl", "MultiPlanet")


def test_policy_visible_floor_cannot_access_privileged_time_or_bpm():
    compiled = compile_adofai(_chart())
    visible = compiled.floors[1].visible()
    names = {item.name for item in fields(visible)}
    assert "target_time_s" not in names
    assert "exit_time_s" not in names
    assert "bpm" not in names
    assert names >= {"x", "y", "entry_angle_rad", "exit_angle_rad", "midspin"}


def test_segment_selection_and_audio_chart_time_transform():
    chart = _chart(
        """[
          {"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm", "beatsPerMinute": 240}
        ]""",
        settings='"bpm": 120, "countdownTicks": 4, "separateCountdownTime": true, "offset": 100, "pitch": 50',
    )
    compiled = compile_adofai(chart)
    audio = compiled.chart_to_audio_time(2.0)
    assert audio == pytest.approx(0.2)
    assert compiled.audio_to_chart_time(audio) == pytest.approx(2.0)

    entries = compiled.floor_entries_between(1.99, 2.26)
    assert entries
    assert all(1.99 <= floor.target_time_s <= 2.26 for floor in entries)
