from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_fingering_replay_viewer_v03 as v03

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


def _compiled(actions="[]", angles=(0, 0, 0), bpm=120):
    chart = parse_adofai_text(json.dumps({
        "angleData": list(angles),
        "settings": {"bpm": bpm, "pitch": 100, "countdownTicks": 0},
        "actions": json.loads(actions),
    }))
    compiled = compile_adofai(chart)
    segment = build_playable_segment(compiled)
    return chart, compiled, segment, v03.rotation_profiles(chart, compiled, segment)


def test_key_counts_total_kps_and_seek_recompute_from_log():
    rows = [
        {"episode_s": .1, "contact": "left_1"},
        {"episode_s": .9, "contact": "right_8"},
        {"episode_s": 1.2, "contact": "left_1"},
    ]
    late = v03.key_overlay_snapshot(rows, 1.2)
    assert late["total"] == 3 and late["kps"] == 2
    assert sum(late["counts"].values()) == late["total"]
    assert late["counts"]["left_1"] == 2
    early = v03.key_overlay_snapshot(rows, .5)
    assert early["total"] == 1 and early["kps"] == 1
    assert early["counts"]["right_8"] == 0


def test_two_planet_orbit_keeps_radius_and_reaches_next_floor():
    _, _, _, floors = _compiled()
    floor = floors[1]
    middle = (floor["rotation_start_s"] + floor["rotation_end_s"]) * .5
    x, y = v03.orbit_point(floor, middle)
    assert math.hypot(x - floor["x"], y - floor["y"]) == pytest.approx(1.5)
    x, y = v03.orbit_point(floor, floor["rotation_end_s"])
    assert (x, y) == pytest.approx((floors[2]["x"], floors[2]["y"]), abs=2e-6)


def test_boundary_swaps_roles_without_position_teleport():
    _, _, _, floors = _compiled()
    before_fixed = (floors[1]["x"], floors[1]["y"])
    arriving = v03.orbit_point(floors[1], floors[1]["rotation_end_s"])
    after_fixed = (floors[2]["x"], floors[2]["y"])
    after_moving = v03.orbit_point(floors[2], floors[2]["rotation_start_s"])
    assert arriving == pytest.approx(after_fixed, abs=2e-6)
    assert after_moving == pytest.approx(before_fixed, abs=2e-6)
    assert floors[2]["fixed_identity"] != floors[1]["fixed_identity"]


def test_twirl_reverses_direction_and_full_turn_is_not_shortest_arc():
    actions = '[{"floor":1,"eventType":"Twirl"}]'
    _, _, _, floors = _compiled(actions=actions, angles=(0, 0, 0))
    assert floors[1]["rotation_sign"] == -1
    _, _, _, full = _compiled(angles=(0, 180, 180))
    full_turns = [row for row in full if row["angle_moved_rad"] > 6.0]
    assert full_turns and full_turns[0]["angle_moved_rad"] == pytest.approx(math.tau, abs=2e-6)


def test_pause_midspin_setspeed_and_multiplanet_profiles():
    actions = json.dumps([
        {"floor": 1, "eventType": "Pause", "duration": 2},
        {"floor": 1, "eventType": "SetSpeed", "speedType": "Bpm",
         "beatsPerMinute": 240, "angleOffset": 90},
        {"floor": 2, "eventType": "MultiPlanet", "planets": "ThreePlanets"},
    ])
    _, _, _, floors = _compiled(actions=actions, angles=(0, 999, 0, 0))
    paused = floors[1]
    assert paused["rotation_start_s"] > paused["episode_s"]
    assert v03.orbit_point(paused, paused["episode_s"]) == pytest.approx(
        v03.orbit_point(paused, paused["rotation_start_s"]), abs=1e-9)
    assert paused["midspin"] and paused["angle_moved_rad"] == 0
    assert floors[2]["num_planets"] == 3

    speed_actions = '[{"floor":1,"eventType":"SetSpeed","speedType":"Bpm","beatsPerMinute":240,"angleOffset":90}]'
    _, _, _, speed = _compiled(actions=speed_actions)
    assert len(speed[1]["motion_segments"]) == 2
    first, second = speed[1]["motion_segments"]
    assert first["end_s"] - first["start_s"] == pytest.approx(.25, abs=2e-5)
    assert second["end_s"] - second["start_s"] == pytest.approx(.125, abs=2e-5)


def _html_payload():
    trace = [[0.0, [0.0] * 12, [0.0] * 12, [0] * 12, ["off", "off"], [0, 0],
              {"hits": 0, "misses": 0, "too_early": 0, "overloaded": False}]]
    metrics = {"targets": 1, "hits": 1, "misses": 0, "too_early": 0,
               "overloaded": False, "x_accuracy_percent": 100.0, "timing_mae_ms": 1.0}
    floor = {"index": 0, "x": 0, "y": 0, "episode_s": 0, "exit_episode_s": 1,
             "chart_s": 0, "bpm": 120, "entry_angle_rad": 0, "exit_angle_rad": math.pi,
             "orbit_start_angle_rad": 0, "orbit_end_angle_rad": math.pi,
             "angle_moved_rad": math.pi, "rotation_sign": 1, "rotation_start_s": 0,
             "rotation_end_s": 1, "motion_segments": [{"start_s": 0, "end_s": 1,
             "from_rad": 0, "to_rad": math.pi, "bpm": 120}], "midspin": False,
             "previous_midspin": False, "is_ccw": False, "num_planets": 2,
             "fixed_identity": 0, "pause_chart_s": 0, "markers": []}
    return {"chart": "fixture", "timebase": {"episode_duration_s": 1,
            "segment_start_chart_s": 0, "pitch_ratio": 1, "offset_ms": 0,
            "separate_countdown_chart_s": 0}, "floors": [floor],
            "events": [{"event_index": 0, "episode_s": .5, "floor_indices": [0],
                        "network": None}],
            "policies": {"network": {"trace": trace, "keydowns": [], "saved_metrics": metrics}},
            "key_overlay": {"mapping": [{"label": x, "contact": y}
                          for x, y in v03.KEY_DISPLAY_MAP]}}


def test_html_is_self_contained_and_javascript_parses(tmp_path):
    html = v03.render_html(_html_payload())
    assert "16 Contact KeyDown Counter" in html and "arcMoved" in html
    assert "Math.sin(Math.PI" not in html
    assert "https://" not in html and "<script src=" not in html
    match = re.search(r"<script>(.*)</script>", html, re.S)
    assert match
    if not shutil.which("node"):
        pytest.skip("node is unavailable")
    script = tmp_path / "viewer.js"
    script.write_text(match.group(1), encoding="utf-8")
    completed = subprocess.run(["node", "--check", str(script)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
