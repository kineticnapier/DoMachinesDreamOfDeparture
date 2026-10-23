from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_dmdogo_v06_pattern_first as v06

from dmdod.adofai.chart import parse_adofai_text
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.planning.pattern_motor_adapter_v06 import PatternMotorMapping


def _segment(count=20, bpm=1920):
    return build_playable_segment(compile_adofai(parse_adofai_text(json.dumps({
        "angleData": [0] * count,
        "settings": {"bpm": bpm, "pitch": 100, "countdownTicks": 0},
        "actions": [],
    }))))


def test_default_mapping_uses_distinct_uncoupled_ordinary_axes():
    mapping = PatternMotorMapping.default_eight()
    axes = tuple(mapping.axis_for(f"key_{index}") for index in range(8))
    assert len(set(axes)) == 8
    assert all(axis.endswith(("_1", "_2", "_3", "_4")) for axis in axes)
    assert mapping.network_used is False


def test_candidate_order_and_bounded_fallback():
    mapping = PatternMotorMapping.default_eight()
    candidates = mapping.candidate_axes(("key_0",), allow_local_replan=True)
    assert candidates[0] == (mapping.axis_for("key_0"),)
    assert len(candidates) == 8
    assert len(mapping.candidate_axes(("key_0", "key_1"), True)) <= 32


def test_unavailable_pattern_is_deferred_until_44ms_deadline():
    mapping = PatternMotorMapping.default_eight()

    def eligibility(axes):
        return axes != ("left_1",), "reserved" if axes == ("left_1",) else "eligible"

    early = mapping.choose(("key_0",), remaining_s=.090,
                           legacy_lead_s=.044, eligible=eligibility)
    assert early.axes is None
    assert early.disposition == "deferred"
    assert early.attempted == 1
    late = mapping.choose(("key_0",), remaining_s=.044,
                          legacy_lead_s=.044, eligible=eligibility)
    assert late.axes == ("right_1",)
    assert late.disposition == "local_replan"
    assert late.desired_reason == "reserved"


def test_no_live_candidate_does_not_fabricate_eligibility():
    mapping = PatternMotorMapping.default_eight()
    decision = mapping.choose(
        ("key_0",), remaining_s=.010, legacy_lead_s=.044,
        eligible=lambda axes: (False, "pressed"),
    )
    assert decision.axes is None
    assert decision.disposition == "no_live_candidate"


def test_invalid_physical_mapping_and_duplicate_chord_fail_closed():
    with pytest.raises(ValueError):
        PatternMotorMapping(("key_0",), ("left_5",))
    with pytest.raises(ValueError):
        PatternMotorMapping(("key_0",), ("right_0",))
    with pytest.raises(ValueError):
        PatternMotorMapping.default_eight().desired_axes(("key_0", "key_0"))
    with pytest.raises(ValueError):
        PatternMotorMapping(("a",), ("left_1",), max_candidates=0)


def test_pattern_physical_runner_uses_no_teacher_candidate_rollout(monkeypatch):
    def leak(*args, **kwargs):
        raise AssertionError("Teacher candidate rollout leaked")
    monkeypatch.setattr(v06.teacher, "rank_first_choices", leak)
    result = v06.run_pattern_arm(
        _segment(), minimum_reuse_s=.080,
        stop_after_s=1.0, wall_timeout_s=30.0)
    assert result["abstract_schedule"]["feasible"]
    assert result["network_used"] is False
    assert result["teacher_shadow_rollout_used"] is False
    assert result["teacher_candidate_scores_used"] is False
    assert result["stats"]["misses"] == 0
    assert result["stats"]["too_early"] == 0
    assert not result["stats"]["overloaded"]
    assert all(row["physical_eligible"] for row in result["decisions"])
    assert all(row["abstract_keys"] for row in result["decisions"])
    assert result["physical"]["local_replans"] == sum(
        row["local_replan"] for row in result["decisions"])


def test_runner_refuses_to_overwrite_existing_output(tmp_path):
    directory = tmp_path / "already-there"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        v06.run(tmp_path / "not-even-a-chart.adofai", directory)
