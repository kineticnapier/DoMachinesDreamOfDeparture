from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_fingering_network_v011_comparison as compare


def _row(index, chosen, style, error=1.0, outcome="Hit"):
    return {"event_index": index, "chosen": chosen, "style_class": style,
            "actual_error_ms": error, "outcome": outcome}


def test_comparison_reports_differences_worsening_and_dense_specials():
    free = {"decisions": [_row(0, "left_5/tip", "thumb_or_side", 1),
                           _row(1, "right_2", "ordinary_proxy", 2)]}
    human = {"decisions": [_row(0, "left_1", "index_proxy", 3),
                            _row(1, "right_2", "ordinary_proxy", 1)]}
    result = compare.compare_results(free, human, {0: .1, 1: .12})
    assert result["different_events"] == [0]
    assert result["human_proxy_worsened_events"][0]["event_index"] == 0
    assert result["human_proxy_improved_timing_events"] == [1]
    assert result["special_fingering_avoided_events"] == [0]
    assert result["free_style"]["dense_le_35ms_special_events"] == [0]
    assert result["human_proxy_style"]["thumb_or_side_uses"] == 0


def test_fixed_alpha_is_declared_without_search():
    assert compare.FIXED_HUMAN_ALPHA == 0.5


def test_saved_policy_roles_must_be_distinct(tmp_path):
    # Validation occurs before chart/checkpoint access.
    row = {"policy": "free"}
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(__import__("json").dumps(row), encoding="utf-8")
    right.write_text(__import__("json").dumps(row), encoding="utf-8")
    try:
        compare.assemble_saved_policies(Path("missing-chart"), Path("missing-checkpoint"),
                                        tmp_path / "out", left, right)
    except ValueError as error:
        assert "roles" in str(error)
    else:
        raise AssertionError("duplicate saved policy role accepted")
