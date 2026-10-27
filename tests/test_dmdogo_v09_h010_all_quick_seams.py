from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_dmdogo_v09_h010_all_quick_seams as h010


def _path(*, delta=-.2, protected=True, new_quick=None, rapid_new=None):
    if new_quick is None:
        new_quick = []
    if rapid_new is None:
        rapid_new = []
    return {
        "delta_cost_vs_frozen_v08": delta,
        "is_from_protected_lineage": protected,
        "quick": {
            "added": new_quick,
            "removed": [{"event_pair": [4, 5]}],
            "added_count": len(new_quick),
            "removed_count": 1,
        },
        "rapid": {
            "added": rapid_new, "removed": [],
            "added_count": len(rapid_new), "removed_count": 0,
        },
        "changed_event_assignment_count": 6,
        "minimum_axis_reuse_ms": 140.0,
        "axis_reuse_under_80ms": 0,
        "hand_axis_uses": {"left": 2, "right": 5},
        "score_components": {"hand_exposure": 1.0},
    }


def _row(*, cohort="previously_nonimproved_15", delta=.01):
    original = _path(delta=delta, protected=False)
    protected = _path(delta=-.1, protected=True)
    return {
        "flag_pair": [4, 5],
        "h06_cohort": cohort,
        "quota8_result": {
            "protected_descendants_reached_terminal": True,
            "slot_replacement_count": 2,
            "total_displaced_nonprotected_slots": 5,
            "windows_with_tagged_count_below_quota": 1,
        },
        "winner_effects": h010._measure(original, (4, 5)),
        "best_protected_effects": h010._measure(protected, (4, 5)),
    }


def test_quick_target_removal_is_location_based_and_no_false_claims():
    assert h010._target_missing({
        "removed": [{"event_pair": [4, 5]}],
        "added": [{"event_pair": [8, 9]}],
    }, (4, 5)) is True
    assert h010._target_missing({
        "removed": [], "added": [{"event_pair": [8, 9]}],
    }, (4, 5)) is False
    assert h010._target_missing({
        "removed": [{"event_pair": [4, 5]}],
        "added": [{"event_pair": [4, 5]}],
    }, (4, 5)) is False


def test_winner_and_best_protected_are_not_conflated():
    row = _row(delta=.02)
    out = h010.summarize([row])
    assert out["winner_lower_cost"] == 0
    assert out["winner_higher_cost"] == 1
    assert out["winner_is_protected"] == 0
    assert out["best_protected_lower_cost"] == 1
    assert out["terminal_protected_survivors"] == 1
    assert out["rescue_windows"] == 2
    assert out["displaced_untagged_slots"] == 5
    assert out["winner_target_quick_removed"] == 1
    assert out["best_protected_target_quick_removed"] == 1


def test_quick_relocation_counted_as_added_and_removed_not_single_identity():
    raw = _path(delta=-.01, new_quick=[{"event_pair": [9, 10]}],
                rapid_new=[{"event_range": [12, 17]}])
    measured = h010._measure(raw, (4, 5))
    assert measured["new_quick_flag_count"] == 1
    assert measured["removed_quick_flag_count"] == 1
    assert measured["new_rapid_flag_count"] == 1
    assert measured["target_quick_flag_removed"] is True


def test_nonimproved_cohort_may_improve_or_regress_without_unsupported_assumptions():
    entries = [_row(delta=-.2), _row(delta=.4), _row(delta=0)]
    got = h010.summarize(entries)
    assert got["target_count"] == 3
    assert got["winner_lower_cost"] == 1
    assert got["winner_higher_cost"] == 1
    assert got["winner_roughly_equal_cost"] == 1
    assert got["winner_cost_delta_distribution"]["median"] == 0
    assert got["winner_changed_events_distribution"]["min"] == 6


def test_no_tagged_terminal_does_not_fabricate_protected_improvements():
    row = _row()
    row["quota8_result"]["protected_descendants_reached_terminal"] = False
    row["best_protected_effects"] = None
    got = h010.summarize([row])
    assert got["terminal_protected_survivors"] == 0
    assert got.get("best_protected_lower_cost", 0) == 0
    assert got["best_protected_cost_delta_distribution"]["count"] == 0


def test_reference_reproduced_or_fails_closed():
    entry = {"improved_final_cost": True, "cost": {"final_delta": -.123}}
    result = {"final_cost": 1., "flags": {"added": []}}
    reference = {"variants": {"8": {"final_cost": 1., "flags": {"added": []}}},
                 "h06_forced_suffix_final_delta": -.123}
    h010.verify_improving_reference(entry, result, reference)
    reference["variants"]["8"]["final_cost"] = 2.
    with pytest.raises(ValueError, match="does not reproduce"):
        h010.verify_improving_reference(entry, result, reference)
    reference["variants"]["8"]["final_cost"] = 1.
    reference["h06_forced_suffix_final_delta"] = -.10
    with pytest.raises(ValueError, match="does not reproduce"):
        h010.verify_improving_reference(entry, result, reference)
    entry["improved_final_cost"] = False
    with pytest.raises(ValueError, match="original seven"):
        h010.verify_improving_reference(entry, result, reference)


def test_duplicate_target_pair_rejected():
    with pytest.raises(ValueError, match="duplicate H0.6"):
        h010._pairs([{"flag_pair": [1, 2]}, {"flag_pair": [1, 2]}], "H0.6")


def test_committed_h06_census_22_and_7_plus_15():
    saved = json.loads((
        Path(__file__).resolve().parents[1]
        / "reports/dmdogo-v09-h06-improved-suffix-r2.json"
    ).read_text(encoding="utf-8"))
    assert saved["aggregate"]["targeted_count"] == 22
    assert saved["aggregate"]["improved_final_cost_count"] == 7
    rows = [r for case in saved["cases"] for r in case["evidence"]]
    assert len(rows) == 22
    assert sum(row["improved_final_cost"] for row in rows) == 7
    assert all(row["nonquick_axes"] for row in rows)
    assert all(row["cost"]["original_v08_final"] for row in rows)


def test_quota_and_frozen_beam_constants():
    assert h010.BEAM_WIDTH == 24
    assert h010.QUOTA == 8
