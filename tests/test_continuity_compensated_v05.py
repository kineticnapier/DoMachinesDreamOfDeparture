from dmdod.evaluation.continuity_compensated_v05 import analyze_continuity_compensation


def _row(index, due, decision, chosen, free, labels, eligible, error=1.0):
    return {"event_index": index, "due_ms": due, "decision_ms": decision,
            "chosen": chosen, "free_chosen": free, "axes": [chosen],
            "candidate_labels": labels, "eligible": eligible,
            "actual_error_ms": error}


def test_analysis_separates_context_change_from_live_unavailability():
    continuity = [
        _row(0, 100, 60, "left_1", "left_1", [], []),
        _row(1, 130, 90, "left_2", "right_1", [], []),
    ]
    compensated = [
        _row(0, 100, 10, "right_1", "right_1",
             ["left_1", "right_1"], [False, True]),
        _row(1, 130, 40, "left_3", "left_3",
             ["left_2", "left_3"], [True, True]),
    ]
    result = analyze_continuity_compensation(
        continuity, compensated, bpm_by_event={0: 480, 1: 1920})
    assert result["choice_changed"] == 2
    assert result["free_argmin_changed"] == 2
    assert result["decision_advance_ms"]["mean"] == 50
    assert result["fast_events"] == 1
    assert result["v02_preferred_live_eligibility"] == {
        "checked_changed_events": 2,
        "unavailable": 1,
        "available_but_not_selected": 1,
    }
    assert result["by_bpm"]["1920"]["events"] == 1
