from __future__ import annotations

import pytest

from scripts.audit_phase16_event89_causal_v01 import attribute_keydowns


def _arm(*, ambiguous=False):
    reservation = {
        "local_id": 38,
        "due_local_s": 3.2647,
        "axes": ["right_2"],
        "phase": "keydown_observed",
        "down_axes": ["right_2"],
        "up_axes": [],
        "retired_reason": None,
    }
    reservations = [reservation, dict(reservation, local_id=99)] if ambiguous else [reservation]
    return {
        "decisions": [{
            "local_id": 38,
            "due_local_s": 3.2647,
            "choice": "right_2",
            "reason": "assigned",
        }],
        "target_outcomes_posthoc": [
            {"ordinal": 37, "due_s": 3.2206},
            {"ordinal": 38, "due_s": 3.2647},
        ],
        "diagnostic_window_posthoc": {
            "start_s": 3.0,
            "end_s": 3.5,
            "judgement_trace": [{
                "time_s": 3.261,
                "key": "right_2",
                "target_index": 37,
                "margin": "very_late",
            }],
            "frames": [{
                "logical_edges": [{
                    "time_s": 3.261,
                    "contact": "right_2",
                    "edge": "down",
                }],
                "reservations_after_physics": reservations,
            }],
        },
    }


def test_posthoc_audit_separates_intended_id_from_fifo_judged_target():
    rows = attribute_keydowns(_arm())
    assert rows == [{
        "time_s": 3.261,
        "contact": "right_2",
        "local_id": 38,
        "intended_ordinal_posthoc": 38,
        "judged_ordinal_posthoc": 37,
        "ordinal_delta": 1,
        "due_local_s": 3.2647,
        "judgement": "very_late",
    }]


def test_posthoc_audit_fails_closed_on_ambiguous_reservation_attribution():
    with pytest.raises(RuntimeError, match="exactly one"):
        attribute_keydowns(_arm(ambiguous=True))
