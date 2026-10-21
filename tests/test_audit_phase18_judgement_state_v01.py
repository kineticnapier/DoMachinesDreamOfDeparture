from __future__ import annotations

from scripts.audit_phase18_judgement_state_v01 import extract_causal_keydowns


def test_causal_keydown_extraction_uses_reservation_not_scorer_identity():
    frames = [{
        "logical_edges": [
            {"time_s": 3.261, "contact": "right_2", "edge": "down"},
        ],
        "reservations_after_physics": [{
            "local_id": 38,
            "down_axes": ["right_2"],
        }],
    }]

    rows = extract_causal_keydowns(frames, 0.010)

    assert rows == [{
        "input_id": "3.261000000:right_2",
        "intended_local_id": "38",
        "occurred_at_s": 3.261,
        "observed_at_s": 3.27,
        "contact": "right_2",
    }]
    assert "target_index" not in rows[0]
