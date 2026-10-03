from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_n_key_set_actions as audit


def test_set_audit_accepts_different_free_finger_identity() -> None:
    # Teacher happens to choose key 0, but key 1 is equally valid because both
    # are free.  Release identity on key 3 remains fixed.
    target = torch.tensor([[1.0, 0.0, 0.0, -1.0]], dtype=torch.float32)
    predicted = torch.tensor([[0.0, 0.8, 0.0, -0.8]], dtype=torch.float32)

    counts = audit.audit_set_actions(predicted, target)

    assert counts.press_slots == 1
    assert counts.press_ge_070 == 1
    assert counts.release_slots == 1
    assert counts.release_le_m030 == 1
    assert counts.extra_gt_005 == 0
    assert counts.count_exact_070_frames == 1
    assert counts.strict_frames == 1


def test_set_audit_detects_extra_free_push_separately_from_required_press() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0, -1.0]], dtype=torch.float32)
    predicted = torch.tensor([[0.8, 0.6, 0.0, -0.8]], dtype=torch.float32)

    counts = audit.audit_set_actions(predicted, target)

    # Strongest free output satisfies the one required press; the second push
    # is an extra free-key command and must still fail the strict frame audit.
    assert counts.press_ge_070 == 1
    assert counts.extra_gt_005 == 1
    assert counts.extra_gt_025 == 1
    assert counts.extra_gt_005_frames == 1
    assert counts.strict_frames == 0


def test_set_audit_reports_press_count_under_and_over_at_margin() -> None:
    target = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    predicted = torch.tensor(
        [
            [0.8, 0.4, 0.0, 0.0],  # requires two, only one reaches .70
            [0.8, 0.9, 0.0, 0.0],  # requires one, two reach .70
        ],
        dtype=torch.float32,
    )

    counts = audit.audit_set_actions(predicted, target)

    assert counts.frames == 2
    assert counts.count_under_070_frames == 1
    assert counts.count_over_070_frames == 1
    assert counts.count_exact_070_frames == 0


def test_set_audit_counts_add_for_aggregate() -> None:
    first = audit.SetActionAuditCounts(frames=3, press_slots=4, press_ge_070=2)
    second = audit.SetActionAuditCounts(frames=5, press_slots=6, press_ge_070=5)

    merged = first + second

    assert merged.frames == 8
    assert merged.press_slots == 10
    assert merged.press_ge_070 == 7


def test_set_audit_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        audit.audit_set_actions(torch.zeros((2, 8)), torch.zeros((3, 8)))
