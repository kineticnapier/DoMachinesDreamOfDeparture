from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import audit_n_key_bc_actions as audit


def test_action_audit_counts_press_release_and_neutral_thresholds() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, -1.0],
            [0.0, -1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    predicted = torch.tensor(
        [
            [0.80, 0.30],
            [0.55, -0.40],
            [0.10, -0.20],
            [0.04, 0.26],
        ],
        dtype=torch.float32,
    )

    report = audit.audit_actions(predicted, target)
    counts = report.overall

    assert counts.press_total == 2
    assert counts.press_ge_025 == 2
    assert counts.press_ge_050 == 2
    assert counts.press_ge_070 == 1
    assert counts.release_total == 2
    assert counts.release_le_m030 == 1
    assert counts.neutral_total == 4
    assert counts.neutral_gt_005 == 3
    assert counts.neutral_gt_025 == 2

    assert len(report.per_key) == 2
    assert report.per_key[0].press_total == 2
    assert report.per_key[1].release_total == 2


def test_action_audit_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="matching shape"):
        audit.audit_actions(torch.zeros((3, 8)), torch.zeros((2, 8)))

    with pytest.raises(ValueError, match="even integer"):
        audit.audit_actions(torch.zeros((3, 3)), torch.zeros((3, 3)))


def test_merge_reports_sums_all_keys_and_rejects_width_mismatch() -> None:
    target = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    first = audit.audit_actions(target.clone(), target)
    second = audit.audit_actions(target.clone(), target)
    merged = audit.merge_reports([first, second])

    assert merged.overall.press_total == 2
    assert merged.overall.press_ge_070 == 2
    assert merged.overall.release_total == 2
    assert merged.overall.release_le_m030 == 2
    assert merged.overall.neutral_total == 4

    bad = audit.ActionAuditReport(
        overall=audit.ActionAuditCounts(),
        per_key=(audit.ActionAuditCounts(),) * 4,
    )
    with pytest.raises(ValueError, match="same key width"):
        audit.merge_reports([first, bad])


def test_format_counts_reports_percentages() -> None:
    counts = audit.ActionAuditCounts(
        press_total=4,
        press_ge_025=3,
        press_ge_050=2,
        press_ge_070=1,
        release_total=2,
        release_le_m030=1,
        neutral_total=200,
        neutral_gt_005=10,
        neutral_gt_025=2,
    )

    text = audit._format_counts("all:", counts)

    assert "recall@.25=75.00%" in text
    assert "@.50=50.00%" in text
    assert "@.70=25.00%" in text
    assert "recall@-.30=50.00%" in text
    assert "false-push>.05=5.000%" in text
    assert ">.25=1.000%" in text
