from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from scripts.audit_hud_timing_phase10_source_identity_v01 import (
    _forecast_differences,
)


def _row(step, tokens):
    return {
        "step": step,
        "raw_pending": [{
            "id": local_id,
            "due_lower_s": lower,
            "due_upper_s": upper,
            "uncertainty_s": (upper - lower) * .5,
        } for local_id, lower, upper in tokens],
    }


def test_difference_audit_isolates_only_reconciled_countdown_alias():
    before = [_row(0, [
        ("countdown:0", 1.0, 1.0),
        ("v14:1", 1.2, 1.2),
    ])]
    after = [_row(0, [
        ("countdown:0", 1.000001, 1.000001),
        ("v14:1", 1.201, 1.201),
    ])]
    result = _forecast_differences(before, after)
    assert result["reconciled_alias_due"] == [
        {"step": 0, "id": "countdown:0"}
    ]
    assert result["outside_reconciled_alias_due"] == [
        {"step": 0, "id": "v14:1"}
    ]


def test_cli_help_starts_from_repository_root():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/audit_hud_timing_phase10_source_identity_v01.py",
            "--help",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
