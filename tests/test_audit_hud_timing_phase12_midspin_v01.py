from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_hud_timing_phase12_midspin_v01 as phase12


def test_phase12_target_set_is_frozen_to_thirteen_events():
    assert sum(len(indices) for indices in phase12.TARGETS.values()) == 13
    assert set(phase12.TARGETS) == set(phase12.TARGET_HASHES)
    assert phase12.MOTOR_LEAD_S == .044
    assert phase12.TRACE_WINDOW_S == .080


def test_baseline_and_candidate_differ_only_by_explicit_midspin_option():
    baseline = phase12._source(recovery=False)
    candidate = phase12._source(recovery=True)
    assert baseline.lifecycle_version == candidate.lifecycle_version == "v1.5"
    assert baseline.midspin_recovery is False
    assert candidate.midspin_recovery is True
    assert baseline.composite._reconcile_countdown_v14_identity is True
    assert candidate.composite._reconcile_countdown_v14_identity is True


def test_phase12_output_never_overwrites(tmp_path: Path):
    output = tmp_path / "result.json"
    phase12._write_new(output, {"format": "first"})
    with pytest.raises(FileExistsError):
        phase12._write_new(output, {"format": "second"})
    assert '"first"' in output.read_text(encoding="utf-8")
