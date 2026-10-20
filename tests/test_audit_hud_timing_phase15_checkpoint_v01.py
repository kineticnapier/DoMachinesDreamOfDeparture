from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_hud_timing_phase15_checkpoint_v01 as phase15


def test_sources_are_the_same_except_checkpoint_opt_in():
    baseline = phase15._source(False)
    candidate = phase15._source(True)
    assert baseline.lifecycle_version == candidate.lifecycle_version == "v1.5"
    assert baseline.midspin_recovery and candidate.midspin_recovery
    assert baseline.twirl_recovery and candidate.twirl_recovery
    assert baseline.checkpoint_recovery is False
    assert candidate.checkpoint_recovery is True


def test_fixed_hero_target_is_recovered_without_unsafe_ids():
    manifest = json.loads(Path(
        "reports/hud-timing-phase11-selection-20261019-r1.json"
    ).read_text(encoding="utf-8"))
    result = phase15.run(phase15.TRAIN_DIR, manifest, target_only=True)
    assert result["verdict"] == "CHECKPOINT_TARGET_PROBE_PASS"
    assert result["target_54"]["baseline"]["category"] != 1
    assert result["target_54"]["candidate"]["category"] == 1
    assert result["target_54"]["motor_ready_44ms"]
    assert result["checks"]["false_positive_ids_zero"]
    assert result["checks"]["existing_clean_regression_zero"]


def test_report_writer_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "audit.json"
    phase15._write_new(path, {"v": 1})
    try:
        phase15._write_new(path, {"v": 2})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing output may not be overwritten")
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}
