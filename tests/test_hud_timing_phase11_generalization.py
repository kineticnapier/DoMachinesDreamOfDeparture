from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_hud_timing_phase11_generalization_v01 as phase11


def _populate_train(tmp_path: Path, *, count: int = 18) -> Path:
    train = tmp_path / "Train"
    train.mkdir()
    for i in range(count):
        (train / f"Train-{i:03d}.adofai").write_text(str(i), encoding="utf-8")
    for name in phase11.REFERENCE:
        (train / name).write_text("reference", encoding="utf-8")
    return train


def test_selection_is_deterministic_and_does_not_include_references(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(phase11, "REFERENCE", {
        "Known.adofai": ("irrelevant", 12.0),
    })
    train = tmp_path / "Train"
    train.mkdir()
    (train / "Known.adofai").write_text("known", encoding="utf-8")
    for i in range(20):
        (train / f"{i:02}.adofai").write_text(str(i), encoding="utf-8")
    names = phase11.candidate_names(train)
    a = phase11.selected_names(names)
    b = phase11.selected_names(list(reversed(names)))
    assert a == b
    assert len(a) == 16
    assert "Known.adofai" not in a
    assert a != sorted(names)[:16]  # selection order is SHA256-based


def test_freeze_and_verify_checks_both_manifest_and_bytes(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(phase11, "REFERENCE", {
        "Known.adofai": (phase11.sha256(b"reference").hexdigest(), 5.0),
    })
    train = _populate_train(tmp_path)
    # Restore the one reference expected by this synthetic test.
    (train / "Known.adofai").write_bytes(b"reference")
    manifest = phase11.freeze_selection(train)
    paths = phase11.verify_selection(manifest, train)
    assert len(paths) == 16
    assert len({p.name for p in paths}) == 16

    edited = json.loads(json.dumps(manifest))
    edited["charts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash changed"):
        phase11.verify_selection(edited, train)
    with pytest.raises(ValueError, match="pool changed"):
        (train / "new.adofai").write_text("extra", encoding="utf-8")
        phase11.verify_selection(manifest, train)


def test_insufficient_unseen_pool_fails_before_writing(tmp_path: Path):
    train = tmp_path / "Train"
    train.mkdir()
    for i in range(15):
        (train / f"{i:02}.adofai").write_bytes(b"{}")
    with pytest.raises(ValueError, match="at least 16"):
        phase11.freeze_selection(train)
    assert list(tmp_path.rglob("*.json")) == []


def test_manifest_output_never_overwrites_existing_file(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        phase11._write_new(path, {"format": "test"})
    assert path.read_text(encoding="utf-8") == "original"


def test_three_arms_are_fixed_and_identity_is_on_only_for_c():
    a, b, c = (phase11._source(arm) for arm in phase11.ARMS)
    assert a.lifecycle_version == "v1.4"
    assert b.lifecycle_version == c.lifecycle_version == "v1.5"
    assert a.composite._reconcile_countdown_v14_identity is False
    assert b.composite._reconcile_countdown_v14_identity is False
    assert c.composite._reconcile_countdown_v14_identity is True
    with pytest.raises(ValueError, match="unknown arm"):
        phase11._source("tuned-arm")


def test_aggregate_reports_zero_event_case_without_dividing_by_zero():
    row = {
        "arms": {
            "C": {
                "events": [],
                "metrics": {"coverage": 0.0, "false_positive_ids": 0},
            }
        },
        "lifecycle_counts": {
            "C": {"new": 0, "reused": 0, "canceled": 0, "elapsed": 0}
        },
    }
    result = phase11.aggregate([row], "C")
    assert result["events"] == 0
    assert result["clean_coverage"] is None
    assert result["timing_mae_ms_conditional_on_clean"] is None
