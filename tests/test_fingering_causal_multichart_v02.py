from __future__ import annotations

from pathlib import Path

import pytest

from scripts import audit_fingering_causal_multichart_v02 as audit


def _chart(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def test_automatic_selection_prefers_test_and_is_precommitted(tmp_path):
    root = tmp_path / "data"
    _chart(root / "Train", "original.adofai", "train")
    _chart(root / "Validation", "validation.adofai", "val")
    for i in range(5):
        _chart(root / "Test", f"{i:02}.adofai", f"unique {i}")
    chosen, manifest = audit.select_charts(root, [], max_charts=3)
    assert [x["name"] for x in chosen] == [
        "00.adofai", "02.adofai", "04.adofai"
    ]
    assert manifest["selection_basis"] == "automatic:test"
    assert manifest["population_n"] == 5
    assert all(x["split"] == "test" for x in chosen)
    assert all(len(x["sha256"]) == 64 for x in chosen)
    # The pool is recorded before reading test performance.
    assert len(manifest["full_ordered_candidate_manifest"]) == 5


def test_never_auto_falls_back_to_train(tmp_path):
    root = tmp_path / "dataset"
    _chart(root / "Train", "only.adofai", "only training")
    with pytest.raises(FileNotFoundError, match="no recognized non-Train"):
        audit.select_charts(root, [], max_charts=3)


def test_explicit_nontrain_can_be_outside_labeled_split(tmp_path):
    path = _chart(tmp_path / "external", "chart.adofai", "separate")
    selected, manifest = audit.select_charts(tmp_path / "not-a-root", [path], 2)
    assert len(selected) == 1
    assert manifest["selection_basis"] == "explicit"
    assert selected[0]["split"] == "unspecified"
    assert selected[0]["provenance"] == "externally_supplied_unverified_split"


def test_explicit_train_chart_is_refused(tmp_path):
    path = _chart(tmp_path / "Train", "train.adofai", "data")
    with pytest.raises(ValueError, match="Train chart forbidden"):
        audit.select_charts(tmp_path, [path], 1)


def test_selected_duplicate_chart_content_is_refused(tmp_path):
    a = _chart(tmp_path / "Test", "a.adofai", "same bytes")
    b = _chart(tmp_path / "Test", "b.adofai", "same bytes")
    with pytest.raises(ValueError, match="duplicate chart content"):
        audit.select_charts(tmp_path, [a, b], 2)


def test_prior_bool_identity_is_refused(tmp_path, monkeypatch):
    chart = _chart(tmp_path / "Test", "copy.adofai", "arbitrary")
    monkeypatch.setattr(
        audit.viewer, "file_sha256", lambda _: audit.PREVIOUS_BOOL_SHA256
    )
    with pytest.raises(ValueError, match="previously evaluated bool"):
        audit.select_charts(tmp_path, [chart], 1)


def test_chart_sampling_is_deterministic(tmp_path):
    ordered = [Path(f"{i}.adofai") for i in range(6)]
    assert audit.equally_spaced(ordered, 3) == [
        ordered[1], ordered[3], ordered[5]
    ]
    assert audit.equally_spaced(ordered, 10) == ordered
    with pytest.raises(ValueError):
        audit.equally_spaced(ordered, 0)


def test_metric_pair_detects_accuracy_loss_and_changed_fingering():
    old = {
        "stats": {
            "hits": 20, "targets": 20, "misses": 0, "too_early": 0,
            "x_accuracy_percent": 90.0, "timing_mae_ms": 10.0,
            "physical_keydowns": 20, "overloaded": False
        },
        "decisions": [{"event_index": 1, "chosen": "left_1"}],
        "termination_reason": "chart_complete",
        "elapsed_wall_s": 1.0, "interventions": {}
    }
    new = {
        **old,
        "stats": {**old["stats"],
                  "hits": 19, "misses": 1, "x_accuracy_percent": 85.0,
                  "too_early": 2, "physical_keydowns": 21},
        "decisions": [{"event_index": 1, "chosen": "right_1"}],
        "interventions": {"actual_early_axis_steps": 2},
    }
    report = audit.compare_pair(old, new)
    assert report["delta_accuracy_pp"] == -5.0
    assert report["delta_hits"] == -1
    assert report["delta_misses"] == 1
    assert report["delta_too_early"] == 2
    assert report["delta_physical_keydowns"] == 1
    assert report["changed_finger_choices"] == 1
    assert report["future_saved_allocation_consulted"] is False


def test_error_cases_are_not_counted_as_completed():
    results = [
        {"policies": {"free": {"error": "UnsupportedChart"}}},
        {"chart_load_error": "BadChart", "policies": {}},
    ]
    result = audit.aggregate(results)
    assert result["free"]["completed_charts"] == 0
    assert result["human_proxy"]["completed_charts"] == 0


def test_explicit_paths_above_limit_are_rejected_not_silently_ignored(tmp_path):
    a = _chart(tmp_path / "Test", "a.adofai", "a")
    b = _chart(tmp_path / "Test", "b.adofai", "b")
    with pytest.raises(ValueError, match="increase --max-charts"):
        audit.select_charts(tmp_path, [a, b], 1)


def test_frozen_checkpoint_identity_checked_before_model_load(
        tmp_path, monkeypatch):
    ck = tmp_path / "frozen.pt"
    ck.write_bytes(b"fake checkpoint; must not load")
    selected = []
    called = []
    monkeypatch.setattr(
        audit.causal.runner, "load_model",
        lambda _path: called.append(True)
    )
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        audit.evaluate(ck, selected, 1.0, "0" * 64)
    assert called == []
