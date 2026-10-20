from __future__ import annotations

from pathlib import Path
import pytest

from scripts import run_hud_timing_phase16_motor_ab_v01 as p16


def _arm(*, hits=1, too_early=0, overloaded=False, collisions=0):
    return {
        "stats": {
            "targets": 1, "hits": hits, "misses": 1 - hits,
            "too_early": too_early, "overloaded": overloaded,
            "overload_counter": 0.0,
            "physical_keydowns": hits, "timing_mae_ms": 0.0,
            "x_accuracy_percent": float(hits * 100),
        },
        "scheduling": {
            "insufficient_lead": 0, "unassigned": 0,
            "reservation_collisions": collisions,
            "inflight_cancellations": 0,
        },
        "target_outcomes_posthoc": [
            {"ordinal": 0, "due_s": 4.0, "hit": bool(hits),
             "miss": not hits, "keydown_s": 4.0 if hits else None}
        ],
        "physical_keydowns": [] if not hits else [
            {"time_s": 4.0, "axis": "left_1", "edge": "down"}
        ],
        "logical_keydowns": [], "decisions": [],
        "recovery_flags": {"midspin": False, "twirl": False,
                           "checkpoint": False},
        "source_conflict_count": 0,
        "timing_status_counts": {"eligible": 1},
        "frame_count": 500, "run_wall_s": .1,
        "frames": [],
        "lifecycle_contract_violations": [],
        "reservation_contract_violations": [],
    }


def test_regression_checks_detect_lost_physical_hit():
    baseline = _arm(hits=1)
    candidate = _arm(hits=0)
    comparison = p16._difference(baseline, candidate)
    assert comparison["valid"]
    assert comparison["lost_hit_ordinals_posthoc"] == [0]
    assert not comparison["checks"]["no_hit_regression"]
    assert comparison["delta"]["hits"] == -1


def test_new_physical_hit_is_distinct_from_timing_coverage():
    baseline, candidate = _arm(hits=0), _arm(hits=1)
    comparison = p16._difference(baseline, candidate)
    assert comparison["newly_hit_ordinals_posthoc"] == [0]
    assert comparison["delta"]["hits"] == 1
    assert all(comparison["checks"].values())


def test_too_early_and_reservation_collision_are_separate_failures():
    before = _arm()
    after = _arm(too_early=1, collisions=1)
    checks = p16._difference(before, after)["checks"]
    assert not checks["too_early_not_increased"]
    assert not checks["reservation_collisions_not_increased"]
    assert checks["no_hit_regression"]


def test_no_mixed_target_population_allowed():
    before, after = _arm(), _arm()
    after["stats"]["targets"] = 2
    assert not p16._difference(before, after)["valid"]


def test_frozen_bool_sha_mismatch_is_rejected(tmp_path):
    (tmp_path / p16.BOOL_CHART).write_text("not the frozen chart", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        p16._fixed_paths(tmp_path, {}, chart_name=None, all_unseen=False)


def test_unlisted_chart_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(p16, "verify_selection", lambda _m, _d: (
        tmp_path / "one.adofai", tmp_path / "two.adofai",
    ))
    with pytest.raises(ValueError, match="must be an entry"):
        p16._fixed_paths(tmp_path, {}, chart_name="third.adofai", all_unseen=False)


def test_full_selection_keeps_manifest_order(monkeypatch, tmp_path):
    expected = (tmp_path / "one.adofai", tmp_path / "two.adofai")
    monkeypatch.setattr(p16, "verify_selection", lambda _m, _d: expected)
    observed = p16._fixed_paths(tmp_path, {}, chart_name=None, all_unseen=True)
    assert observed == tuple((path, p16.UNSEEN_END_S) for path in expected)


def test_mutually_exclusive_selection_rejected(tmp_path):
    with pytest.raises(ValueError, match="either one chart"):
        p16._fixed_paths(tmp_path, {}, chart_name="a", all_unseen=True)


def test_output_is_exclusive_create(tmp_path):
    out = tmp_path / "report.json"
    p16._write_new(out, {"a": 1})
    with pytest.raises(FileExistsError):
        p16._write_new(out, {"a": 2})


def test_policy_flags_forwarded_only_to_candidate(monkeypatch, tmp_path):
    chart = tmp_path / p16.BOOL_CHART
    chart.write_text("frozen chart stub", encoding="utf-8")
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"frozen model stub")
    original_hash = p16._sha256
    monkeypatch.setattr(
        p16, "_sha256",
        lambda path: (
            p16.FIXED_FINGERING_SHA256 if path == checkpoint
            else p16.BOOL_SHA256 if path == chart else original_hash(path)
        ),
    )
    monkeypatch.setattr(p16.phase3.frozen, "load_model", lambda _: object())
    monkeypatch.setattr(p16, "load_adofai", lambda _: object())
    monkeypatch.setattr(p16, "compile_adofai", lambda _: object())
    monkeypatch.setattr(
        p16, "build_playable_segment",
        lambda chart_data, end_s: type("Segment", (), {"targets": (object(),)})(),
    )
    calls = []

    def simulate(segment, model, arm, **kwargs):
        calls.append(kwargs)
        result = _arm()
        result["recovery_flags"] = {
            "midspin": kwargs.get("midspin_recovery", False),
            "twirl": kwargs.get("twirl_recovery", False),
            "checkpoint": kwargs.get("checkpoint_recovery", False),
        }
        if kwargs.get("capture_judgement_trace", False):
            result["judgement_trace_posthoc"] = []
        # Match real phase3.simulate: contract summaries are NOT returned.
        # The Phase16 caller must compact and derive them before comparison.
        del result["lifecycle_contract_violations"]
        del result["reservation_contract_violations"]
        return result

    monkeypatch.setattr(p16.phase3, "simulate", simulate)
    report = p16.run(tmp_path, checkpoint, {}, per_arm_timeout_s=10.0)
    assert report["verdict"] == "MOTOR_AB_NONREGRESSION_PASS"
    assert len(calls) == 2
    assert calls[0].get("checkpoint_recovery") is None
    assert calls[1]["midspin_recovery"] is True
    assert calls[1]["twirl_recovery"] is True
    assert calls[1]["checkpoint_recovery"] is True
    assert report["cases"][0]["arms"][p16.BASELINE]["stats"]["hits"] == 1
    for arm in (p16.BASELINE, p16.CANDIDATE):
        compact = report["cases"][0]["arms"][arm]
        assert compact["lifecycle_contract_violations"] == []
        assert compact["reservation_contract_violations"] == []
    assert report["cases"][0]["difference"]["checks"]["lifecycle_contract_valid"]
    assert report["cases"][0]["difference"]["checks"]["reservation_contract_valid"]

    # Diagnostic capture is strictly opt-in and must reach BOTH physical arms.
    diagnosed = p16.run(
        tmp_path, checkpoint, {}, per_arm_timeout_s=10.0,
        diagnostic_center_s=5.25, diagnostic_radius_s=.25,
    )
    assert len(calls) == 4
    assert calls[0]["capture_judgement_trace"] is False
    assert calls[1]["capture_judgement_trace"] is False
    assert calls[2]["capture_judgement_trace"] is True
    assert calls[3]["capture_judgement_trace"] is True
    assert diagnosed["diagnostic_window_s"] == (5.0, 5.5)
    for arm in (p16.BASELINE, p16.CANDIDATE):
        assert "diagnostic_window_posthoc" in diagnosed["cases"][0]["arms"][arm]


def test_non_v3_arm_rejects_recovery_flags():
    with pytest.raises(ValueError, match="require hud_v3_composite"):
        p16.phase3.simulate(
            None, None, "hud_v14", wall_timeout_s=1.0,
            checkpoint_recovery=True,
        )


def test_diagnostic_window_is_posthoc_bounded_and_opt_in(monkeypatch):
    monkeypatch.setattr(p16, "_lifecycle_contract_violations", lambda _: [])
    monkeypatch.setattr(p16, "_reservation_violations", lambda _: [])
    arm = _arm()
    arm["frames"] = [
        {"local_clock_s": 4.90, "step": 490},
        {"local_clock_s": 5.25, "step": 525},
        {"local_clock_s": 5.60, "step": 560},
    ]
    arm["judgement_trace_posthoc"] = [
        {"time_s": 4.90, "key": "left_1",
         "target_index": 87, "margin": "perfect"},
        {"time_s": 5.253, "key": "right_3",
         "target_index": 88, "margin": "very_late"},
        {"time_s": 5.60, "key": "right_1",
         "target_index": 90, "margin": "perfect"},
    ]
    arm["physical_keydowns"] = [
        {"time_s": 5.253, "axis": "right_3", "edge": "down"},
    ]
    arm["logical_keydowns"] = [
        {"time_s": 5.253, "contact": "right_3", "edge": "down"},
    ]
    arm["decisions"] = [
        {"local_id": 89, "due_local_s": 5.25, "choice": "right_3"},
    ]
    arm["target_outcomes_posthoc"] = [
        {"ordinal": 88, "due_s": 5.205882, "hit": True,
         "keydown_s": 5.253, "contact": "right_3"},
        {"ordinal": 89, "due_s": 5.25, "hit": False,
         "keydown_s": None, "contact": None},
    ]
    ordinary = p16._compact(arm)
    assert "diagnostic_window_posthoc" not in ordinary
    detail = p16._compact(arm, diagnostic_window_s=(5.0, 5.5))
    rows = detail["diagnostic_window_posthoc"]
    assert [row["step"] for row in rows["frames"]] == [525]
    assert rows["judgement_trace"] == [
        {"time_s": 5.253, "key": "right_3",
         "target_index": 88, "margin": "very_late"}
    ]
    assert rows["decisions"][0]["local_id"] == 89
    assert rows["target_outcomes"][1]["ordinal"] == 89
    assert rows["target_outcomes"][1]["hit"] is False
    assert len(rows["physical_keydowns"]) == 1
    assert len(rows["logical_keydowns"]) == 1


def test_diagnostic_window_missing_real_scorer_trace_fails_closed():
    arm = _arm()
    with pytest.raises(KeyError, match="judgement_trace_posthoc"):
        p16._compact(arm, diagnostic_window_s=(5.0, 5.5))


def test_diagnostic_radius_must_be_valid_even_without_replay(tmp_path, monkeypatch):
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"frozen model stub")
    monkeypatch.setattr(p16, "_sha256", lambda _: p16.FIXED_FINGERING_SHA256)
    with pytest.raises(ValueError, match="diagnostic radius"):
        p16.run(tmp_path, checkpoint, {},
                diagnostic_center_s=5.25, diagnostic_radius_s=0.0)

