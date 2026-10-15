from __future__ import annotations

import copy
import json

import pytest

from dmdod.training import prestige1_fingering_audit as audit


def _fake_docs():
    rows, plans = [], []
    for anchor in range(1, 21):
        n = 43 if anchor <= 10 else 42
        events = [
            {
                "target_index": i, "time_s": .5 + i * .15,
                "key": "left_1" if i % 2 == 0 else "right_1",
                "error_ms": 0., "judgement": "perfect",
            }
            for i in range(n)
        ]
        physical = dict(
            hits=n, targets=n, misses=0, too_early=0,
            keydowns=n, unjudged=0, overload=False,
        )
        rows.append(dict(
            anchor=anchor, chart=f"Train {anchor}",
            chart_sha256=f"sha-{anchor}", final_physics=physical,
            final_timing={"mae_ms": 0.}, final_hit_events=events,
        ))
        plans.append(dict(
            anchor=anchor, chart=f"Train {anchor}",
            chart_sha256=f"sha-{anchor}", independently_replayed=True,
            verified_hits=n, verified_mae_ms=0.,
            actions_8k=[
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., -1., 1., 0., 0., 0.],
            ],
        ))
    report = dict(
        format=audit.SOURCE_FORMAT, stage="C-v3.0",
        role="Train20 only", no_training=True,
        student_checkpoint_loaded=False, validation_evaluated=False,
        final_evaluated=False, stage_a_gate_overridden=False,
        original_stage_a_gate={"passed": False, "may_run_stage_c": False},
        gates={"full_train20_safe850": True, "student_training_permitted": False},
        saved_plan_sha256="synthetic-plan-sha",
        saved_plan_file="stage_c_v30_safe_teacher_plans.json",
        config_sha256="synthetic-config", train_manifest=["train"] * 20,
        per_anchor=rows,
        candidate={
            "physics": {
                "hits": 850, "keydowns": 850, "misses": 0,
                "too_early": 0, "overloaded": 0, "unjudged": 0,
            },
            "timing": {"mae_ms": 0.},
        },
    )
    document = dict(
        format=audit.PLAN_FORMAT, stage="C-v3.0",
        role="Train20 only", no_training=True,
        student_checkpoint_loaded=False, validation_evaluated=False,
        final_evaluated=False, stage_a_gate_overridden=False,
        config_sha256="synthetic-config", train_manifest=["train"] * 20,
        finger_order=list(audit.FINGER_ORDER),
        control_dt_s=.010, physics_dt_s=.001, plans=plans,
    )
    return report, document


def test_descriptive_audit_counts_without_inferring_teacher_humanity():
    report, plans = _fake_docs()
    doc = audit.audit_documents(
        report, plans, plan_file_sha256="synthetic-plan-sha")
    counts = doc["fingering"]
    assert counts["hit_count"] == 850
    assert counts["finger_hit_counts"]["left_1"] == 430
    assert counts["finger_hit_counts"]["right_1"] == 420
    assert counts["hand_hit_counts"] == {"left": 430, "right": 420}
    assert counts["central_pair_hits"] == 850
    assert counts["same_target_time_adjacent_pairs"] == 0
    assert counts["adjacent_same_finger_under_20ms"] == 0
    assert counts["adjacent_hit_gaps"]["samples"] == 830
    assert counts["mae_ms_from_events"] == 0.
    assert doc["motor_commands"]["positive_command_runs"]["left_1"] == 20
    assert doc["motor_commands"]["positive_command_runs"]["right_1"] == 20
    assert doc["no_counterfactual_claims"] is True
    assert doc["no_human_expert_claims"] is True
    assert doc["no_training"] is True


def test_tiny_chord_detection_uses_target_not_keydown_time():
    row = dict(
        anchor=1, chart="Chord", final_physics={"hits": 3},
        final_hit_events=[
            dict(target_index=0, time_s=.100, error_ms=0., key="left_1", judgement="perfect"),
            dict(target_index=1, time_s=.101, error_ms=1., key="right_1", judgement="perfect"),
            dict(target_index=2, time_s=.150, error_ms=0., key="left_1", judgement="perfect"),
        ],
    )
    stats = audit._hit_summary([row])
    assert stats["same_target_time_adjacent_pairs"] == 1
    assert stats["adjacent_hit_gaps"]["under_20ms"] == 1
    assert stats["same_finger_inter_hit_gaps"]["left_1"]["min_ms"] == pytest.approx(50.)


@pytest.mark.parametrize("tamper", [
    lambda r, p: r.update(role="Validation20"),
    lambda r, p: p.update(role="Validation20"),
    lambda r, p: r["original_stage_a_gate"].update(may_run_stage_c=True),
    lambda r, p: r["gates"].update(full_train20_safe850=False),
    lambda r, p: r["candidate"]["physics"].update(misses=1),
    lambda r, p: r["per_anchor"][0]["final_hit_events"][0].update(key="unknown"),
    lambda r, p: r["per_anchor"][0]["final_hit_events"][0].update(error_ms=float("nan")),
    lambda r, p: p["plans"][0].update(actions_8k=[[0.] * 7]),
    lambda r, p: p["plans"][0].update(actions_8k=[[0.] * 7 + [2.]]),
    lambda r, p: p["plans"][0].update(independently_replayed=False),
    lambda r, p: r["per_anchor"][0]["final_physics"].update(misses=1),
])
def test_fail_closed_on_corrupt_or_unsafe_teacher_data(tamper):
    report, plans = _fake_docs()
    tamper(report, plans)
    with pytest.raises((ValueError, KeyError)):
        audit.audit_documents(
            report, plans, plan_file_sha256="synthetic-plan-sha")


def test_source_hash_is_required():
    report, plans = _fake_docs()
    with pytest.raises(ValueError, match="provenance"):
        audit.audit_documents(report, plans, plan_file_sha256="wrong")


def test_write_once_never_changes_old_audit(tmp_path):
    output = tmp_path / "stage_c_v30_fingering_audit.json"
    audit.write_once(output, {"test": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"test": 1}
    with pytest.raises(FileExistsError):
        audit.write_once(output, {"test": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"test": 1}
