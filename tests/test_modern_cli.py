from __future__ import annotations

from dmdod.modern_cli import parse_training_line


def test_parse_teacher_turbo_start():
    event = parse_training_line(
        "teacher-turbo: anchors=96 cache-hit=64 generate=32 workers=12 dir=checkpoints/.teacher-cache"
    )
    assert event is not None
    assert event.kind == "teacher_start"
    assert event.values == {
        "total": 96,
        "cache_hits": 64,
        "generate": 32,
        "workers": 12,
    }


def test_parse_resume_progress():
    event = parse_training_line(
        "resume=checkpoints/real_chart_v090_press_persistence.pt completed-round=5/48 V=H1/1 A=H2/2"
    )
    assert event is not None
    assert event.kind == "resume"
    assert event.values["completed"] == 5
    assert event.values["total"] == 48


def test_parse_bootstrap_keep_and_prune():
    kept = parse_training_line(
        "bootstrap 12: loss=0.123456 safe=True completion=94.2% meanX=91.7% KEEP"
    )
    assert kept is not None
    assert kept.kind == "bootstrap_step"
    assert kept.values["epoch"] == 12
    assert kept.values["status"] == "KEEP"
    assert kept.values["loss"] == 0.123456
    assert kept.values["completion"] == 94.2
    assert kept.values["mean_x"] == 91.7

    pruned = parse_training_line(
        "bootstrap 13: loss=0.130000 PRUNE[safety] eval=12/106"
    )
    assert pruned is not None
    assert pruned.kind == "bootstrap_step"
    assert pruned.values["status"] == "PRUNE:safety"
    assert pruned.values["evaluated"] == 12
    assert pruned.values["eval_total"] == 106


def test_parse_round_and_epoch_postfix_metrics():
    started = parse_training_line(
        "round 006 chart=Example Chart train=10.00..40.00s base=H10/12 X88.5% | mix50=H11/12 X90.2%"
    )
    assert started is not None
    assert started.kind == "round_start"
    assert started.values["round"] == 6
    assert started.values["chart"] == "Example Chart"
    assert started.values["base_x"] == 88.5
    assert started.values["mix_x"] == 90.2

    accepted = parse_training_line(
        "round 006 e03: ACCEPT a=0.25 loss=0.0312 T=H11/12 X91.1% V=H10/10 A=H90/96"
    )
    assert accepted is not None
    assert accepted.kind == "epoch_step"
    assert accepted.values["status"] == "ACCEPT"
    assert accepted.values["alpha"] == 0.25
    assert accepted.values["loss"] == 0.0312
    assert accepted.values["x"] == 91.1

    rollback = parse_training_line("round 006 e04: ROLLBACK loss=0.0299")
    assert rollback is not None
    assert rollback.kind == "epoch_step"
    assert rollback.values["status"] == "ROLLBACK"
    assert rollback.values["loss"] == 0.0299


def test_parse_fixed_point_skip_and_checkpoint():
    skipped = parse_training_line(
        "round 006: exact odd/even proposal fixed point; skip remaining 7 epochs"
    )
    assert skipped is not None
    assert skipped.kind == "epoch_skip"
    assert skipped.values["remaining"] == 7

    checkpoint = parse_training_line(
        "checkpoint round 006: checkpoints/real_chart_v090_press_persistence.pt"
    )
    assert checkpoint is not None
    assert checkpoint.kind == "checkpoint"
    assert checkpoint.values["round"] == 6
