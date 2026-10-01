from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v110_rejection_telemetry as telemetry


def _decision(accepted: bool, reason: str = ""):
    return SimpleNamespace(accepted=accepted, reason=reason)


def test_reason_bucket_groups_guard_failures() -> None:
    assert telemetry._reason_bucket("not better") == "not-better"
    assert telemetry._reason_bucket("validation XAcc regression>2pt") == "XAcc"
    assert telemetry._reason_bucket("anchor hit regression>3") == "hit"
    assert telemetry._reason_bucket("anchor early regression>4") == "early"
    assert telemetry._reason_bucket("validation safe->overload") == "overload"


def test_anchor_stage_distinguishes_start_micro() -> None:
    assert telemetry._anchor_stage(SimpleNamespace(role="start-micro-4")) == "start-micro"
    assert telemetry._anchor_stage(SimpleNamespace(role="anchor-1")) == "anchor"


def test_terminal_stage_reports_first_guard_layer() -> None:
    train_dead = SimpleNamespace(
        train_decision=_decision(False),
        validation_decisions=(),
        anchor_decisions=(),
    )
    validation_dead = SimpleNamespace(
        train_decision=_decision(True),
        validation_decisions=(_decision(True), _decision(False)),
        anchor_decisions=(),
    )
    anchor_dead = SimpleNamespace(
        train_decision=_decision(True),
        validation_decisions=(_decision(True), _decision(True)),
        anchor_decisions=(_decision(True), _decision(False)),
    )
    complete = SimpleNamespace(
        train_decision=_decision(True),
        validation_decisions=(_decision(True), _decision(True)),
        anchor_decisions=(_decision(True), _decision(True)),
    )

    assert telemetry._terminal_stage(train_dead, validation_count=2, anchor_count=2) == "train"
    assert telemetry._terminal_stage(validation_dead, validation_count=2, anchor_count=2) == "validation"
    assert telemetry._terminal_stage(anchor_dead, validation_count=2, anchor_count=2) == "anchor"
    assert telemetry._terminal_stage(complete, validation_count=2, anchor_count=2) == "complete"
