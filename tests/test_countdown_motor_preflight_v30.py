from __future__ import annotations

import ast
import inspect

from scripts import audit_countdown_motor_preflight_v30 as audit


def test_collection_loop_does_not_read_target_truth() -> None:
    tree = ast.parse(inspect.getsource(audit.collect))
    attributes = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert attributes.isdisjoint({
        "targets", "target_time_s", "episode_time_s", "_used", "_missed",
    })


def test_evaluator_is_separate_from_collection() -> None:
    collection = inspect.signature(audit.collect)
    evaluator = inspect.signature(audit.evaluate)
    assert "segment" not in collection.parameters
    assert "segment" in evaluator.parameters
    assert audit.NORMAL_LEAD_S == .044
    assert audit.DECISION_LEAD_S == .069
