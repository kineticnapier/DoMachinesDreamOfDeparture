from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from scripts import run_countdown_hud_v3_motor_phase3 as runner


def test_fixed_protocol_constants() -> None:
    assert runner.ARMS == ("hud_v14", "hud_v3_composite")
    assert runner.CONTROL_DT_S == .010
    assert runner.PHYSICS_DT_S == .001


def test_policy_loop_does_not_read_evaluator_truth() -> None:
    forbidden = {
        "_used", "_missed", "targets", "target_time_s",
        "target_ordinal", "judgement_trace",
    }
    # Posthoc-only tail starts after the explicit boundary comment and calls;
    # inspect the loop prefix independently from evaluator joining.
    prefix = textwrap.dedent(inspect.getsource(runner.simulate)).split(
        "# Evaluator truth is joined only after the final action"
    )[0]
    attrs = {
        node.attr for node in ast.walk(ast.parse(prefix))
        if isinstance(node, ast.Attribute)
    }
    assert attrs.isdisjoint(forbidden)


def test_output_directory_is_create_only(tmp_path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        existing.mkdir(parents=True, exist_ok=False)


def test_cli_help_starts_from_repository_root() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_countdown_hud_v3_motor_phase3.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stderr
