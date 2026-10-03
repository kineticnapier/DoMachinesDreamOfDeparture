from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import sweep_n_key_action_thresholds as sweep


def test_parse_float_list_accepts_comma_separated_thresholds() -> None:
    assert sweep._parse_float_list("0.25, 0.5,0.70") == (0.25, 0.5, 0.70)
    with pytest.raises(ValueError, match="must not be empty"):
        sweep._parse_float_list(" , ")


def test_discretize_action_maps_soft_outputs_to_hard_commands() -> None:
    action = sweep.discretize_action(
        (-0.31, -0.30, -0.29, 0.24, 0.25, 0.50, 0.70, 0.90),
        press_threshold=0.50,
        release_threshold=-0.30,
    )

    assert action.values == (-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)


def test_discretize_action_rejects_invalid_threshold_ranges() -> None:
    with pytest.raises(ValueError, match="press_threshold"):
        sweep.discretize_action((0.0, 0.0), press_threshold=0.0, release_threshold=-0.3)
    with pytest.raises(ValueError, match="release_threshold"):
        sweep.discretize_action((0.0, 0.0), press_threshold=0.5, release_threshold=0.0)


def test_aggregate_combines_xacc_components_across_anchors() -> None:
    first = SimpleNamespace(
        hits=8,
        targets=10,
        too_early_presses=1,
        overloaded=False,
        x_accuracy_points=7.0,
        x_accuracy_denominator=10.0,
    )
    second = SimpleNamespace(
        hits=9,
        targets=20,
        too_early_presses=2,
        overloaded=True,
        x_accuracy_points=11.0,
        x_accuracy_denominator=20.0,
    )

    aggregate = sweep._aggregate([(first, 9), (second, 12)])

    assert aggregate.hits == 17
    assert aggregate.targets == 30
    assert aggregate.x_accuracy_percent == pytest.approx(60.0)
    assert aggregate.early == 3
    assert aggregate.overloaded is True
    assert aggregate.keydowns == 21
