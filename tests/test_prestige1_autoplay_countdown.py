from __future__ import annotations

from math import pi, sin, cos
from types import SimpleNamespace

import pytest
import torch

from dmdod.features.real_chart import REAL_CHART_FLOOR_FEATURE_DIM
from dmdod.training import prestige1_autoplay_countdown as clock
from dmdod.training import prestige1_autoplay_readout as readout


def sequence(*, anchor=2):
    # Exact same-event countdown of 10ms, followed by next-event jump.
    truth = torch.tensor([.100, .090, .080, .200, .190, .180])
    events = (0, 0, 0, 1, 1, 1)
    visible = torch.zeros(6, 263)
    near = torch.ones(6, dtype=torch.bool)
    mask = torch.ones(6, dtype=torch.bool)
    return readout.TimingSequence(
        anchor=anchor, chart_sha256=f"{anchor:064x}",
        visible=visible, remaining_s=truth,
        near=near, mask=mask, event_indices=events,
    )


def test_countdown_correctly_separates_same_target_and_switch():
    s = sequence()
    perfect = [float(v) for v in s.remaining_s.tolist()]
    result = clock._pair_dynamics(s, perfect)
    assert len(result["same_residuals"]) == 4
    assert max(abs(v) for v in result["same_residuals"]) < 1e-4
    assert result["switch_truth_positive"] == 1
    assert result["switch_predicted_positive"] == 1
    assert len(result["switch_residuals"]) == 1
    assert result["switch_residuals"][0] == pytest.approx(0., abs=1e-5)


def test_constant_predictor_is_not_a_countdown():
    s = sequence()
    constant = [0.07] * 6
    result = clock._pair_dynamics(s, constant)
    assert result["same_residuals"] == pytest.approx([10.] * 4, abs=1e-4)
    assert result["switch_truth_positive"] == 1
    assert result["switch_predicted_positive"] == 0
    summary = clock._summary(result["same_residuals"])
    assert summary["mae_ms"] == pytest.approx(10., abs=1e-4)
    assert summary["within_5ms"] == 0.


def test_predictions_can_count_down_but_be_offset():
    s = sequence()
    # Numerically correct slope but 20ms late everywhere; global MAE bad.
    shifted = [float(v) + .020 for v in s.remaining_s.tolist()]
    residual = clock._pair_dynamics(s, shifted)
    assert max(abs(v) for v in residual["same_residuals"]) < 1e-4
    assert abs(shifted[0] - float(s.remaining_s[0])) > .019


def test_event_switch_is_excluded_even_when_next_event_within_ten_ms():
    s = sequence()
    # The next target comes only 8ms after the old one. Its observed
    # remaining time can decrease across the event-index boundary.
    changed = readout.TimingSequence(
        anchor=s.anchor, chart_sha256=s.chart_sha256, visible=s.visible,
        remaining_s=torch.tensor([.025, .015, .005, .003, .0, .0]),
        near=s.near,
        mask=torch.tensor([True, True, True, True, False, False]),
        event_indices=(0, 0, 0, 1, 1, 1),
    )
    result = clock._pair_dynamics(
        changed, [float(v) for v in changed.remaining_s])
    assert len(result["same_residuals"]) == 2
    assert len(result["switch_residuals"]) == 1
    assert result["switch_truth_positive"] == 0
    assert result["switch_truth_nonpositive"] == 1
    assert result["switch_residuals"][0] == pytest.approx(0., abs=1e-5)


def _floor(slot: int, features: list[float], *,
           entry: float = 0., exit: float = pi, special: int | None = None):
    off = clock.SLOT_START + slot * REAL_CHART_FLOOR_FEATURE_DIM
    values = [0.] * REAL_CHART_FLOOR_FEATURE_DIM
    values[2:6] = [sin(entry), cos(entry), sin(exit), cos(exit)]
    if special is not None:
        values[special] = 1.
    values[-1] = 1.
    features[off:off+REAL_CHART_FLOOR_FEATURE_DIM] = values


def hud(theta: float, *, special=None):
    result = [0.] * 263
    result[24:26] = [sin(theta), cos(theta)]
    for slot in (clock.CURRENT_SLOT - 1, clock.CURRENT_SLOT, clock.NEXT_SLOT):
        _floor(slot, result, special=special if slot == clock.CURRENT_SLOT else None)
    result[clock.HUD_OFFSET] = 0.  # Tile BPM = 120, angular rate 2pi rad/s
    return tuple(result)


def test_simple_hud_geometry_predicts_angle_over_speed():
    prev = hud(.4)
    now = hud(.4 + 2*pi*.010)
    prediction, status = clock.geometry_one(prev, now)
    assert status == "eligible_simple_floor"
    assert prediction == pytest.approx(
        (pi - (.4 + 2*pi*.010)) / (2*pi), abs=1e-6)


@pytest.mark.parametrize("special,status", [
    (6, "midspin_or_multiplanet"),
    (7, "midspin_or_multiplanet"),
    (8, "special_event"),  # Twirl
    (9, "special_event"),  # SetSpeed
    (11, "special_event"), # Pause
])
def test_geometry_abstains_from_unhandled_special_events(special, status):
    prev = hud(.4, special=special)
    now = hud(.4 + 2*pi*.01, special=special)
    assert clock.geometry_one(prev, now)[1] == status


def test_geometry_requires_motion_and_no_lookahead():
    past = hud(.4)
    assert clock.geometry_one(past, past)[1] == "stopped_orbit"
    with pytest.raises(ValueError, match="263D"):
        clock.geometry_one((0.,) * 262, past)
    future = list(hud(.4 + 2*pi*.01))
    # Removing next visible tile masks the geometry baseline, not replacing
    # its target with an oracle timestamp.
    idx = clock.SLOT_START + clock.NEXT_SLOT * clock.FLOOR_DIM + 14
    future[idx] = 0.
    assert clock.geometry_one(past, tuple(future))[1] == "unavailable_ordinary_floor"


def test_svg_creates_static_honest_visualization_with_switch_markers():
    s = sequence()
    truth = [float(v) for v in s.remaining_s]
    svg = clock.svg_timing_trace(
        s, {"constant": [0.07]*6, "mlp": truth, "gru": truth})
    assert svg.startswith("<svg ")
    assert "</svg>" in svg
    assert "Auto truth" in svg
    assert "GRU" in svg and "MLP" in svg
    assert "stroke-dasharray" in svg
    assert "display capped at 350ms" in svg


def test_original_stage_t_loss_contribution_sums_to_total():
    source = sequence()
    breakdown = clock.loss_contributions(
        None, [source], {"time_s": .070, "near_probability": .5})
    rows = breakdown["by_distance"]
    assert sum(rows[name]["frames"] for name in rows) == 6
    assert sum(rows[name]["time_term_normalized_contribution"]
               for name in rows) == pytest.approx(breakdown["time_term_total"])
    assert sum(rows[name]["near_event_bce_normalized_contribution"]
               for name in rows) == pytest.approx(breakdown["event_bce_total"])
    assert sum(rows[name]["time_term_fraction"]
               for name in rows) == pytest.approx(1.)
    assert sum(rows[name]["bce_fraction"]
               for name in rows) == pytest.approx(1.)
    assert breakdown["event_bce_total"] == pytest.approx(.69314718, abs=1e-5)


def test_countdown_audit_never_unlocks_stage_a():
    assert clock.FORMAT != readout.FORMAT
    assert clock.REPORT != readout.REPORT
    assert clock.CONTROL_S == .010
    assert clock.PLOT_ANCHORS == (2, 7, 13, 14, 20)
