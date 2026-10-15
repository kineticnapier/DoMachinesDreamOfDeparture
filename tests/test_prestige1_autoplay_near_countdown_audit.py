from __future__ import annotations

from copy import deepcopy
from math import pi, sin, cos
from types import SimpleNamespace

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as old
from dmdod.training import prestige1_autoplay_near_countdown_audit as audit
from dmdod.training import prestige1_autoplay_readout as t


def seq():
    frames = torch.zeros(7, p.INPUT_DIM)
    remaining = torch.tensor([.38, .12, .11, .10, .25, .24, .23])
    return t.TimingSequence(
        anchor=2,
        chart_sha256="a"*64,
        visible=frames,
        remaining_s=remaining,
        near=torch.tensor([False, True, True, True, False, False, False]),
        mask=torch.ones(7, dtype=torch.bool),
        event_indices=(0, 1, 1, 1, 2, 2, 2),
    )


def test_near_only_ablation_metric_does_not_prove_a_countdown():
    subject = seq()
    stable = old._pair_dynamics(subject, [.1]*7)
    # Zero slope predicts a constant; error to ideal -10ms remains +10ms.
    assert stable["same_near_residuals"] == pytest.approx([10.]*2, abs=.001)
    perfect = old._pair_dynamics(
        subject, [float(x) for x in subject.remaining_s.tolist()])
    assert max(map(abs, perfect["same_near_residuals"])) < .001
    old_score = old.dynamics(None, [subject], {
        "time_s": .1, "near_probability": .3,
    })
    numbers = audit._near_slope_metrics(old_score)
    assert numbers["mean_predicted_near_step_ms"] == pytest.approx(0, abs=.001)
    assert numbers["near_countdown_mae_ms"] == pytest.approx(10, abs=.001)
    assert numbers["near_pair_count"] == 2


def test_compare_restored_checkpoint_metrics_fail_closed_on_mae_and_counts():
    subject = seq()
    reference = {"time_s": .1, "near_probability": .3}
    saved = t.evaluate(None, [subject], reference)
    audit._check_scores(saved, deepcopy(saved), label="synthetic")
    changed = deepcopy(saved)
    changed["aggregate"]["near_mae_ms"] += .1
    with pytest.raises(RuntimeError, match="frozen error mismatch"):
        audit._check_scores(saved, changed, label="synthetic")
    changed = deepcopy(saved)
    changed["aggregate"]["fp"] += 1
    with pytest.raises(RuntimeError, match="frozen score count mismatch"):
        audit._check_scores(saved, changed, label="synthetic")
    changed = deepcopy(saved)
    changed["per_anchor"][0]["near_mae_ms"] += 1.
    with pytest.raises(RuntimeError, match="frozen anchor MAE mismatch"):
        audit._check_scores(saved, changed, label="synthetic")


def test_missing_checkpoint_or_frozen_ablation_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required frozen"):
        audit._restore_near_only(
            SimpleNamespace(), SimpleNamespace(root=tmp_path),
        )
    assert not list(tmp_path.iterdir())


def _add_floor(frame, slot: int):
    i = old.SLOT_START + slot * old.FLOOR_DIM
    frame[i+2:i+6] = torch.tensor([sin(0), cos(0), sin(pi), cos(pi)])
    frame[i+14] = 1.


def simple_hud_sequence():
    obs = torch.zeros(3, p.INPUT_DIM)
    for i, theta in enumerate((.4, .4+2*pi*.01, .4+2*pi*.02)):
        obs[i, 24] = sin(theta)
        obs[i, 25] = cos(theta)
        for slot in (old.CURRENT_SLOT-1, old.CURRENT_SLOT, old.NEXT_SLOT):
            _add_floor(obs[i], slot)
        obs[i, old.HUD_OFFSET] = 0.  # 120 Tile BPM.
    return t.TimingSequence(
        anchor=2, chart_sha256="f"*64, visible=obs,
        remaining_s=torch.tensor([.435, .425, .415]),
        near=torch.ones(3, dtype=torch.bool),
        mask=torch.ones(3, dtype=torch.bool), event_indices=(0, 0, 0),
    )


def test_hud_only_geometry_matches_exact_same_frame_subset_for_two_grus():
    sequence = simple_hud_sequence()
    torch.manual_seed(123)
    model = t.TimingReadout(16, recurrent=True).eval()
    result = audit._geometry_near_subset(
        [sequence], model, model,
        {"time_s": .1, "near_probability": .5},
    )
    assert result["matched_only"] is True
    assert result["eligible_near_frames"] == 2
    assert result["error"]["old_gru"]["count"] == 2
    assert result["error"]["near_only_gru"]["count"] == 2
    assert result["error"]["old_gru"]["mae_ms"] == pytest.approx(
        result["error"]["near_only_gru"]["mae_ms"]
    )
    assert result["error"]["geometry"]["count"] == 2


def test_saved_model_overlay_is_svg_without_future_input():
    subject = seq()
    model = t.TimingReadout(8, recurrent=True).eval()
    svg = audit._comparison_svg(
        subject, {"time_s": .1, "near_probability": .5}, model, model,
    )
    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>\n")
    for label in ("Auto truth", "Constant", "Old GRU", "Near-only GRU"):
        assert label in svg
    assert "stroke-dasharray" in svg
    assert "view capped at 350ms" in svg
    assert "Validation" not in svg


def test_new_audit_never_enables_motor_or_overwrites_original_results():
    assert audit.FORMAT != t.FORMAT
    assert audit.REPORT != t.REPORT
    assert audit.REPORT != old.REPORT
    assert audit.REPORT != "stage_timing_autoplay_near_only_gru_train20.json"
    assert old.PLOT_ANCHORS == (2, 7, 13, 14, 20)
    assert audit.PLOT_PREFIX != old.PLOT_PREFIX
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
