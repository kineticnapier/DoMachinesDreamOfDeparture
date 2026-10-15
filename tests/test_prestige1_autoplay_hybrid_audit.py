from __future__ import annotations

from dataclasses import replace
from math import cos, pi, sin

import pytest
import torch

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_autoplay_countdown as cd
from dmdod.training import prestige1_autoplay_hybrid_audit as hy
from dmdod.training import prestige1_autoplay_readout as st


class FrozenStub(torch.nn.Module):
    """A fixed inference-only readout, NOT trained and has no oracle inputs."""
    def forward(self, visible):
        # Softplus inverse gives a non-negative 90ms baseline on every frame.
        raw = torch.log(torch.expm1(visible.new_tensor(.09/st.TIME_UNIT_S)))
        logits = torch.zeros(len(visible), 2, device=visible.device)
        logits[:, 0] = raw
        return logits


def hud(theta: float, *, twirl=False) -> torch.Tensor:
    frame = torch.zeros(p.INPUT_DIM)
    frame[24:26] = torch.tensor((sin(theta), cos(theta)))
    for slot in (cd.CURRENT_SLOT-1, cd.CURRENT_SLOT, cd.NEXT_SLOT):
        off = cd.SLOT_START + slot*cd.FLOOR_DIM
        frame[off+2:off+6] = torch.tensor([0., 1., sin(pi), cos(pi)])
        frame[off+14] = 1.
    if twirl:
        off = cd.SLOT_START + cd.CURRENT_SLOT*cd.FLOOR_DIM
        frame[off+8] = 1.
    frame[cd.HUD_OFFSET] = 0.  # tile BPM 120
    return frame


def sequence(event_switch=False):
    # Ordinary HUD angle yields an almost exact 10ms physical countdown.
    angles = [2.65 + 2*pi*.01*k for k in range(6)]
    visible = torch.stack([hud(angle, twirl=k==3)
                           for k, angle in enumerate(angles)])
    true_seconds = [(pi-angle)/(2*pi) for angle in angles]
    if event_switch:
        # Oracle-only event transition, never fed to inference.
        true_seconds[3:] = [.16, .15, .14]
        events = (0,0,0,1,1,1)
    else:
        events = (0,0,0,0,0,0)
    return st.TimingSequence(
        anchor=2, chart_sha256="a"*64,
        visible=visible, remaining_s=torch.tensor(true_seconds),
        near=torch.ones(6,dtype=torch.bool),
        mask=torch.ones(6,dtype=torch.bool),
        event_indices=events,
    )


def test_hybrid_selects_geometry_by_hud_only_and_falls_back_on_events():
    seq = sequence()
    trace = hy.infer_hybrid(seq.visible, FrozenStub())
    assert trace.mode == ("gru", "geometry", "geometry", "gru", "gru", "geometry")
    assert trace.abstain_reasons[0] == "no_previous_hud_frame"
    assert trace.abstain_reasons[3] == "special_event"
    assert trace.abstain_reasons[4] == "floor_changed"
    for i, mode in enumerate(trace.mode):
        if mode == "geometry":
            assert trace.geometry_s[i] is not None
            assert trace.chosen_s[i] == trace.geometry_s[i]
        else:
            assert trace.geometry_s[i] is None
            assert trace.chosen_s[i] == pytest.approx(.09, abs=1e-6)


def test_predictions_are_independent_of_hidden_oracle_labels_and_ids():
    first = sequence()
    changed = replace(
        first, event_indices=(12,13,14,15,16,17),
        remaining_s=torch.ones(6)*.5,
        near=torch.zeros(6,dtype=torch.bool),
        mask=torch.zeros(6,dtype=torch.bool))
    a = hy.infer_hybrid(first.visible, FrozenStub())
    b = hy.infer_hybrid(changed.visible, FrozenStub())
    assert a == b


def test_forward_only_observation_prefix_is_causal():
    seq = sequence()
    a = hy.infer_hybrid(seq.visible[:4], FrozenStub())
    b = hy.infer_hybrid(seq.visible, FrozenStub())
    assert a.chosen_s == b.chosen_s[:4]
    assert a.mode == b.mode[:4]
    assert a.abstain_reasons == b.abstain_reasons[:4]


def test_all_near_hybrid_mae_is_frame_matched_and_never_uses_oracle_for_switch():
    seq = sequence()
    traces = {2: hy.infer_hybrid(seq.visible, FrozenStub())}
    result = hy._aggregate_scores([seq], traces, {"time_s":.08})
    assert result["visible_frames"] == 6
    assert result["near_frames"] == 6
    assert result["geometry_visible_frames"] == 3
    assert result["geometry_near_frames"] == 3
    assert result["geometry_near_coverage"] == .5
    assert result["matched_baselines"]["geometry"]["near"]["count"] == 3
    assert result["matched_baselines"]["near_only_gru"]["near"]["count"] == 6
    # Hybrid must use the exact on-frame Geometry errors + fallback GRU errors.
    per_frame = [
        abs(value-float(seq.remaining_s[k]))*1000
        for k,value in enumerate(traces[2].chosen_s)
    ]
    expected = sum(per_frame)/len(per_frame)
    assert result["matched_baselines"]["hybrid"]["near"]["mae_ms"] == pytest.approx(expected)


def test_transition_residual_separates_model_change_from_true_note_change():
    # Force an oracle event switch exactly when Geometry becomes unavailable.
    seq = sequence(event_switch=True)
    trace = hy.infer_hybrid(seq.visible, FrozenStub())
    result = hy._switching_scores([seq], {2:trace})
    pair = result["by_mode_transition_and_oracle_event"]
    assert pair["geometry->gru"]["event_changed"]["pairs"] == 1
    assert pair["geometry->gru"]["same_event"]["pairs"] == 0
    assert pair["gru->geometry"]["same_event"]["pairs"] == 1
    assert result["event_switches_observed"] == 1
    assert result["positive_oracle_event_jumps"] == 1
    assert result["same_oracle_event_pairs"] == 4
    assert result["all_hud_mode_pair_counts"]["geometry->gru"] == 1
    assert result["all_hud_mode_pair_counts"]["gru->geometry"] == 2


def test_mode_transition_has_nonzero_jump_error_even_if_each_model_finite():
    seq = sequence()
    trace = hy.infer_hybrid(seq.visible, FrozenStub())
    stats = hy._switching_scores([seq], {2:trace})
    g_to_gru = stats["by_mode_transition_and_oracle_event"]["geometry->gru"]["same_event"]
    assert g_to_gru["pairs"] == 1
    assert g_to_gru["hybrid_countdown_or_jump_residual"]["mae_ms"] > 10.
    assert g_to_gru["after_hybrid_absolute_error"]["count"] == 1
    assert g_to_gru["spikes_over_20ms"] == 1
    assert stats["event_switches_observed"] == 0
    assert stats["positive_jump_direction_recall"] is None


def test_all_unobservable_oracle_frames_do_not_affect_prediction():
    seq = sequence()
    trace = hy.infer_hybrid(seq.visible, FrozenStub())
    masked = replace(seq, mask=torch.zeros(6,dtype=torch.bool))
    with pytest.raises(RuntimeError, match="no scored"):
        hy._aggregate_scores([masked], {2:trace}, {"time_s": .08})
    switches = hy._switching_scores([masked], {2:trace})
    assert switches["event_switches_observed"] == 0
    assert switches["all_hud_mode_pair_counts"]["geometry->gru"] == 1
    assert switches["observable_mode_pair_counts"] == {}


def test_input_dimension_and_nonfinite_data_fail_before_inference():
    stub = FrozenStub()
    with pytest.raises(ValueError, match="263"):
        hy.infer_hybrid(torch.zeros(6,264), stub)
    with pytest.raises(ValueError, match="Nonfinite"):
        wrong = sequence().visible.clone()
        wrong[0,0] = float("nan")
        hy.infer_hybrid(wrong, stub)


def test_hybrid_plot_reuses_same_frame_predictions_and_event_labels_only_as_annotations():
    seq = sequence(event_switch=True)
    trace = hy.infer_hybrid(seq.visible, FrozenStub())
    svg = hy._plot(seq,trace)
    assert svg.startswith("<svg ")
    assert svg.endswith("</svg>\n")
    assert "Framewise Hybrid" in svg
    assert "Near-only GRU" in svg
    assert "stroke-dasharray" in svg
    assert "view capped at 350ms" in svg


def test_hybrid_research_only_and_fixed():
    assert hy.REPORT not in (cd.REPORT, "stage_timing_autoplay_near_only_countdown_train20.json")
    assert hy.JUMP_WARNING_MS == 20.0
    assert hy.TRANSITIONS == ("geometry->gru", "gru->geometry")
    assert p.STAGE_A_REPORT == "stage_a_v2.json"
