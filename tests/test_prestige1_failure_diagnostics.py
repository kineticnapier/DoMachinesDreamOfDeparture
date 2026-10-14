from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dmdod.envs.n_key import (
    EpisodeTerminationTrace, NKeyActionFrameTrace, NKeyOverloadTrace,
    NKeyPhysicalKeyDownTrace, TooEarlyKeyDownTrace,
)
from dmdod.training import prestige1_failure_diagnostics as diag
from dmdod.training import prestige1 as p


def config(tmp_path):
    return p.PrestigeConfig(
        dataset="TEST", output_dir=str(tmp_path), core_path="fake-core",
        old_best_path="fake-best", device="cpu")


def test_checkpoint_digest_rejects_mutated_checkpoint(tmp_path):
    c = config(tmp_path)
    folder = diag.transfer.checkpoint_dir(c)
    folder.mkdir(parents=True)
    (folder / "control_v2.pt").write_bytes(b"unexpected checkpoint data")
    stage_c = {
        "effective_updates_per_arm": 154,
        "_validated_train_anchor_manifest": [],
        "results": {"control": {"checkpoint_sha256": "0"*64}},
    }
    with pytest.raises(RuntimeError, match="hash mismatch"):
        diag._load_model(c, arm="control", device=torch.device("cpu"),
                         stage_b={}, stage_c=stage_c, expected_train_manifest=[])


def test_replay_fails_closed_on_different_hits_or_chart():
    stored = {"chart": "A", "hits": 1, "misses": 2,
              "too_early": 1, "overload": True, "keydowns": 3}
    record = {"chart": "A", "stats": dict(stored)}
    diag._verify_replay(record, stored, anchor=7, arm="prestige")
    record["stats"]["hits"] = 2
    with pytest.raises(RuntimeError, match="divergence"):
        diag._verify_replay(record, stored, anchor=7, arm="prestige")
    record["stats"]["hits"] = 1
    record["chart"] = "B"
    with pytest.raises(RuntimeError, match="chart name"):
        diag._verify_replay(record, stored, anchor=7, arm="prestige")


def test_readonly_refuses_existing_report_before_loading(tmp_path):
    c = config(tmp_path)
    folder = diag.transfer.output_dir(c)
    folder.mkdir(parents=True)
    (folder / diag.REPORT).write_text("original")
    with pytest.raises(FileExistsError):
        diag.run_readonly_diagnostics(c)
    assert (folder / diag.REPORT).read_text() == "original"


def test_anchor_selection_requires_unique_validation_indices(tmp_path):
    c = config(tmp_path)
    for anchors in ([], [7,7], [0], [21], ["7"], [True]):
        with pytest.raises(ValueError):
            diag.run_readonly_diagnostics(c, anchors=anchors)


def test_trace_stays_synchronized_and_counts_unjudged(monkeypatch):
    early = TooEarlyKeyDownTrace(
        time_s=.005, key="left_4", target_index=0, error_ms=-95.,
        overload_before=5., overload_after=6., fail_overload=True)
    down = NKeyPhysicalKeyDownTrace(
        time_s=.005, key="left_4", target_index=0, target_time_s=.1)
    terminal = EpisodeTerminationTrace(
        time_s=.005, reason="Overload", hits=0, too_early=1,
        keydowns=1, next_target_index=0, overload_value=6.)

    class FakeEnv:
        misalign = False

        def __init__(self, *args, **kwargs):
            assert kwargs["capture_overload_trace"] is True
            assert kwargs["key_count"] == 8
            self.physical_keydowns = 1
            self.stats = SimpleNamespace(
                targets=1, hits=0, misses=0,
                too_early_presses=1, overloaded=True, x_accuracy_percent=0.)

        def reset(self):
            return SimpleNamespace()

        def privileged_episode_time_s(self):
            return 0.

        def _floor_index_at_chart_time(self, chart_time):
            return 0

        def step(self, action):
            values = tuple(action.values)
            if self.misalign:
                values = (0.5,) + values[1:]
            self.trace = NKeyOverloadTrace(
                termination=terminal,
                keydown_times_s=(.005,),
                too_early_events=(early,),
                action_frames=(
                    NKeyActionFrameTrace(
                        time_s=0., action_values=values,
                        positions_m=(0.,)*8, velocities_m_s=(0.,)*8,
                        pressed_flags=(False,)*8,
                        next_target_index=0, next_target_time_s=.1),
                ),
                physical_keydowns=(down,))
            return SimpleNamespace(observation=SimpleNamespace(), done=True)

        @property
        def overload_trace(self):
            return self.trace

    class DummyModel:
        def eval(self):
            pass

        def initial_state(self, device):
            return None

        def forward_step_with_timing(self, x, state):
            assert x.shape == (263,)
            return (torch.zeros(8), None, None, state, torch.zeros(10))

    monkeypatch.setattr(diag, "DiagnosticHudNKeyRealChartMotorEnv", FakeEnv)
    monkeypatch.setattr(
        diag, "encode_n_key_hud_real_chart_observation", lambda observation: (0.,)*263)
    segment = SimpleNamespace(
        duration_s=1., chart_time_from_episode=lambda t:t,
        targets=[SimpleNamespace(episode_time_s=.1,floor_index=0)])
    named = SimpleNamespace(segment=segment, chart_name="Synthetic")
    result = diag._closed_loop(DummyModel(), named, device=torch.device("cpu"), anchor=7)
    assert result["stats"]["unjudged_targets"] == 1
    assert result["termination"]["time_s"] == .005
    assert result["termination"]["reason"] == "Overload"
    assert result["executed_frames"] == 1
    assert len(result["last_5_control_frames"]) == 1
    assert len(result["keydowns_last_150ms"]) == 1
    assert result["too_early_last_150ms"][0]["fail_overload"] is True
    assert result["closed_loop_timing"]["valid_frames"] == 1

    FakeEnv.misalign = True
    with pytest.raises(RuntimeError, match="not synchronized"):
        diag._closed_loop(DummyModel(), named, device=torch.device("cpu"), anchor=7)
