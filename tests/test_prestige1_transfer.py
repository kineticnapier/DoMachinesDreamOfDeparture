from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pytest
import torch
from torch import nn

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_transfer as bc


def config(tmp_path):
    core = tmp_path / "core4096.pt"
    core.write_bytes(b"fake MaleCNS4096 topology")
    return p.PrestigeConfig(
        dataset="data/FAKE", output_dir=str(tmp_path), core_path=str(core),
        old_best_path=str(tmp_path.parent / "old_best.pt"),
        seed=2401, device="cpu", temporal_hidden=8,
        stage_c_updates=2, batch_frames=5,
    )


class FakeCNS(nn.Module):
    def __init__(self, *, input_dim, key_count, core_path):
        super().__init__()
        self.hidden_dim = 4096
        self.input_dim = input_dim
        self.key_count = key_count
        self.sensory = nn.Linear(input_dim, 8)
        self.controller = nn.Linear(8, 8)

    def prepare_recurrent_runtime(self):
        pass

    def initial_state(self, device):
        return torch.zeros(1, device=device)

    def forward_step(self, x, state):
        return (self.controller(self.sensory(x)), torch.ones(8, device=x.device),
                torch.zeros((), device=x.device), state + 1)


def fake_backbone(monkeypatch):
    pkg = types.ModuleType("dmdod.connectome")
    mod = types.ModuleType("dmdod.connectome.human_visible_policy")
    mod.NKeyHumanVisibleControllerActorCritic = FakeCNS
    monkeypatch.setitem(sys.modules, "dmdod.connectome", pkg)
    monkeypatch.setitem(sys.modules, "dmdod.connectome.human_visible_policy", mod)


def manifest():
    return {
        "train": [{"anchor": i, "name": f"T{i}", "sha256": f"T{i}"}
                  for i in range(1, 21)],
        "validation": [{"anchor": i, "name": f"V{i}", "sha256": f"V{i}"}
                       for i in range(1, 21)],
    }


def create_pretrained_stage_a(c):
    folder = Path(c.output_dir) / "diagnostics" / "prestige1"
    folder.mkdir(parents=True)
    torch.manual_seed(c.seed)
    temporal = p.TemporalReadout(c.temporal_hidden, recurrent=True)
    partition = {"validation_used_for_selection": False,
                 "fit_anchor_ids": list(range(1, 16)),
                 "heldout_anchor_ids": list(range(16, 21))}
    metadata = {"config_sha256": c.signature(),
                "dataset_split": manifest(),
                "expected_old_best_sha256": p.BEST_SHA256}
    payload = {"format": p.VERSION, "stage": "A", "name": "gru",
               "initial_seed": c.seed, "metadata": metadata,
               "train_partition": partition, "state": temporal.state_dict()}
    path = folder / "gru_v2.pt"
    torch.save(payload, path)
    audit = {
        "format": p.VERSION, "stage": "A", "metadata": metadata,
        "train_partition": partition,
        "checkpoint_sha256": {"gru": p.sha256(path)},
        "gate": {"passed": False, "gate_data_role": "train_heldout",
                 "reasons": ["gru_near_event_mae_not_20pct_better_than_constant_and_single"]},
    }
    (folder / p.STAGE_A_REPORT).write_text(json.dumps(audit))
    return folder, audit


def temporal_sequence(role, idx):
    t = 9
    visible = torch.zeros(t, p.INPUT_DIM)
    visible[:, 0] = torch.arange(t) * .01
    labels = torch.zeros(t, 3)
    labels[:, 0] = .1
    labels[:, 1] = 1.
    return p.TemporalSequence(role, idx, f"{role}{idx}", visible,
                              torch.zeros(t, 8), labels, torch.ones(t, dtype=torch.bool))


def test_gru_to_grucell_weights_reproduce_all_timesteps(monkeypatch, tmp_path):
    fake_backbone(monkeypatch)
    c = config(tmp_path)
    torch.manual_seed(c.seed)
    readout = p.TemporalReadout(c.temporal_hidden, recurrent=True)
    torch.manual_seed(c.seed)
    student = p.PrestigeMotorStudent(c.core_path, c.temporal_hidden)
    bc.transfer_temporal_weights(readout, student)
    report = bc.transfer_parity(readout, student, seed=c.seed)
    assert report["passed"] and report["frames"] == 24
    assert report["max_abs"] <= report["atol"]
    assert torch.equal(readout.head.weight, student.timing_head.weight)
    assert torch.equal(readout.encoder.weight_ih_l0, student.temporal.weight_ih)


def test_stage_b_audit_does_not_override_failed_stage_a(monkeypatch, tmp_path):
    fake_backbone(monkeypatch)
    c = config(tmp_path)
    folder, original = create_pretrained_stage_a(c)
    result = bc.run_stage_b(c)
    assert result["parity"]["passed"]
    assert result["stage_a_original_gate"] == original["gate"]
    assert result["stage_a_gate_overridden"] is False
    assert result["no_motor_updates"] is True
    assert result["validation_used_for_selection"] is False
    assert (folder / bc.TRANSFER_REPORT).is_file()
    assert original["gate"]["passed"] is False
    with pytest.raises(FileExistsError):
        bc.run_stage_b(c)


def test_stage_b_rejects_tampered_pretraining(monkeypatch, tmp_path):
    fake_backbone(monkeypatch)
    c = config(tmp_path)
    folder, _ = create_pretrained_stage_a(c)
    with (folder / "gru_v2.pt").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        bc.run_stage_b(c)
    assert not (folder / bc.TRANSFER_REPORT).exists()


def test_stage_c_v2_matched_pretraining_no_validation_training(monkeypatch, tmp_path):
    fake_backbone(monkeypatch)
    c = config(tmp_path)
    _, old_stage = create_pretrained_stage_a(c)
    stage_b = bc.run_stage_b(c)
    captures = []
    data = {
        "train": [temporal_sequence("train", i) for i in range(1, 21)],
        "validation": [temporal_sequence("validation", i) for i in range(1, 21)],
    }
    named = [SimpleNamespace(chart_name=f"V{i}") for i in range(1, 21)]
    monkeypatch.setattr(p, "load_role_sequences",
                        lambda *args, **kwargs: (data, manifest(), {"validation": named}))
    fake_n_key = types.ModuleType("dmdod.training.n_key")
    fake_n_key.n_key_actuation_loss = lambda action, expert: (action - expert).square().mean()
    monkeypatch.setitem(sys.modules, "dmdod.training.n_key", fake_n_key)
    fake_real = types.ModuleType("dmdod.training.real_chart")

    def evaluate(model, validation, **kwargs):
        captures.append((kwargs["label"], [w.detach().clone() for w in model.parameters()]))
        assert len(validation) == 20
        assert kwargs["control_dt_s"] == .01
        assert kwargs["physics_dt_s"] == .001
        stats = SimpleNamespace(hits=2, misses=1, too_early_presses=0,
                                overloaded=False, x_accuracy_percent=85.,
                                perfect_rate=.5, mean_abs_error_ms=30.)
        return [(stats, 3)] * 20

    fake_real.evaluate_role_continuous = evaluate
    monkeypatch.setitem(sys.modules, "dmdod.training.real_chart", fake_real)
    result = bc.run_stage_c_v2(c)
    assert len(captures) == 2
    assert result["effective_updates_per_arm"] == 40
    assert result["training_chunks"] == 40
    assert result["full_train_pass"] is True
    assert result["identical_pretrained_initial_weights"] is True
    assert result["same_minibatch_schedule"] is True
    assert result["stage_a_original_gate"]["passed"] is False
    assert result["stage_a_gate_overridden"] is False
    assert result["pretrained_sha256"] == stage_b["pretrained_sha256"]
    assert result["validation_used_for_selection"] is False
    assert all(len(arm["loss_history"]) == 40 for arm in result["results"].values())
    assert all(arm["validation"]["safe"] == 20 for arm in result["results"].values())
    for arm in ("control", "prestige"):
        item = result["results"][arm]
        assert p.sha256(Path(item["checkpoint"])) == item["checkpoint_sha256"]
        cp = torch.load(item["checkpoint"], map_location="cpu", weights_only=True)
        assert cp["updates"] == 40 and cp["warm_start_old_best"] is False
        assert cp["pretrained_sha256"] == stage_b["pretrained_sha256"]
    assert old_stage["gate"]["passed"] is False
    with pytest.raises(FileExistsError):
        bc.run_stage_c_v2(c)


def test_stage_c_v2_refuses_stage_b_config_tampering(monkeypatch, tmp_path):
    fake_backbone(monkeypatch)
    c = config(tmp_path)
    create_pretrained_stage_a(c)
    bc.run_stage_b(c)
    file = Path(c.output_dir) / "diagnostics" / "prestige1" / bc.TRANSFER_REPORT
    report = json.loads(file.read_text())
    report["config_sha256"] = "0" * 64
    file.write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="provenance"):
        bc.run_stage_c_v2(c)
