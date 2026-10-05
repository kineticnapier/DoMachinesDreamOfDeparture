from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v172_n_key_connectome_reject_early_stop_dagger as v172


def test_epoch_range_stops_before_repeated_epoch(capsys) -> None:
    v172._stop_before_next_epoch = False
    v172._last_started_epoch = 0

    epochs = v172._epoch_range(1, 5)
    assert next(epochs) == 1
    assert v172._last_started_epoch == 1

    v172._stop_before_next_epoch = True
    assert list(epochs) == []
    assert "EARLY STOP after rejected proposal" in capsys.readouterr().out


def test_checkpoint_payload_records_actual_completed_epoch(monkeypatch) -> None:
    def fake_payload(*args, **kwargs):
        return dict(kwargs)

    monkeypatch.setattr(v172, "_ORIGINAL_V166_CHECKPOINT_PAYLOAD", fake_payload)
    v172._stop_before_next_epoch = True
    v172._last_started_epoch = 3

    payload = v172._checkpoint_payload(
        completed_epoch=8,
        requested_epochs=8,
    )

    assert payload["completed_epoch"] == 3
    assert payload["requested_epochs"] == 8
    assert payload["dagger_reject_early_stop"] is True
    assert payload["dagger_reject_early_stop_epoch"] == 3
