from __future__ import annotations

from dataclasses import dataclass

from dmdod.motor.contact_16k import ContactAction

from scripts import diagnose_fingering_causal_failure_v02 as diagnostic


@dataclass
class _Event:
    event_index: int = 7
    episode_time_s: float = 1.0
    target_ordinals: tuple[int, ...] = (9,)
    chord_size: int = 1


def test_window_probe_ignores_frames_outside_target_window() -> None:
    probe = diagnostic.WindowProbe({7: 1.0}, radius_s=.080)
    probe("after_step", {"now_s": .90})
    assert probe.frames == []


def test_window_probe_records_decision_state(monkeypatch) -> None:
    monkeypatch.setattr(
        diagnostic, "physical_snapshot", lambda env: {"episode_ms": 950.0}
    )
    probe = diagnostic.WindowProbe({7: 1.0}, radius_s=.080)
    probe("decision", {
        "now_s": .95,
        "event": _Event(),
        "decision": {"chosen": "left_1", "extra_lead_ms": 4.0},
        "env": object(),
    })
    assert probe.decisions == [{
        "event_index": 7,
        "due_ms": 1000.0,
        "target_ordinals": [9],
        "chord_size": 1,
        "chosen": "left_1",
        "extra_lead_ms": 4.0,
        "state": {"episode_ms": 950.0},
    }]


def test_execute_probe_parameter_is_optional() -> None:
    # The signature guard protects all existing audit callers from needing a
    # diagnostic callback and makes the probe observational by construction.
    assert diagnostic.causal.execute.__kwdefaults__["diagnostic_probe"] is None
