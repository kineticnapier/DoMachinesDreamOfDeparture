from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v063 as trainer  # noqa: E402


def _result():
    stats = SimpleNamespace(
        hits=91,
        misses=9,
        targets=100,
        x_accuracy_percent=71.25,
        perfect_rate=0.54,
        too_early_presses=7,
        overloaded=False,
    )
    return trainer.v054.StudentEvalResult(stats, 98)


def _args(**overrides):
    values = dict(
        hidden=128,
        seed=3,
        cross_hand=False,
        control_dt=0.010,
        train_window=30.0,
        bootstrap_segments=6,
        round_epochs=12,
        lr=1e-4,
        chunk_steps=192,
        press_recovery_cap=8,
        rounds=24,
        bootstrap_epochs=96,
        keep_round_checkpoints=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_eval_payload_round_trip_preserves_validation_guard_fields():
    original = _result()
    restored = trainer._eval_from_payload(trainer._eval_to_payload(original))
    assert restored.stats.hits == 91
    assert restored.stats.misses == 9
    assert restored.stats.targets == 100
    assert restored.stats.x_accuracy_percent == 71.25
    assert restored.stats.perfect_rate == 0.54
    assert restored.stats.too_early_presses == 7
    assert restored.stats.overloaded is False
    assert restored.physical_keydowns == 98


def test_run_signature_allows_extending_total_rounds():
    train = trainer.v062.SegmentWindow(0.0, 90.0)
    validation = trainer.v062.SegmentWindow(90.0, 110.0)
    sight = trainer.v062.SegmentWindow(110.0, 130.0)
    first = trainer._run_signature(
        _args(rounds=12, bootstrap_epochs=56),
        chart_path="C:/chart/main.adofai",
        train_pool=train,
        validation_window=validation,
        sight_window=sight,
    )
    extended = trainer._run_signature(
        _args(rounds=24, bootstrap_epochs=96),
        chart_path="C:/chart/main.adofai",
        train_pool=train,
        validation_window=validation,
        sight_window=sight,
    )
    assert first == extended


def test_run_signature_rejects_training_semantics_change():
    train = trainer.v062.SegmentWindow(0.0, 90.0)
    validation = trainer.v062.SegmentWindow(90.0, 110.0)
    sight = trainer.v062.SegmentWindow(110.0, 130.0)
    saved = trainer._run_signature(
        _args(),
        chart_path="C:/chart/main.adofai",
        train_pool=train,
        validation_window=validation,
        sight_window=sight,
    )
    changed = trainer._run_signature(
        _args(round_epochs=10),
        chart_path="C:/chart/main.adofai",
        train_pool=train,
        validation_window=validation,
        sight_window=sight,
    )
    assert trainer._signature_mismatches(saved, changed) == ["round_epochs"]


def test_rng_state_resumes_next_training_window_exactly():
    rng = random.Random(12345)
    first = trainer.v062._sample_window(rng, 0.0, 90.0, 30.0)
    saved_state = rng.getstate()
    expected_next = trainer.v062._sample_window(rng, 0.0, 90.0, 30.0)

    resumed = random.Random()
    resumed.setstate(saved_state)
    actual_next = trainer.v062._sample_window(resumed, 0.0, 90.0, 30.0)
    assert first != expected_next
    assert actual_next == expected_next


def test_round_snapshot_path_keeps_extension():
    path = Path("checkpoints/schoolrun.pt")
    assert trainer._round_snapshot_path(path, 7) == Path("checkpoints/schoolrun.round007.pt")


def test_checkpoint_payload_records_round_rng_and_validation_floor():
    args = _args()
    result = _result()
    payload = trainer._checkpoint_payload(
        model_state={"x": trainer.torch.tensor([1.0])},
        args=args,
        chart_path="C:/chart/main.adofai",
        signature={"seed": 3},
        completed_round=7,
        rng_state=random.Random(9).getstate(),
        validation_reference=result,
        bootstrap_history=[{"epoch": 1}],
        round_history=[{"round": 7}],
        finalized=False,
    )
    assert payload["format_version"] == 11
    assert payload["completed_round"] == 7
    assert payload["requested_rounds"] == 24
    assert payload["validation_reference"]["hits"] == 91
    assert payload["round_history"] == [{"round": 7}]
    assert payload["finalized"] is False
    assert payload["sight_used_for_selection"] is False


def test_v063_checkpoint_format_and_name():
    assert trainer.CHECKPOINT_FORMAT_VERSION == 11
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v063_resumable.pt")
