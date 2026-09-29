from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_real_chart_v080 as trainer  # noqa: E402
from dmdod.multichart_dataset import discover_multichart_dataset  # noqa: E402


def _chart_bytes(bpm: float) -> bytes:
    payload = {
        "angleData": [0, 180, 90, 0],
        "settings": {
            "bpm": bpm,
            "pitch": 100,
            "countdownTicks": 0,
            "separateCountdownTime": False,
        },
        "actions": [],
    }
    return json.dumps(payload).encode("utf-8")


def _chart_zip(bpm: float, filename: str = "level.adofai") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(filename, _chart_bytes(bpm))
    return stream.getvalue()


def _backup_plus_named_chart_zip() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("backup.adofai", _chart_bytes(90.0))
        archive.writestr("the_chart.adofai", _chart_bytes(123.0))
    return stream.getvalue()


def test_directory_dataset_discovers_train_validation_final_zip_charts(tmp_path: Path):
    for role in ("Train", "Validation", "Final"):
        (tmp_path / role).mkdir()
    (tmp_path / "Train" / "A.zip").write_bytes(_chart_zip(120.0))
    (tmp_path / "Train" / "B.zip").write_bytes(_chart_zip(150.0, "main.adofai"))
    (tmp_path / "Validation" / "V.zip").write_bytes(_chart_zip(180.0))
    (tmp_path / "Final" / "F.zip").write_bytes(_chart_zip(210.0))

    dataset = discover_multichart_dataset(tmp_path)
    assert [item.name for item in dataset.train] == ["A", "B"]
    assert [item.name for item in dataset.validation] == ["V"]
    assert [item.name for item in dataset.final] == ["F"]
    assert len({item.content_sha256 for item in dataset.all_charts}) == 4
    assert all(Path(item.resolved_path).exists() for item in dataset.all_charts)


def test_bundle_zip_supports_nested_chart_zips(tmp_path: Path):
    bundle = tmp_path / "dataset.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("DMDOD/Train/A.zip", _chart_zip(120.0))
        archive.writestr("DMDOD/Validation/V.zip", _chart_zip(160.0))
        archive.writestr("DMDOD/Final/F.zip", _chart_zip(200.0))

    dataset = discover_multichart_dataset(bundle)
    assert [item.name for item in dataset.train] == ["A"]
    assert [item.name for item in dataset.validation] == ["V"]
    assert [item.name for item in dataset.final] == ["F"]


def test_chart_zip_prefers_unique_non_backup_when_no_main_or_level(tmp_path: Path):
    for role in ("Train", "Validation", "Final"):
        (tmp_path / role).mkdir()
    (tmp_path / "Train" / "named.zip").write_bytes(_backup_plus_named_chart_zip())
    (tmp_path / "Validation" / "V.zip").write_bytes(_chart_zip(160.0))
    (tmp_path / "Final" / "F.zip").write_bytes(_chart_zip(200.0))

    dataset = discover_multichart_dataset(tmp_path)
    selected = Path(dataset.train[0].resolved_path).read_text(encoding="utf-8")
    assert '"bpm": 123.0' in selected


def test_dataset_rejects_same_chart_in_multiple_roles(tmp_path: Path):
    for role in ("Train", "Validation", "Final"):
        (tmp_path / role).mkdir()
    same = _chart_zip(120.0)
    (tmp_path / "Train" / "A.zip").write_bytes(same)
    (tmp_path / "Validation" / "V.zip").write_bytes(same)
    (tmp_path / "Final" / "F.zip").write_bytes(_chart_zip(200.0))

    try:
        discover_multichart_dataset(tmp_path)
    except ValueError as exc:
        assert "same chart content" in str(exc)
    else:
        raise AssertionError("duplicate chart content across splits must be rejected")


def test_anchor_windows_cover_ends_and_deduplicate_short_charts():
    assert trainer._anchor_windows(120.0, 30.0, 2) == ((0.0, 30.0), (90.0, 120.0))
    assert trainer._anchor_windows(20.0, 30.0, 2) == ((0.0, 20.0),)


def test_validation_window_is_centered():
    assert trainer._validation_window(100.0, 30.0) == (35.0, 65.0)
    assert trainer._validation_window(20.0, 30.0) == (0.0, 20.0)


def test_v080_checkpoint_identity_is_separate_from_v070():
    assert trainer.TRAINER_VERSION == "0.8.0-multichart-hud"
    assert trainer.CHECKPOINT_FORMAT_VERSION == 15
    assert trainer.DEFAULT_CHECKPOINT.endswith("real_chart_v080_multichart.pt")
