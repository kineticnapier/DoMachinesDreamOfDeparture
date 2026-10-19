import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import audit_stock_setspeed_icon_capture_v01 as audit
from scripts import prepare_stock_setspeed_icon_capture_v01 as prepare


def synthetic_chart() -> bytes:
    return json.dumps({
        "angleData": [0, 150, 0],
        "settings": {"songFilename": "bool.ogg", "bpm": 60.01},
        "actions": [{
            "floor": 1, "eventType": "SetSpeed", "speedType": "Multiplier",
            "beatsPerMinute": 100, "bpmMultiplier": 8,
        }],
        "decorations": [],
    }, separators=(",", ":")).encode()


def interval(state: str, lower: float, upper: float, *, first=False) -> dict:
    return {
        "state": state, "arrival_lower_s": lower, "arrival_upper_s": upper,
        "first_occurrence_confirmed": first,
    }


def test_counterfactual_changes_only_floor1_multiplier():
    raw = synthetic_chart()
    assert prepare.mutate_floor1_multiplier(raw, 8) == raw
    changed = prepare.mutate_floor1_multiplier(raw, 1)
    left, right = json.loads(raw), json.loads(changed)
    prepare._assert_only_multiplier_diff(left, right)
    assert right["actions"][0]["bpmMultiplier"] == 1


def test_prepare_rejects_wrong_or_ambiguous_event():
    root = json.loads(synthetic_chart())
    root["actions"][0]["bpmMultiplier"] = 4
    with pytest.raises(ValueError, match="8x"):
        prepare.mutate_floor1_multiplier(json.dumps(root).encode(), 1)
    root["actions"].append(dict(root["actions"][0], bpmMultiplier=8))
    with pytest.raises(ValueError, match="exactly one"):
        prepare.mutate_floor1_multiplier(json.dumps(root).encode(), 1)


def test_verdict_uses_conservative_arrival_bounds():
    early = {
        "original-8x": interval("DOUBLE_RABBIT_VISIBLE", 3.0, 3.2),
        "counterfactual-1x": interval("TILE_VISIBLE_NO_SPEED_ICON", 3.1, 3.3),
    }
    assert audit.judge(early)[0] == "OBSERVABLE_BEFORE_DEADLINE"
    late = {
        "original-8x": interval("DOUBLE_RABBIT_VISIBLE", 4.0, 4.1, first=True),
        "counterfactual-1x": interval("TILE_VISIBLE_NO_SPEED_ICON", 4.0, 4.1, first=True),
    }
    assert audit.judge(late)[0] == "TOO_LATE"
    late["counterfactual-1x"]["first_occurrence_confirmed"] = False
    assert audit.judge(late)[0] == "UNVERIFIED"
    overlap = {
        "original-8x": interval("DOUBLE_RABBIT_VISIBLE", 3.9, 4.0),
        "counterfactual-1x": interval("TILE_VISIBLE_NO_SPEED_ICON", 3.9, 4.0),
    }
    assert audit.judge(overlap)[0] == "UNVERIFIED"


def test_unknown_or_occluded_and_invisible_tile_cannot_prove_one_x():
    rows = {
        "original-8x": interval("DOUBLE_RABBIT_VISIBLE", 2.0, 2.1),
        "counterfactual-1x": interval("UNKNOWN_OR_OCCLUDED", 2.0, 2.1),
    }
    assert audit.judge(rows)[0] == "UNVERIFIED"


def test_duplicate_world_entry_is_rejected(tmp_path: Path):
    manifest = prepare.capture_manifest_template(tmp_path)
    manifest["captures"].append(dict(manifest["captures"][0]))
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one entry"):
        audit.audit(path)


def test_frame_interval_includes_sync_latency_and_half_frame():
    capture = {
        "sync": {"anchor_video_s": 10.0, "anchor_episode_s": 0.0, "uncertainty_s": .002},
        "capture_latency": {"min_s": .003, "max_s": .007},
    }
    row = audit.annotation_interval(
        {"video_time_s": 13.0, "state": "DOUBLE_RABBIT_VISIBLE"}, capture, 120.0
    )
    assert row["episode_center_s"] == pytest.approx(3.0)
    assert row["arrival_lower_s"] == pytest.approx(3.0 - .002 - 1 / 240 + .003)
    assert row["arrival_upper_s"] == pytest.approx(3.0 + .002 + 1 / 240 + .007)


def test_pixel_difference_is_descriptive_and_exact():
    result = audit.pixel_difference(bytes([0, 10, 255]), bytes([16, 30, 200]))
    assert result["mean_absolute_rgb_difference"] == pytest.approx((16 + 20 + 55) / 3)
    assert result["max_absolute_rgb_difference"] == 55
    assert result["fraction_rgb_channels_over_16"] == pytest.approx(2 / 3)


def test_missing_recordings_remain_unverified(tmp_path: Path):
    manifest = prepare.capture_manifest_template(tmp_path)
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit.audit(path)
    assert result["verdict"] == "UNVERIFIED"
    assert "original-8x: video_path" in result["missing_or_invalid_evidence"]
    assert "counterfactual-1x: video_path" in result["missing_or_invalid_evidence"]


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="ffmpeg/ffprobe unavailable")
def test_ffmpeg_probe_crop_alignment_and_pixel_difference(tmp_path: Path):
    for name, color in (("eight.mp4", "white"), ("one.mp4", "black")):
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
            f"color=c={color}:s=64x64:r=120:d=4.2", "-c:v", "mpeg4", "-q:v", "2",
            str(tmp_path / name),
        ], check=True)
    manifest = prepare.capture_manifest_template(tmp_path)
    manifest["crop_xywh"] = [0, 0, 64, 64]
    for capture, video, state in zip(
        manifest["captures"], ("eight.mp4", "one.mp4"),
        ("DOUBLE_RABBIT_VISIBLE", "TILE_VISIBLE_NO_SPEED_ICON"), strict=True,
    ):
        capture["video_path"] = video
        capture["recording"] = {
            "recording_utc": "2026-10-19T00:00:00Z",
            "game_build": "synthetic",
            "capture_software": "ffmpeg",
            "capture_mode": "synthetic",
            "reported_fps": 120.0,
            "resolution": [64, 64],
            "mods_enabled": [],
            "camera_and_display_conditions": "fixed",
        }
        capture["sync"].update(anchor_video_s=0.0, uncertainty_s=0.0)
        capture["capture_latency"] = {"min_s": 0.0, "max_s": 0.0, "basis": "synthetic"}
        capture["annotations"] = [{"video_time_s": .1, "state": state}]
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = audit.audit(path)
    assert result["verdict"] == "OBSERVABLE_BEFORE_DEADLINE"
    pixels = result["pixel_difference_analysis"]
    assert pixels["performed"]
    assert pixels["paired_frames"] > 400
    assert pixels["top_differences"][0]["mean_absolute_rgb_difference"] > 200

    # A malformed crop or inaccurate video metadata must never produce
    # OBSERVABLE_BEFORE_DEADLINE, even with valid-looking annotations.
    manifest["crop_xywh"] = [0, 0, 65, 64]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    bad_crop = audit.audit(path)
    assert bad_crop["verdict"] == "UNVERIFIED"
    assert any("crop_xywh" in x for x in bad_crop["missing_or_invalid_evidence"])

    manifest["crop_xywh"] = [0, 0, 64, 64]
    manifest["captures"][0]["recording"]["resolution"] = [32, 64]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong_resolution = audit.audit(path)
    assert wrong_resolution["verdict"] == "UNVERIFIED"
    assert any("declared resolution" in x
               for x in wrong_resolution["missing_or_invalid_evidence"])

    manifest["captures"][0]["recording"]["resolution"] = [64, 64]
    manifest["captures"][0]["sync"]["anchor_episode_s"] = 1.0
    path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong_anchor = audit.audit(path)
    assert wrong_anchor["verdict"] == "UNVERIFIED"
    assert any("episode 0" in x
               for x in wrong_anchor["missing_or_invalid_evidence"])
