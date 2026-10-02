from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dmdod.extremeeditor_ipc import ExtremeEditorPersistentRenderer


_FAKE_SERVER = r'''
import argparse
import struct
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--stdio-rgb", action="store_true")
parser.add_argument("--chart", required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
args = parser.parse_args()

stdin = sys.stdin.buffer
stdout = sys.stdout.buffer
stdout.write(b"EERGB01\n")
stdout.write(struct.pack("<II", args.width, args.height))
stdout.flush()

count = 0
size = args.width * args.height * 3
while True:
    opcode = stdin.read(1)
    if not opcode:
        break
    code = opcode[0]
    if code == 0x01:
        raw = stdin.read(16)
        if len(raw) != 16:
            break
        scene_time, visual_time = struct.unpack("<dd", raw)
        count += 1
        pixel = bytes((
            count & 0xFF,
            int(round(scene_time * 10.0)) & 0xFF,
            int(round(visual_time * 10.0)) & 0xFF,
        ))
        frame = (pixel * ((size + 2) // 3))[:size]
        stdout.write(struct.pack("<BI", 0, len(frame)))
        stdout.write(frame)
        stdout.flush()
    elif code == 0x02:
        count = 0
        stdout.write(struct.pack("<BI", 0, 0))
        stdout.flush()
    elif code == 0x03:
        break
    else:
        message = f"unknown opcode {code}".encode("utf-8")
        stdout.write(struct.pack("<BI", 1, len(message)))
        stdout.write(message)
        stdout.flush()
'''


def _server(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_ee_server.py"
    script.write_text(_FAKE_SERVER, encoding="utf-8")
    return [sys.executable, str(script)]


def test_persistent_renderer_reuses_one_process_for_multiple_frames(tmp_path: Path) -> None:
    with ExtremeEditorPersistentRenderer(
        _server(tmp_path),
        tmp_path / "level.adofai",
        width=2,
        height=1,
    ) as renderer:
        pid = renderer.process_id
        first = renderer.render_explicit(1.0, 2.0)
        second = renderer.render_explicit(1.0, 2.0)

        assert renderer.process_id == pid
        assert first.width == second.width == 2
        assert first.height == second.height == 1
        assert first.data[:3] == bytes((1, 10, 20))
        assert second.data[:3] == bytes((2, 10, 20))


def test_level_b_render_uses_same_scene_and_visual_time(tmp_path: Path) -> None:
    with ExtremeEditorPersistentRenderer(
        _server(tmp_path),
        tmp_path / "level.adofai",
        width=2,
        height=1,
    ) as renderer:
        frame = renderer.render(3.0, width=2, height=1)

    assert frame.data[:3] == bytes((1, 30, 30))


def test_reset_keeps_process_alive_and_resets_renderer_session(tmp_path: Path) -> None:
    with ExtremeEditorPersistentRenderer(
        _server(tmp_path),
        tmp_path / "level.adofai",
        width=2,
        height=1,
    ) as renderer:
        pid = renderer.process_id
        first = renderer.render(0.0, width=2, height=1)
        renderer.reset()
        after_reset = renderer.render(0.0, width=2, height=1)

        assert renderer.process_id == pid
        assert first.data[0] == 1
        assert after_reset.data[0] == 1


def test_persistent_renderer_rejects_dimension_changes(tmp_path: Path) -> None:
    with ExtremeEditorPersistentRenderer(
        _server(tmp_path),
        tmp_path / "level.adofai",
        width=2,
        height=1,
    ) as renderer:
        with pytest.raises(ValueError, match="fixed at 2x1"):
            renderer.render(0.0, width=320, height=180)


def test_persistent_renderer_rejects_non_finite_times(tmp_path: Path) -> None:
    with ExtremeEditorPersistentRenderer(
        _server(tmp_path),
        tmp_path / "level.adofai",
        width=2,
        height=1,
    ) as renderer:
        with pytest.raises(ValueError, match="finite"):
            renderer.render_explicit(float("nan"), 0.0)
