from __future__ import annotations

import binascii
import struct
import sys
import zlib
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from smoke_visual_policy_png import load_png_rgb888, smoke_frame


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(kind)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def _write_rgb_png(path: Path, width: int = 320, height: int = 180) -> bytes:
    rows = bytearray()
    expected = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter: None
        for x in range(width):
            pixel = bytes(((x + y) & 0xFF, (2 * x) & 0xFF, (3 * y) & 0xFF))
            rows.extend(pixel)
            expected.extend(pixel)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(rows)))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)
    return bytes(expected)


def test_png_loader_recovers_tightly_packed_rgb888(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    expected = _write_rgb_png(path, width=7, height=3)

    frame = load_png_rgb888(path)

    assert frame.width == 7
    assert frame.height == 3
    assert frame.data == expected


def test_real_png_smoke_reaches_level_b_motor_head(tmp_path: Path) -> None:
    path = tmp_path / "ee-frame.png"
    _write_rgb_png(path)

    result = smoke_frame(path)

    assert result["width"] == 320
    assert result["height"] == 180
    assert result["tensor_shape"] == (3, 180, 320)
    assert result["state_shape"] == (128,)
    assert len(result["action"]) == 2
    assert len(result["std"]) == 2
