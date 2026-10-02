from __future__ import annotations

"""Feed one real renderer PNG through the Level B visual policy.

This is an integration smoke test for the current ExtremeEditor B0 output.  It
intentionally accepts only ordinary 8-bit non-interlaced RGB/RGBA PNGs and uses
only the Python standard library for PNG decoding, so no image dependency is
required by DMDOD.
"""

import argparse
import struct
import zlib
from pathlib import Path

import torch

from dmdod.visual_observation import (
    DEFAULT_VISUAL_HEIGHT,
    DEFAULT_VISUAL_WIDTH,
    RgbFrame,
)
from dmdod.visual_policy import LevelBVisualPolicy, rgb_frame_to_tensor

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def load_png_rgb888(path: str | Path) -> RgbFrame:
    raw = Path(path).read_bytes()
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError("input is not a PNG file")

    offset = len(PNG_SIGNATURE)
    width = height = None
    color_type = None
    bit_depth = None
    interlace = None
    compressed = bytearray()

    while offset + 12 <= len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        if data_end + 4 > len(raw):
            raise ValueError("truncated PNG chunk")
        data = raw[data_start:data_end]
        offset = data_end + 4

        if chunk_type == b"IHDR":
            if length != 13:
                raise ValueError("invalid PNG IHDR")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            if compression != 0 or filtering != 0:
                raise ValueError("unsupported PNG compression/filter method")
        elif chunk_type == b"IDAT":
            compressed.extend(data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None:
        raise ValueError("PNG is missing IHDR")
    if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
        raise ValueError(
            "only 8-bit non-interlaced RGB/RGBA PNG is supported "
            f"(bit_depth={bit_depth}, color_type={color_type}, interlace={interlace})"
        )

    channels = 3 if color_type == 2 else 4
    stride = width * channels
    scanlines = zlib.decompress(bytes(compressed))
    expected = height * (stride + 1)
    if len(scanlines) != expected:
        raise ValueError(
            f"unexpected PNG scanline size: expected {expected}, got {len(scanlines)}"
        )

    decoded = bytearray(height * stride)
    previous = bytearray(stride)
    for y in range(height):
        row_start = y * (stride + 1)
        filter_type = scanlines[row_start]
        source = scanlines[row_start + 1 : row_start + 1 + stride]
        row = bytearray(stride)

        for x, value in enumerate(source):
            left = row[x - channels] if x >= channels else 0
            up = previous[x]
            up_left = previous[x - channels] if x >= channels else 0
            if filter_type == 0:
                reconstructed = value
            elif filter_type == 1:
                reconstructed = value + left
            elif filter_type == 2:
                reconstructed = value + up
            elif filter_type == 3:
                reconstructed = value + ((left + up) // 2)
            elif filter_type == 4:
                reconstructed = value + _paeth(left, up, up_left)
            else:
                raise ValueError(f"unsupported PNG filter type: {filter_type}")
            row[x] = reconstructed & 0xFF

        decoded[y * stride : (y + 1) * stride] = row
        previous = row

    if channels == 3:
        rgb = bytes(decoded)
    else:
        rgb = bytes(
            component
            for i in range(0, len(decoded), 4)
            for component in decoded[i : i + 3]
        )
    return RgbFrame(width=width, height=height, data=rgb)


def smoke_frame(path: str | Path, *, device: str = "cpu") -> dict[str, object]:
    frame = load_png_rgb888(path)
    if (frame.width, frame.height) != (DEFAULT_VISUAL_WIDTH, DEFAULT_VISUAL_HEIGHT):
        raise ValueError(
            f"expected {DEFAULT_VISUAL_WIDTH}x{DEFAULT_VISUAL_HEIGHT}, "
            f"got {frame.width}x{frame.height}"
        )

    torch_device = torch.device(device)
    policy = LevelBVisualPolicy().to(torch_device)
    policy.eval()
    frame_tensor = rgb_frame_to_tensor(frame, device=torch_device)
    proprioception = torch.zeros(6, dtype=torch.float32, device=torch_device)
    state = policy.initial_state(torch_device)

    with torch.no_grad():
        mean, std, value, next_state = policy.forward_step(
            frame_tensor,
            proprioception,
            state,
        )
        action = torch.tanh(mean)

    return {
        "width": frame.width,
        "height": frame.height,
        "tensor_shape": tuple(frame_tensor.shape),
        "action": tuple(float(v) for v in action.cpu()),
        "std": tuple(float(v) for v in std.cpu()),
        "value": float(value.cpu()),
        "state_shape": tuple(next_state.shape),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feed one ExtremeEditor headless PNG through DMDOD Level B policy."
    )
    parser.add_argument("png", help="320x180 RGB/RGBA PNG emitted by ExtremeEditor.Headless")
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = smoke_frame(args.png, device=args.device)
    print(
        "visual-smoke OK "
        f"frame={result['width']}x{result['height']} "
        f"tensor={result['tensor_shape']} state={result['state_shape']} "
        f"action={result['action']} std={result['std']} value={result['value']:.6f}"
    )


if __name__ == "__main__":
    main()
