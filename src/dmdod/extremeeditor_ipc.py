from __future__ import annotations

"""Persistent ExtremeEditor renderer transport for Level B visual input.

The current one-shot ``ExtremeEditor.Headless`` PNG command is useful for smoke
tests but cannot be used at 60 Hz during training.  This module defines the
DMDOD side of a tiny binary stdio protocol so one renderer process can stay
alive for an entire chart/session.

Expected ExtremeEditor invocation::

    ExtremeEditor.Headless.exe \
        --stdio-rgb \
        --chart level.adofai \
        --width 320 \
        --height 180

Protocol v1 (little endian):

Server -> client startup::

    8 bytes  magic = b"EERGB01\\n"
    u32      width
    u32      height

Client requests::

    0x01 + f64 scene_time + f64 visual_time   render
    0x02                                      reset
    0x03                                      quit (no response required)

Server responses to render/reset::

    u8 status + u32 payload_size + payload

``status=0`` means success.  A render payload is tightly packed RGB888;
reset has an empty payload.  ``status=1`` carries a UTF-8 error message.

Only RGB crosses into the Level B policy path.  The chart path and requested
render times are transport/control metadata and are never neural-network input.
"""

import math
import struct
import subprocess
from pathlib import Path
from typing import BinaryIO, Sequence

from .visual_observation import RgbFrame


EXTREMEEDITOR_IPC_VERSION = "ee-stdio-rgb-v1"
_PROTOCOL_MAGIC = b"EERGB01\n"
_HANDSHAKE = struct.Struct("<II")
_RENDER_PAYLOAD = struct.Struct("<dd")
_RESPONSE_HEADER = struct.Struct("<BI")

_OP_RENDER = 0x01
_OP_RESET = 0x02
_OP_QUIT = 0x03
_STATUS_OK = 0x00
_STATUS_ERROR = 0x01


class ExtremeEditorProtocolError(RuntimeError):
    """Raised when the persistent renderer violates the binary protocol."""


class ExtremeEditorPersistentRenderer:
    """One persistent ExtremeEditor process implementing ``ChartFrameRenderer``.

    ``command_prefix`` is normally ``[r"...\\ExtremeEditor.Headless.exe"]``.
    Tests may use another executable plus helper-script arguments.  The class
    appends the stdio-server/chart/size arguments itself.
    """

    def __init__(
        self,
        command_prefix: Sequence[str | Path],
        chart_path: str | Path,
        *,
        width: int = 320,
        height: int = 180,
    ) -> None:
        if not command_prefix:
            raise ValueError("command_prefix must not be empty")
        if width <= 0 or height <= 0:
            raise ValueError("renderer dimensions must be positive")

        self.width = int(width)
        self.height = int(height)
        self.chart_path = str(Path(chart_path))
        command = [
            *(str(value) for value in command_prefix),
            "--stdio-rgb",
            "--chart",
            self.chart_path,
            "--width",
            str(self.width),
            "--height",
            str(self.height),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self._process.stdin is None or self._process.stdout is None:
            self.close()
            raise RuntimeError("failed to open ExtremeEditor stdio pipes")
        self._stdin: BinaryIO = self._process.stdin
        self._stdout: BinaryIO = self._process.stdout
        self._closed = False

        try:
            magic = self._read_exact(len(_PROTOCOL_MAGIC))
            if magic != _PROTOCOL_MAGIC:
                raise ExtremeEditorProtocolError(
                    f"renderer protocol magic mismatch: expected {_PROTOCOL_MAGIC!r}, got {magic!r}"
                )
            server_width, server_height = _HANDSHAKE.unpack(
                self._read_exact(_HANDSHAKE.size)
            )
            if (server_width, server_height) != (self.width, self.height):
                raise ExtremeEditorProtocolError(
                    "renderer size mismatch: "
                    f"requested {self.width}x{self.height}, "
                    f"server reported {server_width}x{server_height}"
                )
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "ExtremeEditorPersistentRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def process_id(self) -> int:
        """Renderer PID for diagnostics only; never a policy input."""

        return int(self._process.pid)

    def render(self, time_s: float, *, width: int, height: int) -> RgbFrame:
        """Render deterministic B0 pixels with ``sceneTime == visualTime``."""

        if (int(width), int(height)) != (self.width, self.height):
            raise ValueError(
                f"persistent renderer is fixed at {self.width}x{self.height}, "
                f"got request {width}x{height}"
            )
        return self.render_explicit(time_s, time_s)

    def render_explicit(self, scene_time_s: float, visual_time_s: float) -> RgbFrame:
        """Render with separate scene/visual clocks for B1 golden comparisons."""

        self._ensure_open()
        scene_time_s = float(scene_time_s)
        visual_time_s = float(visual_time_s)
        if not math.isfinite(scene_time_s) or not math.isfinite(visual_time_s):
            raise ValueError("render times must be finite")

        request = bytes((_OP_RENDER,)) + _RENDER_PAYLOAD.pack(
            scene_time_s, visual_time_s
        )
        self._write(request)
        payload = self._read_response()
        expected = self.width * self.height * 3
        if len(payload) != expected:
            raise ExtremeEditorProtocolError(
                f"RGB payload size mismatch: expected {expected}, got {len(payload)}"
            )
        return RgbFrame(width=self.width, height=self.height, data=payload)

    def reset(self) -> None:
        """Reset renderer session state without re-parsing/restarting the process."""

        self._ensure_open()
        self._write(bytes((_OP_RESET,)))
        payload = self._read_response()
        if payload:
            raise ExtremeEditorProtocolError(
                f"reset response must be empty, got {len(payload)} bytes"
            )

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        process = self._process
        try:
            if process.poll() is None:
                try:
                    self._stdin.write(bytes((_OP_QUIT,)))
                    self._stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
                try:
                    self._stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        finally:
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ExtremeEditor renderer is closed")
        exit_code = self._process.poll()
        if exit_code is not None:
            detail = self._stderr_text()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"ExtremeEditor renderer exited with code {exit_code}{suffix}"
            )

    def _write(self, data: bytes) -> None:
        self._ensure_open()
        try:
            self._stdin.write(data)
            self._stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("ExtremeEditor renderer pipe write failed") from exc

    def _read_response(self) -> bytes:
        status, payload_size = _RESPONSE_HEADER.unpack(
            self._read_exact(_RESPONSE_HEADER.size)
        )
        payload = self._read_exact(payload_size)
        if status == _STATUS_OK:
            return payload
        if status == _STATUS_ERROR:
            message = payload.decode("utf-8", errors="replace")
            raise RuntimeError(f"ExtremeEditor renderer error: {message}")
        raise ExtremeEditorProtocolError(f"unknown renderer response status: {status}")

    def _read_exact(self, size: int) -> bytes:
        if size < 0:
            raise ValueError("read size must be non-negative")
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self._stdout.read(size - len(chunks))
            if not chunk:
                exit_code = self._process.poll()
                detail = self._stderr_text() if exit_code is not None else ""
                suffix = f" stderr={detail!r}" if detail else ""
                raise ExtremeEditorProtocolError(
                    f"unexpected EOF from renderer after {len(chunks)}/{size} bytes{suffix}"
                )
            chunks.extend(chunk)
        return bytes(chunks)

    def _stderr_text(self) -> str:
        stream = self._process.stderr
        if stream is None or self._process.poll() is None:
            return ""
        try:
            return stream.read().decode("utf-8", errors="replace").strip()
        except (OSError, ValueError):
            return ""
