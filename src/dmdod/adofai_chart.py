from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AdoFaiAction:
    floor: int
    event_type: str
    active: bool = True
    speed_type: str | None = None
    beats_per_minute: float | None = None
    bpm_multiplier: float | None = None
    planets: str | None = None
    angle_offset: float | None = None
    duration: float | None = None
    source_index: int = -1


@dataclass(frozen=True, slots=True)
class AdoFaiChart:
    source_path: str
    angles: tuple[float, ...]
    actions: tuple[AdoFaiAction, ...]
    initial_bpm: float = 100.0
    song_filename: str | None = None
    offset_ms: float = 0.0
    pitch_percent: float = 100.0
    countdown_ticks: int = 4
    separate_countdown_time: bool = False

    @property
    def floor_count(self) -> int:
        return len(self.angles) + 1

    def actions_at(self, floor: int) -> tuple[AdoFaiAction, ...]:
        return tuple(action for action in self.actions if action.floor == floor)


def load_adofai(path: str | Path) -> AdoFaiChart:
    source = Path(path)
    return parse_adofai_bytes(source.read_bytes(), source_path=str(source))


def parse_adofai_bytes(data: bytes, *, source_path: str = "<memory>") -> AdoFaiChart:
    if data.startswith(b"\xef\xbb\xbf"):
        text = data[3:].decode("utf-8")
    elif data.startswith(b"\xff\xfe"):
        text = data[2:].decode("utf-16-le")
    elif data.startswith(b"\xfe\xff"):
        text = data[2:].decode("utf-16-be")
    else:
        text = data.decode("utf-8")
    return parse_adofai_text(text, source_path=source_path)


def parse_adofai_text(text: str, *, source_path: str = "<memory>") -> AdoFaiChart:
    root = json.loads(_normalize_loose_json(text))
    if not isinstance(root, dict):
        raise ValueError("ADOFAI root must be a JSON object")

    angle_data = root.get("angleData")
    if not isinstance(angle_data, list):
        if "pathData" in root:
            raise ValueError("legacy pathData is not supported yet; angleData is required")
        raise ValueError("angleData is required")
    angles = tuple(_required_float(value, f"angleData[{index}]") for index, value in enumerate(angle_data))

    settings = root.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    initial_bpm = _optional_float(settings.get("bpm"), 100.0)
    if initial_bpm <= 0.0:
        initial_bpm = 100.0

    actions: list[AdoFaiAction] = []
    raw_actions = root.get("actions")
    if isinstance(raw_actions, list):
        for source_index, raw in enumerate(raw_actions):
            if not isinstance(raw, dict):
                continue
            floor = _optional_int(raw.get("floor"))
            if floor is None:
                continue
            actions.append(
                AdoFaiAction(
                    floor=floor,
                    event_type=_optional_str(raw.get("eventType")) or "<unknown>",
                    active=_optional_bool(raw.get("active"), True),
                    speed_type=_optional_str(raw.get("speedType")),
                    beats_per_minute=_optional_float_or_none(raw.get("beatsPerMinute")),
                    bpm_multiplier=_optional_float_or_none(raw.get("bpmMultiplier")),
                    planets=_optional_str(raw.get("planets")),
                    angle_offset=_optional_float_or_none(raw.get("angleOffset")),
                    duration=_optional_float_or_none(raw.get("duration")),
                    source_index=source_index,
                )
            )

    return AdoFaiChart(
        source_path=source_path,
        angles=angles,
        actions=tuple(actions),
        initial_bpm=initial_bpm,
        song_filename=_optional_str(settings.get("songFilename")),
        offset_ms=_optional_float(settings.get("offset", settings.get("songOffset")), 0.0),
        pitch_percent=_optional_float(settings.get("pitch"), 100.0),
        countdown_ticks=max(0, _optional_int(settings.get("countdownTicks")) or 0)
        if "countdownTicks" in settings
        else 4,
        separate_countdown_time=_optional_bool(settings.get("separateCountdownTime"), False),
    )


def _normalize_loose_json(text: str) -> str:
    """Remove comments and trailing commas without touching quoted strings."""

    out: list[str] = []
    i = 0
    in_string = False
    escape = False
    line_comment = False
    block_comment = False
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if ch in "\r\n":
                line_comment = False
                out.append(ch)
            i += 1
            continue
        if block_comment:
            if ch == "*" and nxt == "/":
                block_comment = False
                i += 2
            else:
                i += 1
            continue
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            line_comment = True
            i += 2
            continue
        if ch == "/" and nxt == "*":
            block_comment = True
            i += 2
            continue
        out.append(ch)
        i += 1

    text = "".join(out)
    out = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "]}":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _required_float(value: Any, label: str) -> float:
    parsed = _optional_float_or_none(value)
    if parsed is None:
        raise ValueError(f"{label} is not numeric: {value!r}")
    return parsed


def _optional_float(value: Any, default: float) -> float:
    parsed = _optional_float_or_none(value)
    return default if parsed is None else parsed


def _optional_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "enabled", "1"}:
        return True
    if text in {"false", "disabled", "0"}:
        return False
    return default
