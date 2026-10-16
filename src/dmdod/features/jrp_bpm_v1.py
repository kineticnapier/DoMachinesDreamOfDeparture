from __future__ import annotations

"""Versioned JipperResourcePack normal-overlay BPM rendering contract.

Ported from Jongye0l/JipperResourcePack main, commit
861a1ff6d12ef1c785ca16fb980a7e9292cfee4f:
  OverlayContents/Overlay.cs (UpdateBpm, lines 614-638).

This is a *renderer-side* adapter. The chart timestamps and floor IDs must
NEVER be given to a Timing Student. Only the printed HUD scalars from this
adapter may cross into the nonprivileged observation stream.

Jongyeol mode's CheckPseudo can aggregate several future floors and cache
its display; this contract explicitly does not emulate that mode.
"""

from dataclasses import dataclass
from math import isfinite
from struct import pack, unpack


JRP_SOURCE_REVISION = "861a1ff6d12ef1c785ca16fb980a7e9292cfee4f"
JRP_OBSERVATION_VERSION = "jrp-normal-bpm-overlay-v1"
DEFAULT_DECIMAL_PLACES = 2


@dataclass(frozen=True, slots=True)
class JrpBpmReading:
    """The three numbers printed by regular JRP when UpdateBpm fires.

    - tbpm = conductor.bpm * pitch * planet_speed
    - cbpm = pitch * 60 / (nextfloor.entryTime - currfloor.entryTime)
    - kps = cbpm / 60
    Values are before JRP's configurable display rounding.
    """

    tbpm: float
    cbpm: float | None
    kps: float | None
    precision: int = DEFAULT_DECIMAL_PLACES

    def __post_init__(self) -> None:
        if not isfinite(self.tbpm) or self.tbpm <= 0:
            raise ValueError("JRP TBPM must be finite and positive")
        if self.cbpm is not None and (not isfinite(self.cbpm) or self.cbpm <= 0):
            raise ValueError("JRP CBPM must be finite and positive if available")
        if self.kps is not None and (not isfinite(self.kps) or self.kps <= 0):
            raise ValueError("JRP KPS must be finite and positive if available")
        if type(self.precision) is not int or not 0 <= self.precision <= 4:
            raise ValueError("JRP decimal places must be 0..4")

    @property
    def printed(self) -> tuple[str, str, str]:
        """Simplified numeric formatting, not a pixel-accurate UI replica."""
        # Overlay.WriteRounded multiplies by 10^decimals, uses Math.Round
        # (ties-to-even), then trims *all* trailing fractional zeroes.
        scale = 10 ** self.precision
        def show(value: float | None) -> str:
            if value is None:
                return "unavailable"
            scaled = round(value * scale)
            sign = "-" if scaled < 0 else ""
            magnitude = abs(scaled)
            whole, fraction = divmod(magnitude, scale)
            if not fraction:
                return f"{sign}{whole}"
            tail = f"{fraction:0{self.precision}d}".rstrip("0")
            return f"{sign}{whole}.{tail}"
        return (show(self.tbpm), show(self.cbpm), show(self.kps))

    @property
    def visible_numeric(self) -> tuple[float, float | None, float | None]:
        """Numbers a Student can read from text (raw internal floats excluded)."""
        return tuple(float(x) if x != "unavailable" else None for x in self.printed)


def jrp_normal_bpm_for_floor(segment, floor_index: int, *, decimals: int = 2) -> JrpBpmReading:
    """Emulate JRP normal (not pseudo) BPM numbers for compiled chart floors.

    This is privileged rendering code, not a predictor: do not expose segment,
    floor_index, or true timestamps to Student modules.

    DMDOD's compiled floor.bpm is the effective current speed. This matches
    the JRP TBPM expression when its GetPlanetSpeed returns the same effective
    speed; this physical/game-version parity needs independent verification.
    """
    if type(floor_index) is not int or not 0 <= floor_index < len(segment.chart.floors):
        raise ValueError("floor index outside compiled chart")
    floors = segment.chart.floors
    floor = floors[floor_index]
    pitch = float(segment.pitch_ratio)
    # In the JRP C# expression these BPM results are cast to float.
    def as_float32(number: float) -> float:
        return unpack("<f", pack("<f", number))[0]

    tbpm = as_float32(float(floor.bpm) * pitch)
    next_floor = floors[floor_index + 1] if floor_index + 1 < len(floors) else None
    if next_floor is None:
        cbpm = tbpm  # JRP: else bpm
    else:
        delta_chart_s = float(next_floor.target_time_s - floor.target_time_s)
        if not isfinite(delta_chart_s) or delta_chart_s <= 0:
            # This happens e.g. some midspin/zero-time transitions. JRP's
            # division yields an infinite value: cannot encode it as finite
            # 263D BPM. Mark unsupported rather than fabricating finite BPM.
            return JrpBpmReading(tbpm=tbpm, cbpm=None, kps=None, precision=decimals)
        cbpm = as_float32(60.0 * pitch / delta_chart_s)
    return JrpBpmReading(tbpm=tbpm, cbpm=cbpm,
                         kps=as_float32(cbpm / 60.), precision=decimals)


class JrpNormalBpmDisplay:
    """Cache the visible display exactly at simulated JRP update hooks.

    JRP calls Overlay.UpdateBpm on Show and Bpm.OnHit; a floor transition
    *alone* does not trigger a text refresh in this port. Caller decides
    when the public hook fired and may not use secret evaluator events.
    """

    def __init__(self, segment, *, decimals: int = DEFAULT_DECIMAL_PLACES):
        if type(decimals) is not int or not 0 <= decimals <= 4:
            raise ValueError("decimals must be 0..4")
        self._segment = segment
        self._decimals = decimals
        self._reading: JrpBpmReading | None = None

    @property
    def visible(self) -> JrpBpmReading | None:
        return self._reading

    def reset(self) -> None:
        self._reading = None

    def on_show(self, current_floor_index: int) -> JrpBpmReading:
        self._reading = jrp_normal_bpm_for_floor(
            self._segment, current_floor_index, decimals=self._decimals)
        return self._reading

    def on_hit_hook(self, current_floor_index: int) -> JrpBpmReading:
        return self.on_show(current_floor_index)
