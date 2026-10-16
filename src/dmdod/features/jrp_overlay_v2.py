from __future__ import annotations

"""Read-only JRP overlay renderer: event-driven display and Jongyeol pseudo BPM.

Source: Jongye0l/JipperResourcePack commit
861a1ff6d12ef1c785ca16fb980a7e9292cfee4f,
Overlay.cs UpdateBpm + WriteRounded, JOverlay.cs UpdateBpm/CheckPseudo/Show.

PRIVILEGED renderer adapter, not a Timing estimator. Only its *visible_text*
may enter Student observations. Chart IDs, exact times and floor state MUST
remain on the rendering side. Real game scrFloor.angleLength/speed parity is
NOT established by a generated DMDOD chart projection.
"""

from dataclasses import dataclass
from math import isfinite, pi
from struct import pack, unpack
from typing import Sequence

from .jrp_bpm_v1 import JrpBpmReading, JRP_SOURCE_REVISION

JRP_OVERLAY_FEATURE_VERSION = "jrp-overlay-normal-jongyeol-v2"
SOURCE_90_RAD = 1.57079642638564
SOURCE_90_TOL = 1e-14


def _f32(x: float) -> float:
    return unpack('<f', pack('<f', x))[0]


@dataclass(frozen=True, slots=True)
class JrpFloorState:
    """Game-side scrFloor view. Values come from render/physics, NOT Student."""
    seq_id: int
    entry_time_s: float  # unpitched chart seconds
    angle_length_rad: float  # game scrFloor.angleLength (not visible from text)
    speed: float  # game scrFloor.speed multiplier
    mid_spin: bool = False


@dataclass(frozen=True, slots=True)
class JrpDisplayedHud:
    """HUD text plus metadata that the *renderer* can inspect.

    The only policy-safe export is visible_text; keep source_floor,
    pseudo_count and time out of the Student input.
    """
    visible_text: tuple[str, str, str]
    source_floor: int
    pseudo: bool
    pseudo_count: int
    displayed_hook: int

    @property
    def visible_numeric(self) -> tuple[float, float, float]:
        return tuple(float(x) for x in self.visible_text)


@dataclass(frozen=True, slots=True)
class _PseudoOutcome:
    cbpm: float | None
    count: int = 0
    suppress_through: int = -1


def _check90(angle: float) -> bool:
    return abs(angle - SOURCE_90_RAD) < SOURCE_90_TOL


def _pseudo(
    floors: Sequence[JrpFloorState], index: int, bpm: float, pitch: float,
    *, is_game_scene: bool,
) -> _PseudoOutcome:
    """Port of JOverlay.CheckPseudo, preserving the branch order and cache."""
    if bpm < 200 or not is_game_scene:
        return _PseudoOutcome(None)
    all_angle = 0.0
    max_angle = .5236 if bpm < 400 else 1.0472
    cur = floors[index]
    pos = index
    mid_spins = 0
    while pos + 1 < len(floors):
        floor = floors[pos]
        if floor.mid_spin:
            pos += 1
            mid_spins += 1
            continue
        angle = floor.angle_length_rad
        all_angle += angle
        if all_angle > max_angle and (
            bpm < 600 or all_angle - angle > 1e-14 or not _check90(angle)
        ):
            prior_angle = floors[pos - 1].angle_length_rad if pos else 0.0
            if angle < max_angle and abs(angle - prior_angle) < 1e-14:
                speed = floor.speed
                # Source's do/while advances first, then checks same angle
                # and same speed; floor may be null at end of chart.
                while True:
                    pos += 1
                    if pos >= len(floors):
                        # Can't safely assign a null floor's seqID.
                        return _PseudoOutcome(None)
                    nxt = floors[pos]
                    prev = floors[pos - 1]
                    if not (abs(nxt.angle_length_rad - prev.angle_length_rad) < 1e-14
                            and abs(speed - nxt.speed) < 1e-14):
                        break
                return _PseudoOutcome(None, suppress_through=floors[pos].seq_id - 1)
            break
        pos += 1

    # Exact 90-degree exception, as used in JRP source. Retain the source's
    # unusual backward counter structure, but explicitly handle null ends.
    if _check90(cur.angle_length_rad):
        count2 = 0
        pos2 = index + (1 if cur.mid_spin else 0)
        while count2 < 3 and 0 <= pos2 < len(floors) and _check90(floors[pos2].angle_length_rad):
            count2 += 1
            pos2 += 1
            if pos2 < len(floors) and floors[pos2].mid_spin:
                pos2 += 1
        if count2 < 3:
            pos2 = index - (1 if cur.mid_spin else 0)
            # C# code uses while(count2-- < 3) with count2++ inside. This
            # effectively checks adjacent previous 90-degree floors.
            while count2 < 3 and pos2 >= 0 and _check90(floors[pos2].angle_length_rad):
                count2 += 1
                pos2 -= 1
                if pos2 >= 0 and floors[pos2].mid_spin:
                    pos2 -= 1
        if count2 >= 3:
            return _PseudoOutcome(None)

    count = floors[pos].seq_id - cur.seq_id + 1 - mid_spins
    if count <= 1:
        return _PseudoOutcome(None)
    if pos + 1 < len(floors):
        duration = floors[pos + 1].entry_time_s - cur.entry_time_s
    else:
        duration = floors[pos].entry_time_s - cur.entry_time_s + 60.0 / bpm
    if duration <= 0 or not isfinite(duration):
        return _PseudoOutcome(None)
    # C# performs (float) of division before multiplying song.pitch.
    cbpm = _f32(_f32(60.0 / duration) * pitch)
    return _PseudoOutcome(cbpm, count=count, suppress_through=floors[pos].seq_id)


class JrpOverlayRenderer:
    """Event-driven *text* display. A floor transition alone never refreshes.

    The JRP patch for Bpm.OnHit is a gameplay hook. The caller must provide
    explicit OnHit hook occurrence; reading time/clock and guessing the next
    floor is not a legitimate display update. on_show resets pseudo cache.
    """

    def __init__(self, floors: Sequence[JrpFloorState], *, base_bpm: float,
                 pitch: float = 1.0, decimals: int = 2,
                 mode: str = 'normal', check_pseudo: bool = True,
                 is_game_scene: bool = True):
        self._floors = tuple(floors)
        if not self._floors or any(x.seq_id != i for i,x in enumerate(self._floors)):
            raise ValueError('floor sequence must start at zero and be contiguous')
        if base_bpm <= 0 or pitch <= 0 or not isfinite(base_bpm*pitch):
            raise ValueError('invalid base BPM or pitch')
        if mode not in ('normal', 'jongyeol') or type(decimals) is not int or decimals not in range(5):
            raise ValueError('invalid JRP mode or decimal places')
        self._base_bpm = base_bpm
        self._pitch = pitch
        self._decimals = decimals
        self._mode = mode
        self._check_pseudo = check_pseudo
        self._is_game_scene = is_game_scene
        self._pseudo_floor = -1
        self._last_display: JrpDisplayedHud | None = None
        self._hooks = 0

    @property
    def visible(self) -> tuple[str, str, str] | None:
        """Policy-safe export: text only (no floor/time/hidden pseudo count)."""
        return None if self._last_display is None else self._last_display.visible_text

    @property
    def diagnostic(self) -> JrpDisplayedHud | None:
        """Renderer/evaluator only. Do not forward into Student."""
        return self._last_display

    def on_show(self, floor_index: int) -> tuple[str, str, str]:
        self._pseudo_floor = -1  # JOverlay.Show explicitly resets this.
        self._last_display = None
        return self._update(floor_index)

    def on_hit_hook(self, floor_index: int) -> tuple[str, str, str] | None:
        """Called only when the game's *actual* OnHit patch fires."""
        return self._update(floor_index)

    def on_floor_advance(self, floor_index: int) -> tuple[str, str, str] | None:
        """Floor motion alone does not call JRP UpdateBpm."""
        self._get(floor_index)
        return self.visible

    def reset(self) -> None:
        self._pseudo_floor = -1
        self._last_display = None
        self._hooks = 0

    def _get(self, index: int) -> JrpFloorState:
        if type(index) is not int or not 0 <= index < len(self._floors):
            raise ValueError('floor outside renderer chart')
        return self._floors[index]

    def _update(self, index: int) -> tuple[str, str, str] | None:
        floor = self._get(index)
        self._hooks += 1
        if self._mode == 'jongyeol' and floor.seq_id <= self._pseudo_floor:
            return self.visible
        bpm = _f32(self._base_bpm * floor.speed * self._pitch)
        outcome = _PseudoOutcome(None)
        if self._mode == 'jongyeol' and self._check_pseudo:
            outcome = _pseudo(self._floors, index, bpm, self._pitch,
                              is_game_scene=self._is_game_scene)
            self._pseudo_floor = outcome.suppress_through if outcome.suppress_through >= 0 else self._pseudo_floor
        if outcome.cbpm is not None:
            cbpm = outcome.cbpm
            kps = _f32(_f32(cbpm / 60.0) * outcome.count)
            pseudo = True
        else:
            if index + 1 < len(self._floors):
                duration = self._floors[index + 1].entry_time_s - floor.entry_time_s
                if duration <= 0:
                    # Original C# would print Infinity; the numeric HUD
                    # cannot faithfully store it in a finite floating codec.
                    # Keep the previous display rather than fabricate BPM.
                    return self.visible
                cbpm = _f32(_f32(60.0 / duration) * self._pitch)
            else:
                cbpm = bpm
            kps = _f32(cbpm / 60.0)
            pseudo = False
        if not all(isfinite(v) and v > 0 for v in (bpm, cbpm, kps)):
            return self.visible
        reading = JrpBpmReading(bpm, cbpm, kps, precision=self._decimals)
        strings = reading.printed
        self._last_display = JrpDisplayedHud(strings, floor.seq_id, pseudo,
                                             outcome.count if pseudo else 1, self._hooks)
        return strings

    @classmethod
    def from_segment(cls, segment, *, mode='normal', check_pseudo=True,
                     decimals=2, is_game_scene=True) -> 'JrpOverlayRenderer':
        """Research adapter for DMDOD compiled floors (NOT native game parity).

        Reconstructs angleLength from compiled rotation time and floor BPM;
        this is plausible but not proof of the exact Unity scrFloor state.
        """
        chart = segment.chart
        floors = []
        for f in chart.floors:
            rotation_s = max(0., f.exit_time_s - f.target_time_s - f.pause_s)
            angle = rotation_s * pi * max(1e-9, f.bpm) / 60.
            floors.append(JrpFloorState(f.index, f.target_time_s, angle,
                            f.bpm / chart.initial_bpm, f.midspin))
        return cls(floors,base_bpm=chart.initial_bpm, pitch=segment.pitch_ratio,
                   decimals=decimals, mode=mode, check_pseudo=check_pseudo,
                   is_game_scene=is_game_scene)
