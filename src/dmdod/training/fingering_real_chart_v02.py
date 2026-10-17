from __future__ import annotations

"""Train-only real-chart inventory for Fingering Network v0.2.

Difficulty and chart identity come from the saved dataset manifest.  Exact
compiled target times are privileged curriculum-builder data and are never
part of a Fingering Network inference observation.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training.prestige1_autoplay_timing import AutoplayTimingTeacher


FORMAT = "dmdod-fingering-real-chart-inventory-v0.2"
EXCLUDED_FAMILY_TOKENS = ("命日",)
EXCLUDED_VARIANT_TOKENS = ("nerfed",)
SPECIAL_MARKERS = frozenset(("SetSpeed", "Twirl", "MultiPlanet", "Pause"))


@dataclass(frozen=True, slots=True)
class CurriculumWindow:
    chart_sha256: str
    start_s: float
    end_s: float
    first_event: int
    last_event: int
    score: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.chart_sha256) == 64
            and isfinite(self.start_s)
            and isfinite(self.end_s)
            and 0 <= self.start_s < self.end_s
            and 0 <= self.first_event <= self.last_event
            and isfinite(self.score)
            and self.reasons
        ):
            raise ValueError("invalid real-chart curriculum window")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_family(value: str) -> str:
    text = " ".join(value.casefold().replace("_", " ").split())
    for token in ("(nerfed)", "[nerfed]", "nerfed ver.", "nerfed version"):
        text = text.replace(token, "")
    return " ".join(text.split())


def excluded_family_reason(name: str, metadata: Mapping[str, object]) -> str | None:
    fields = " ".join(str(metadata.get(key, "")) for key in ("song", "artist", "creator"))
    joined = f"{name} {fields}".casefold()
    if any(token.casefold() in joined for token in EXCLUDED_FAMILY_TOKENS):
        return "excluded_family_inochi"
    if any(token in joined for token in EXCLUDED_VARIANT_TOKENS):
        return "excluded_nerfed_variant"
    return None


def load_train_manifest(dataset: Path) -> list[dict]:
    """Load only the manifest's Train role and fail closed on path escapes."""

    dataset = dataset.resolve()
    payload = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = payload.get("Train")
    if not isinstance(rows, list) or not rows:
        raise ValueError("dataset manifest has no Train role")
    train_root = (dataset / "Train").resolve()
    output = []
    for raw in rows:
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise ValueError("invalid Train manifest row")
        relative = Path(raw["path"])
        path = (dataset / relative).resolve()
        if train_root not in path.parents or path.suffix.casefold() != ".adofai":
            raise ValueError(f"non-Train chart path in Train manifest: {relative}")
        if not path.is_file():
            raise FileNotFoundError(path)
        row = dict(raw)
        row["resolved_path"] = str(path)
        row["family"] = _normalise_family(str(raw.get("song") or path.stem))
        output.append(row)
    return output


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _event_markers(compiled, floor_indices: Iterable[int]) -> tuple[str, ...]:
    indices = set(floor_indices)
    return tuple(sorted({marker for floor in compiled.floors if floor.index in indices
                         for marker in floor.event_markers if marker in SPECIAL_MARKERS}))


def select_complex_windows(
    *, chart_sha256: str, events: tuple, compiled, window_s: float = 1.2,
    max_windows: int = 2, minimum_score: float = 2.0,
) -> tuple[CurriculumWindow, ...]:
    """Select deterministic, non-overlapping high-complexity event windows."""

    if window_s <= 0 or max_windows < 1:
        raise ValueError("invalid window budget")
    candidates = []
    times = [float(event.episode_time_s) for event in events]
    for index, event in enumerate(events):
        left = max(0, index - 2)
        right = min(len(events) - 1, index + 3)
        local_intervals = [1000.0 * (times[i] - times[i - 1])
                           for i in range(left + 1, right + 1)]
        short = sum(value <= 60.0 + 1e-7 for value in local_intervals)
        very_short = sum(value <= 35.0 + 1e-7 for value in local_intervals)
        chord_surplus = sum(max(0, events[i].chord_size - 1) for i in range(left, right + 1))
        markers = _event_markers(
            compiled, (floor for i in range(left, right + 1)
                       for floor in events[i].floor_indices)
        )
        jump = 0.0
        if index > 1 and index + 1 < len(events):
            before = max(1e-6, times[index] - times[index - 1])
            after = max(1e-6, times[index + 1] - times[index])
            jump = max(before / after, after / before)
        reasons = []
        if short:
            reasons.append("short_interval")
        if very_short:
            reasons.append("burst")
        if chord_surplus:
            reasons.append("multi_contact")
        if jump >= 1.8:
            reasons.append("density_change")
        reasons.extend(f"marker:{value}" for value in markers)
        score = short + 1.5 * very_short + 3.0 * chord_surplus + max(0.0, jump - 1.0)
        score += 1.5 * len(markers)
        if score < minimum_score or not reasons:
            continue
        start = max(0.0, times[index] - window_s * 0.45)
        end = min(float(compiled.duration_s), start + window_s)
        start = max(0.0, end - window_s)
        first = next((i for i, value in enumerate(times) if value >= start - 1e-9), index)
        last = max(i for i, value in enumerate(times) if value <= end + 1e-9)
        candidates.append(CurriculumWindow(
            chart_sha256, round(start, 9), round(end, 9), first, last,
            round(score, 6), tuple(dict.fromkeys(reasons)),
        ))
    chosen = []
    for candidate in sorted(candidates, key=lambda row: (-row.score, row.start_s, row.first_event)):
        overlaps = any(not (candidate.end_s <= old.start_s or candidate.start_s >= old.end_s)
                       for old in chosen)
        if not overlaps:
            chosen.append(candidate)
        if len(chosen) == max_windows:
            break
    return tuple(sorted(chosen, key=lambda row: row.start_s))


def inventory_chart(metadata: Mapping[str, object], *, max_windows: int = 2) -> dict:
    path = Path(str(metadata["resolved_path"]))
    digest = file_sha256(path)
    source = load_adofai(path)
    compiled = compile_adofai(source)
    segment = build_playable_segment(compiled)
    teacher = AutoplayTimingTeacher.from_segment(segment)
    events = teacher.events
    intervals_ms = [1000.0 * (events[i].episode_time_s - events[i - 1].episode_time_s)
                    for i in range(1, len(events))]
    bpms = [float(floor.bpm) for floor in compiled.floors if floor.bpm > 0]
    marker_counts = {marker: sum(marker in floor.event_markers for floor in compiled.floors)
                     for marker in sorted(SPECIAL_MARKERS)}
    full_rotations = sum(
        not floor.midspin and abs(floor.entry_angle_rad - floor.exit_angle_rad) < 1e-6
        for floor in compiled.floors[1:]
    )
    windows = select_complex_windows(
        chart_sha256=digest, events=events, compiled=compiled,
        max_windows=max_windows,
    )
    difficulty = metadata.get("difficulty_name")
    return {
        "name": path.name,
        "path": str(path),
        "sha256": digest,
        "family": metadata.get("family"),
        "level_id": metadata.get("level_id"),
        "difficulty": difficulty if isinstance(difficulty, str) and difficulty else None,
        "difficulty_source": "saved_dataset_manifest" if difficulty else None,
        "g10_or_harder_verified": bool(isinstance(difficulty, str) and difficulty.startswith(("G", "U"))
                                        and difficulty[1:].isdigit() and int(difficulty[1:]) >= 10),
        "excluded_reason": excluded_family_reason(path.name, metadata),
        "duration_s": round(compiled.duration_s, 6),
        "events": len(events),
        "targets": teacher.total_targets,
        "mean_events_per_s": round(len(events) / max(compiled.duration_s, 1e-9), 6),
        "bpm": {
            "manifest": metadata.get("bpm"),
            "compiled_min": min(bpms) if bpms else None,
            "compiled_max": max(bpms) if bpms else None,
            "distinct_rounded": len({round(value, 3) for value in bpms}),
        },
        "interval_ms": {
            "minimum": min(intervals_ms) if intervals_ms else None,
            "p10": _percentile(intervals_ms, 0.10),
            "median": median(intervals_ms) if intervals_ms else None,
            "under_35": sum(value <= 35.0 + 1e-7 for value in intervals_ms),
            "under_60": sum(value <= 60.0 + 1e-7 for value in intervals_ms),
        },
        "multi_contact_events": sum(event.chord_size > 1 for event in events),
        "max_chord_size": max(event.chord_size for event in events),
        "midspins": sum(floor.midspin for floor in compiled.floors),
        "full_rotations": full_rotations,
        "markers": marker_counts,
        "complex_windows": [asdict(window) for window in windows],
    }


def build_inventory(dataset: Path, *, max_charts: int | None = None,
                    max_windows_per_chart: int = 2) -> dict:
    rows = load_train_manifest(dataset)
    if max_charts is not None:
        if max_charts < 1:
            raise ValueError("max_charts must be positive")
        rows = rows[:max_charts]
    charts = [inventory_chart(row, max_windows=max_windows_per_chart) for row in rows]
    hashes = [chart["sha256"] for chart in charts]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("duplicate Train chart SHA256")
    windows = [window for chart in charts for window in chart["complex_windows"]]
    return {
        "format": FORMAT,
        "source_role": "Train only",
        "dataset": str(dataset.resolve()),
        "limits": {"max_charts": max_charts, "max_windows_per_chart": max_windows_per_chart},
        "charts": charts,
        "summary": {
            "charts": len(charts),
            "unique_sha256": len(set(hashes)),
            "verified_g10_or_harder": sum(chart["g10_or_harder_verified"] for chart in charts),
            "excluded_family_charts": sum(chart["excluded_reason"] is not None for chart in charts),
            "candidate_windows": len(windows),
            "charts_with_candidate_windows": sum(bool(chart["complex_windows"]) for chart in charts),
            "difficulty_counts": {
                difficulty: sum(chart["difficulty"] == difficulty for chart in charts)
                for difficulty in sorted({chart["difficulty"] for chart in charts if chart["difficulty"]})
            },
        },
        "holdout_roles_opened": [],
    }
