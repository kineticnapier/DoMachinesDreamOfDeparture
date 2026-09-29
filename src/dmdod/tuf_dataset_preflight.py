from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .adofai_chart import parse_adofai_bytes
from .tuf_dataset import DatasetSelection, TufCandidate, download_chart, select_dataset


@dataclass(frozen=True, slots=True)
class PreflightResult:
    selection: DatasetSelection
    rejected: tuple[tuple[int, str], ...]


def select_parser_compatible_dataset(
    candidates: Iterable[TufCandidate],
    *,
    train_count: int,
    validation_count: int,
    timeout_s: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> PreflightResult:
    """Select a dataset, rejecting charts the current DMDOD parser cannot load.

    TUF metadata does not expose whether a chart uses legacy ``pathData``.  The
    current DMDOD parser intentionally requires ``angleData``.  Therefore the
    selected charts are downloaded and parsed before the final selection is
    accepted.  Any incompatible chart is removed from the pool and the split is
    recomputed so a replacement candidate is promoted automatically.
    """

    pool = list(candidates)
    rejected: list[tuple[int, str]] = []
    validated_ids: set[int] = set()

    while True:
        selection = select_dataset(
            pool,
            train_count=train_count,
            validation_count=validation_count,
        )
        selected = (*selection.validation, *selection.train)
        bad_ids: set[int] = set()

        for candidate in selected:
            if candidate.level_id in validated_ids:
                continue
            if progress is not None:
                progress(
                    f"preflight #{candidate.level_id} {candidate.song} "
                    f"({candidate.difficulty_name})"
                )
            try:
                raw = download_chart(candidate, timeout_s=timeout_s)
                parse_adofai_bytes(raw, source_path=f"TUF:{candidate.level_id}")
            except Exception as exc:  # deliberate: network/format/parser failures all disqualify the chart
                reason = str(exc).strip() or type(exc).__name__
                rejected.append((candidate.level_id, reason))
                bad_ids.add(candidate.level_id)
                if progress is not None:
                    progress(f"reject #{candidate.level_id}: {reason}")
            else:
                validated_ids.add(candidate.level_id)

        if not bad_ids:
            return PreflightResult(selection=selection, rejected=tuple(rejected))

        pool = [candidate for candidate in pool if candidate.level_id not in bad_ids]
        if len(pool) < train_count + validation_count:
            raise ValueError(
                "not enough parser-compatible TUF charts after preflight: "
                f"{len(pool)} < {train_count + validation_count}"
            )
