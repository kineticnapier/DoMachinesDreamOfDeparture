from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dmdod.tuf_dataset import build_dataset, select_dataset
from dmdod.tuf_dataset_parallel import fetch_candidates_parallel
from dmdod.tuf_dataset_preflight import select_parser_compatible_dataset


def _difficulty_label(min_p: int, max_p: int) -> str:
    return f"P{min_p}" if min_p == max_p else f"P{min_p}-P{max_p}"


def _print_selection(role: str, items) -> None:
    print(f"\n=== {role} ({len(items)}) ===")
    for item in items:
        length_s = item.length_ms / 1000.0
        print(
            f"#{item.level_id:5d} {'C' if item.curated else '-'} "
            f"UC={item.unique_clears:4d} clears={item.clears:5d} "
            f"BPM={item.bpm:7.2f} len={length_s:6.1f}s "
            f"{item.difficulty_name} | {item.song} | {item.creator}"
        )


def _print_distribution(label: str, items) -> None:
    curated = sum(item.curated for item in items)
    unique_clears = sum(item.unique_clears for item in items)
    bpm = Counter(item.bpm_bin for item in items)
    lengths = Counter(item.length_bin for item in items)
    difficulties = Counter(item.difficulty_name for item in items)
    print(
        f"{label}: curated={curated}/{len(items)} "
        f"uniqueClears(sum)={unique_clears} "
        f"BPM={dict(sorted(bpm.items()))} lengths={dict(sorted(lengths.items()))}"
    )
    print(f"{label} difficulties: " + " | ".join(f"{key}:{value}" for key, value in difficulties.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DMDOD Train/Validation dataset from public TUF levels. "
            "Selection prioritizes curation and clear evidence, then spreads BPM, "
            "length, density, difficulty, songs, and creators."
        )
    )
    parser.add_argument("--output", default="data/DMDOD-v090-tuf")
    parser.add_argument(
        "--final-source",
        required=True,
        help="Existing dataset directory/ZIP (uses Final/) or a direct Final directory.",
    )
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--validation", type=int, default=10)
    parser.add_argument(
        "--min-p-difficulty",
        type=int,
        default=1,
        help="Lowest ordinary P difficulty to include (default: 1).",
    )
    parser.add_argument(
        "--max-p-difficulty",
        type=int,
        default=6,
        help="Highest ordinary P difficulty to include (default: 6).",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="Metadata rows to scan; 0 scans the complete public TUF level database.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=500,
        help="Requested first-page size. TUF may clamp it; the actual returned size becomes the stride.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent metadata page requests after the first page.",
    )
    parser.add_argument("--min-unique-clears", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.02)
    parser.add_argument("--download-delay", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.train <= 0 or args.validation <= 0:
        raise SystemExit("--train and --validation must be positive")
    if args.min_p_difficulty <= 0 or args.max_p_difficulty < args.min_p_difficulty:
        raise SystemExit("P difficulty range must satisfy 1 <= min <= max")
    if args.scan_limit < 0 or args.min_unique_clears < 0:
        raise SystemExit("--scan-limit and --min-unique-clears must be non-negative")
    if args.page_size <= 0 or args.workers <= 0:
        raise SystemExit("--page-size and --workers must be positive")
    if args.request_delay < 0.0 or args.download_delay < 0.0:
        raise SystemExit("delays must be non-negative")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")

    difficulty_label = _difficulty_label(args.min_p_difficulty, args.max_p_difficulty)

    print("=== DMDOD TUF Dataset Builder ===")
    print(
        f"scan={'ALL' if args.scan_limit == 0 else args.scan_limit} "
        f"page-request={args.page_size} workers={args.workers} "
        f"minUniqueClears={args.min_unique_clears} difficulty={difficulty_label} "
        f"target=Train{args.train}+Validation{args.validation}"
    )

    last_report = -1

    def scan_progress(scanned: int, total: int | None, eligible: int) -> None:
        nonlocal last_report
        bucket = scanned // 1000
        if scanned == total or bucket != last_report:
            last_report = bucket
            total_text = "?" if total is None else str(total)
            print(f"scan: {scanned}/{total_text} rows, eligible={eligible}")

    candidates = fetch_candidates_parallel(
        page_size=args.page_size,
        workers=args.workers,
        scan_limit=args.scan_limit,
        min_unique_clears=args.min_unique_clears,
        min_p_difficulty=args.min_p_difficulty,
        max_p_difficulty=args.max_p_difficulty,
        request_delay_s=args.request_delay,
        timeout_s=args.timeout,
        progress=scan_progress,
    )
    if len(candidates) < args.train + args.validation:
        raise SystemExit(
            f"not enough usable TUF charts: {len(candidates)} < {args.train + args.validation}"
        )

    curated_count = sum(item.curated for item in candidates)
    print(f"candidate pool: {len(candidates)} usable, curated={curated_count}")

    rejected: tuple[tuple[int, str], ...] = ()
    if args.dry_run:
        selection = select_dataset(
            candidates,
            train_count=args.train,
            validation_count=args.validation,
        )
        print("NOTE: dry-run does not download charts; pathData/parser compatibility is checked on real build")
    else:
        preflight = select_parser_compatible_dataset(
            candidates,
            train_count=args.train,
            validation_count=args.validation,
            timeout_s=args.timeout,
            progress=print,
        )
        selection = preflight.selection
        rejected = preflight.rejected
        if rejected:
            print(f"preflight: rejected {len(rejected)} incompatible chart(s); replacements selected")

    if selection.relaxed_creator_overlap:
        print("WARNING: creator-disjoint Train/Validation was impossible; creator split was relaxed")
    else:
        print("split: Train/Validation songs and known chart creators are disjoint")

    _print_distribution("Validation", selection.validation)
    _print_distribution("Train", selection.train)
    _print_selection("Validation", selection.validation)
    _print_selection("Train", selection.train)

    if args.dry_run:
        print("\ndry-run: no files downloaded")
        return

    output = Path(args.output)
    metadata = {
        "scan_limit": args.scan_limit,
        "page_size_requested": args.page_size,
        "scan_workers": args.workers,
        "min_unique_clears": args.min_unique_clears,
        "difficulty_gate": difficulty_label,
        "min_p_difficulty": args.min_p_difficulty,
        "max_p_difficulty": args.max_p_difficulty,
        "candidate_count": len(candidates),
        "curated_candidate_count": curated_count,
        "parser_preflight_rejected": [
            {"level_id": level_id, "reason": reason} for level_id, reason in rejected
        ],
    }
    built = build_dataset(
        selection,
        output=output,
        final_source=args.final_source,
        download_delay_s=args.download_delay,
        timeout_s=args.timeout,
        metadata=metadata,
        progress=print,
    )
    print(f"\ndataset ready: {built}")
    print(f"manifest: {built / 'manifest.json'}")


if __name__ == "__main__":
    main()
