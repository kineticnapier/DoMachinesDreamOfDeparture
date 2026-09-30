from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dmdod.tuf_balanced_dataset import (
    build_balanced_dataset,
    select_balanced_dataset,
    select_parser_compatible_balanced_dataset,
)
from dmdod.tuf_dataset_parallel import fetch_candidates_parallel


def _print_role(role: str, items) -> None:
    counts = Counter(item.difficulty_name for item in items)
    print(f"\n=== {role} ({len(items)}) === {dict(sorted(counts.items()))}")
    for item in items:
        print(
            f"#{item.level_id:5d} {'C' if item.curated else '-'} "
            f"UC={item.unique_clears:4d} clears={item.clears:5d} "
            f"BPM={item.bpm:7.2f} len={item.length_ms / 1000.0:6.1f}s "
            f"{item.difficulty_name} | {item.song} | {item.creator}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced three-way DMDOD dataset from public TUF P charts. "
            "Defaults target P7-P10 with 16 Train, 4 Validation, and 2 Final "
            "charts per difficulty (64/16/8 total)."
        )
    )
    parser.add_argument("--output", default="data/DMDOD-v120-tuf-p7-p10")
    parser.add_argument("--min-p-difficulty", type=int, default=7)
    parser.add_argument("--max-p-difficulty", type=int, default=10)
    parser.add_argument("--train-per-difficulty", type=int, default=16)
    parser.add_argument("--validation-per-difficulty", type=int, default=4)
    parser.add_argument("--final-per-difficulty", type=int, default=2)
    parser.add_argument("--scan-limit", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-unique-clears", type=int, default=0)
    parser.add_argument("--request-delay", type=float, default=0.02)
    parser.add_argument("--download-delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.min_p_difficulty <= 0 or args.max_p_difficulty < args.min_p_difficulty:
        raise SystemExit("P difficulty range must satisfy 1 <= min <= max")
    if min(
        args.train_per_difficulty,
        args.validation_per_difficulty,
        args.final_per_difficulty,
    ) <= 0:
        raise SystemExit("per-difficulty quotas must be positive")
    if args.scan_limit < 0 or args.min_unique_clears < 0:
        raise SystemExit("--scan-limit and --min-unique-clears must be non-negative")
    if args.page_size <= 0 or args.workers <= 0:
        raise SystemExit("--page-size and --workers must be positive")
    if args.request_delay < 0.0 or args.download_delay < 0.0:
        raise SystemExit("delays must be non-negative")
    if args.timeout <= 0.0:
        raise SystemExit("--timeout must be positive")

    difficulty_count = args.max_p_difficulty - args.min_p_difficulty + 1
    target_train = difficulty_count * args.train_per_difficulty
    target_validation = difficulty_count * args.validation_per_difficulty
    target_final = difficulty_count * args.final_per_difficulty

    print("=== DMDOD balanced TUF Dataset Builder ===")
    print(
        f"difficulty=P{args.min_p_difficulty}-P{args.max_p_difficulty} "
        f"per-P={args.train_per_difficulty}/{args.validation_per_difficulty}/{args.final_per_difficulty} "
        f"target=Train{target_train}+Validation{target_validation}+Final{target_final}"
    )
    print(
        f"scan={'ALL' if args.scan_limit == 0 else args.scan_limit} "
        f"page-request={args.page_size} workers={args.workers} "
        f"minUniqueClears={args.min_unique_clears}"
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

    candidate_counts = Counter(item.difficulty_name for item in candidates)
    print(f"candidate pool: {len(candidates)} {dict(sorted(candidate_counts.items()))}")

    if args.dry_run:
        selection = select_balanced_dataset(
            candidates,
            min_p_difficulty=args.min_p_difficulty,
            max_p_difficulty=args.max_p_difficulty,
            train_per_difficulty=args.train_per_difficulty,
            validation_per_difficulty=args.validation_per_difficulty,
            final_per_difficulty=args.final_per_difficulty,
        )
        rejected = ()
        raw_by_level_id: dict[int, bytes] = {}
        print("NOTE: dry-run skips downloads/parser preflight")
    else:
        preflight = select_parser_compatible_balanced_dataset(
            candidates,
            min_p_difficulty=args.min_p_difficulty,
            max_p_difficulty=args.max_p_difficulty,
            train_per_difficulty=args.train_per_difficulty,
            validation_per_difficulty=args.validation_per_difficulty,
            final_per_difficulty=args.final_per_difficulty,
            timeout_s=args.timeout,
            progress=print,
        )
        selection = preflight.selection
        rejected = preflight.rejected
        raw_by_level_id = preflight.raw_by_level_id
        if rejected:
            print(f"preflight: rejected {len(rejected)} incompatible chart(s); replacements selected")

    if selection.relaxed_creator_overlap:
        print("WARNING: creator-disjoint roles were impossible; creator overlap was relaxed")
    else:
        print("split: Train/Validation/Final songs and known creators are role-disjoint")

    _print_role("Final", selection.final)
    _print_role("Validation", selection.validation)
    _print_role("Train", selection.train)

    if args.dry_run:
        print("\ndry-run: no files downloaded")
        return

    metadata = {
        "difficulty_gate": f"P{args.min_p_difficulty}-P{args.max_p_difficulty}",
        "min_p_difficulty": args.min_p_difficulty,
        "max_p_difficulty": args.max_p_difficulty,
        "train_per_difficulty": args.train_per_difficulty,
        "validation_per_difficulty": args.validation_per_difficulty,
        "final_per_difficulty": args.final_per_difficulty,
        "scan_limit": args.scan_limit,
        "page_size_requested": args.page_size,
        "scan_workers": args.workers,
        "min_unique_clears": args.min_unique_clears,
        "candidate_count": len(candidates),
        "candidate_difficulty_counts": dict(sorted(candidate_counts.items())),
        "parser_preflight_rejected": [
            {"level_id": level_id, "reason": reason} for level_id, reason in rejected
        ],
    }
    built = build_balanced_dataset(
        selection,
        raw_by_level_id=raw_by_level_id,
        output=Path(args.output),
        metadata=metadata,
        download_delay_s=args.download_delay,
        timeout_s=args.timeout,
        progress=print,
    )
    print(f"\ndataset ready: {built}")
    print(f"manifest: {built / 'manifest.json'}")


if __name__ == "__main__":
    main()
