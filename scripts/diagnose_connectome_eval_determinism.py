from __future__ import annotations

"""Diagnose closed-loop evaluation determinism for N-key connectome checkpoints.

This intentionally performs evaluation only: no expert/student collection, BC,
optimizer construction, trust interpolation, or checkpoint writes.  Repeated
runs are compared anchor-by-anchor using the exact metrics that can influence
trust-region selection or its diagnostics.
"""

import argparse
import hashlib
from pathlib import Path

import torch

import train_real_chart_v080 as v080
import train_real_chart_v161_n_key_dagger as v161
import train_real_chart_v162_n_key_continuous_dagger as v162
import train_real_chart_v171_n_key_connectome_failure_continuation_dagger as v171
from dmdod.multichart_dataset import discover_multichart_dataset


def _signature(stats, keydowns: int) -> tuple:
    return (
        int(stats.hits),
        int(stats.targets),
        float(stats.x_accuracy_percent),
        float(stats.perfect_rate),
        None if stats.mean_abs_error_ms is None else float(stats.mean_abs_error_ms),
        int(stats.too_early_presses),
        bool(stats.overloaded),
        int(keydowns),
    )


def _fingerprint(signatures: list[tuple]) -> str:
    return hashlib.sha256(repr(signatures).encode("utf-8")).hexdigest()[:16]


def _format_signature(signature: tuple) -> str:
    hits, targets, xacc, _perfect_rate, mae, early, overloaded, keydowns = signature
    mae_text = "nan" if mae is None else f"{mae:.6f}"
    return (
        f"H={hits}/{targets} X={xacc:.8f}% MAE={mae_text}ms "
        f"early={early} over={overloaded} keydowns={keydowns}"
    )


def _build_model(checkpoint: dict, *, device: torch.device):
    return v171._build_policy_from_checkpoint(checkpoint, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Repeatedly evaluate the same checkpoint and Train anchors, then report "
            "any anchor-level nondeterminism."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("checkpoint")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--anchor-limit", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-lifecycle",
        choices=("rebuild", "reuse"),
        default="rebuild",
        help=(
            "rebuild reconstructs/reloads the model before every run and best matches "
            "separate trainer invocations; reuse isolates repeated evaluation on one model"
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable torch deterministic algorithms before model construction",
    )
    args = parser.parse_args()

    if args.runs < 2:
        raise SystemExit("--runs must be at least 2")
    if args.anchor_limit <= 0:
        raise SystemExit("--anchor-limit must be positive")

    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    device = v161._device_from_arg(args.device)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    control_dt_s = float(checkpoint["control_dt"])
    physics_dt_s = float(checkpoint.get("physics_dt", 0.001))

    dataset = discover_multichart_dataset(args.dataset)
    train_charts = v080._compile_role(dataset.train)
    anchors = v080._build_anchor_segments(
        train_charts,
        window_s=float(checkpoint["train_window"]),
        anchors_per_chart=int(checkpoint["anchors_per_chart"]),
    )[: args.anchor_limit]
    if not anchors:
        raise SystemExit("no Train anchors selected")

    print("=== DMDOD connectome eval determinism diagnostic ===")
    print(
        f"checkpoint={checkpoint_path} backend={checkpoint.get('n_key_policy_backend', 'unknown')} "
        f"device={device} anchors={len(anchors)} runs={args.runs} "
        f"model-lifecycle={args.model_lifecycle} deterministic={args.deterministic}"
    )
    print("No training, collection, optimizer step, trust interpolation, or checkpoint write is performed.")

    model = None
    if args.model_lifecycle == "reuse":
        model = _build_model(checkpoint, device=device)

    baseline: list[tuple] | None = None
    baseline_fingerprint: str | None = None
    differing_runs = 0
    changed_anchor_counts: dict[int, int] = {}

    for run_index in range(1, args.runs + 1):
        if args.model_lifecycle == "rebuild":
            model = _build_model(checkpoint, device=device)
        assert model is not None

        results = []
        signatures: list[tuple] = []
        for named in anchors:
            stats, keydowns = v162._evaluate_continuous(
                model,
                named,
                control_dt_s=control_dt_s,
                physics_dt_s=physics_dt_s,
                device=device,
            )
            results.append((stats, keydowns))
            signatures.append(_signature(stats, keydowns))

        fingerprint = _fingerprint(signatures)
        print(
            f"run {run_index:02d}/{args.runs}: {v161._aggregate(results)} "
            f"fingerprint={fingerprint}"
        )

        if baseline is None:
            baseline = signatures
            baseline_fingerprint = fingerprint
            for anchor_index, (named, signature) in enumerate(zip(anchors, signatures), 1):
                print(
                    f"  baseline {anchor_index:02d} {named.chart_name}: "
                    f"{_format_signature(signature)}"
                )
            continue

        changed = [
            index
            for index, (expected, actual) in enumerate(zip(baseline, signatures), 1)
            if actual != expected
        ]
        if not changed:
            continue

        differing_runs += 1
        print(
            f"  DIFFERENCE vs run 01: anchors={','.join(str(index) for index in changed)} "
            f"baseline-fingerprint={baseline_fingerprint}"
        )
        for anchor_index in changed:
            changed_anchor_counts[anchor_index] = changed_anchor_counts.get(anchor_index, 0) + 1
            named = anchors[anchor_index - 1]
            expected = baseline[anchor_index - 1]
            actual = signatures[anchor_index - 1]
            print(f"    anchor {anchor_index:02d} {named.chart_name}")
            print(f"      run01: {_format_signature(expected)}")
            print(f"      run{run_index:02d}: {_format_signature(actual)}")

    assert baseline_fingerprint is not None
    if differing_runs == 0:
        print(
            f"DETERMINISTIC: all {args.runs} runs exactly matched "
            f"fingerprint={baseline_fingerprint}"
        )
    else:
        hot = ", ".join(
            f"{index}:{count}/{args.runs - 1}"
            for index, count in sorted(changed_anchor_counts.items())
        )
        print(
            f"NONDETERMINISTIC: {differing_runs}/{args.runs - 1} reruns differed from run01; "
            f"changed-anchor-frequency={hot}"
        )


if __name__ == "__main__":
    main()
