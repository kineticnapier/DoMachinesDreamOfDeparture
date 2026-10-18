from __future__ import annotations

"""Interruptible frozen Causal Lead v0.2 multichart evaluation with wall-time goal.

Use --goal-seconds as a *performance target*, never as a hard timeout or as
permission to skip charts or to relax scorer fidelity. Each of the four
chart×policy×arm simulations has one create-once JSON checkpoint. --resume
loads the exact frozen preselected chart manifest and all finished arms.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fingering_causal_early_v02 as causal
import audit_fingering_causal_multichart_v02 as multichart
import audit_oracle_fingering_v02 as teacher
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai
from dmdod.training import fingering_network_v01 as network


FORMAT = "dmdod-fingering-v02-causal-multichart-goal-resumable-v1"
POLICIES = multichart.POLICIES
ARMS = multichart.ARMS
MANIFEST = "run-manifest.json"
POLICY_ARMS = len(POLICIES) * len(ARMS)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path) -> str:
    return viewer.file_sha256(path)


def frozen_code_hashes() -> dict:
    """Reject resume if scripts defining decisions or metrics have changed."""
    paths = {
        "causal": Path(causal.__file__),
        "multichart": Path(multichart.__file__),
        "teacher": Path(teacher.__file__),
        "network": Path(network.__file__),
    }
    # Script versions affecting inference/scoring are frozen for resume.
    return {name: hash_file(path.resolve()) for name, path in paths.items()}


def write_json_atomic(path: Path, payload: dict, *, create_only: bool) -> None:
    """Fsync temporary JSON on the same filesystem, then atomically publish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_only and path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    fd, name = tempfile.mkstemp(prefix=".writing-", suffix=".json", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if create_only and path.exists():
            raise FileExistsError(f"concurrent checkpoint appeared: {path}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def exclusive_run_lock(folder: Path):
    """Refuse concurrent writers; release automatically on Ctrl+C/normal exit."""
    marker = folder / ".running.lock"
    try:
        with marker.open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\nstarted={utc()}\n")
    except FileExistsError as exc:
        raise RuntimeError(
            f"another run or stale lock at {marker}; verify no process is running "
            "before manually removing that lock only"
        ) from exc
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def protocol_metadata(checkpoint: Path, reference_replay: Path) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    reference = json.loads(reference_replay.read_text(encoding="utf-8"))
    if reference.get("format") != "dmdod-fingering-network-v0.1.1-human-proxy-comparison":
        raise ValueError("unexpected bool reference replay format")
    checkpoint_sha = hash_file(checkpoint)
    if checkpoint_sha != reference["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from proven bool reference")
    return {
        "checkpoint_absolute_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "reference_replay_sha256": hash_file(reference_replay),
        "code_hashes": frozen_code_hashes(),
        "settings": {
            "policies": [list(p) for p in POLICIES],
            "arms": list(ARMS),
            "decision_horizon_extra_ms": causal.EXTRA_LEAD_S * 1000,
            "negative_activation_gate": causal.NEGATIVE_ACTIVATION_GATE,
            "control_dt_ms": teacher.CTRL * 1000,
            "physics_dt_ms": teacher.PHYS * 1000,
            "uses_oracle_timing": True,
            "uses_saved_future_allocations": False,
        }
    }


def start_or_resume(output_dir: Path, *, resume: bool, root: Path,
                    charts: list[Path], max_charts: int,
                    protocol: dict) -> dict:
    path = output_dir / MANIFEST
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"no saved run manifest at {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("protocol") != protocol:
            raise ValueError(
                "resume refused: checkpoint, code hash, reference, or control "
                "settings changed; do not mix incompatible evaluations"
            )
        # Do not resample! Verify the paths and bytes that were originally picked.
        for row in manifest["selected"]:
            file = Path(row["path"])
            if not file.is_file() or hash_file(file) != row["sha256"]:
                raise RuntimeError(f"resume chart content drift: {file}")
        return manifest
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    selected, selection = multichart.select_charts(root, charts, max_charts)
    manifest = {
        "format": FORMAT,
        "created_at_utc": utc(),
        "selected": selected,
        "selection": selection,
        "protocol": protocol,
        "chart_order_locked_before_evaluation": True,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(path, manifest, create_only=True)
    return manifest


def checkpoint_path(output_dir: Path, chart_index: int, chart_sha: str,
                    policy: str, arm: str) -> Path:
    if policy not in dict(POLICIES) or arm not in ARMS:
        raise ValueError("unknown policy/arm")
    return output_dir / "arms" / f"chart-{chart_index:03d}-{chart_sha[:16]}-{policy}-{arm}.json"


def checkpoint_payload(run: dict, *, chart_sha: str, checkpoint_sha: str,
                       policy: str, arm: str, elapsed_s: float) -> dict:
    """Do not serialize live Motor environment, trace, or transient tensors."""
    return {
        "format": FORMAT + "-arm",
        "chart_sha256": chart_sha,
        "checkpoint_sha256": checkpoint_sha,
        "policy": policy,
        "arm": arm,
        "measured_wall_s": round(elapsed_s, 6),
        "finished_at_utc": utc(),
        "result": {
            "stats": run["stats"],
            "decisions": run["decisions"],
            "interventions": run["interventions"],
            "termination_reason": run["termination_reason"],
            "elapsed_wall_s": run["elapsed_wall_s"],
        },
    }


def load_arm(path: Path, *, chart_sha: str, checkpoint_sha: str,
             policy: str, arm: str) -> dict | None:
    if not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = (FORMAT + "-arm", chart_sha, checkpoint_sha, policy, arm)
    actual = tuple(saved.get(field) for field in (
        "format", "chart_sha256", "checkpoint_sha256", "policy", "arm"
    ))
    if actual != expected:
        raise ValueError(f"incompatible saved arm checkpoint {path}")
    if not math.isfinite(saved["measured_wall_s"]) or saved["measured_wall_s"] < 0:
        raise ValueError(f"invalid checkpoint execution time: {path}")
    return saved


def snapshot(report: dict, output_dir: Path, goal_seconds: float | None) -> dict:
    """Reconstruct from durable arm checkpoints; incomplete pairs stay visible."""
    rows = []
    duration = 0.0
    completed = 0
    total = len(report["selected"]) * POLICY_ARMS
    for i, chart in enumerate(report["selected"], start=1):
        entry = dict(chart)
        entry["policies"] = {}
        raw_pairs = {}
        for policy, _ in POLICIES:
            runs = {}
            for arm in ARMS:
                path = checkpoint_path(output_dir, i, chart["sha256"], policy, arm)
                saved = load_arm(path, chart_sha=chart["sha256"],
                                 checkpoint_sha=report["protocol"]["checkpoint_sha256"],
                                 policy=policy, arm=arm)
                if saved is None:
                    continue
                runs[arm] = saved["result"]
                duration += saved["measured_wall_s"]
                completed += 1
            raw_pairs[policy] = runs
            if all(arm in runs for arm in ARMS):
                entry["policies"][policy] = multichart.compare_pair(
                    runs["baseline"], runs["causal_lead"]
                )
            else:
                entry["policies"][policy] = {
                    "pending_arms": [arm for arm in ARMS if arm not in runs],
                    "completed_arms": list(runs),
                }
        entry["completed_arm_count"] = sum(len(runs) for runs in raw_pairs.values())
        rows.append(entry)
    remaining = total - completed
    estimated_remaining = (duration / completed * remaining) if completed else None
    projected_total = (duration + estimated_remaining
                       if estimated_remaining is not None else None)
    return {
        "format": FORMAT + "-progress",
        "updated_at_utc": utc(),
        "goal_seconds": goal_seconds,
        "goal_mode_enabled": goal_seconds is not None,
        "goal_kind": "total_accumulated_completed_arm_wall_time_not_hard_timeout",
        "completed_arms": completed,
        "total_arms": total,
        "completed_charts": sum(row["completed_arm_count"] == POLICY_ARMS for row in rows),
        "total_charts": len(rows),
        "spent_completed_arms_s": round(duration, 3),
        "estimated_remaining_s": (
            round(estimated_remaining, 3) if estimated_remaining is not None else None
        ),
        "projected_total_s": (
            round(projected_total, 3) if projected_total is not None else None
        ),
        "goal_status": (
            "not_set" if goal_seconds is None else
            "insufficient_samples" if not completed else
            "projected_within_target" if projected_total <= goal_seconds else
            "projected_over_target"
        ),
        "charts": rows,
        # The legacy aggregator expects complete baseline+lead pairs. Hide
        # pending partial arms from it, but KEEP them in progress.json.
        "aggregate_completed_pairs_only": multichart.aggregate([
            {**row, "policies": {
                policy: pair for policy, pair in row["policies"].items()
                if all(arm in pair for arm in ARMS)
            }} for row in rows
        ]),
    }


def publish_progress(manifest: dict, output_dir: Path,
                     goal_seconds: float | None) -> dict:
    state = snapshot(manifest, output_dir, goal_seconds)
    write_json_atomic(output_dir / "progress.json", state, create_only=False)
    return state


def print_progress(progress: dict) -> None:
    n = progress["completed_arms"]
    total = progress["total_arms"]
    goal = progress["goal_seconds"]
    target_message = ""
    if goal is not None:
        projected = progress["projected_total_s"]
        remaining = progress["estimated_remaining_s"]
        target_message = (
            f"; goal={goal:.1f}s; estimated remaining={remaining:.1f}s"
            f"; projected total={projected:.1f}s ({progress['goal_status']})"
            if projected is not None else f"; goal={goal:.1f}s"
        )
    print(
        f"progress {n}/{total} arms, {progress['completed_charts']}/"
        f"{progress['total_charts']} charts; "
        f"completed-arm CPU/wall run-time sum="
        f"{progress['spent_completed_arms_s']:.1f}s{target_message}",
        flush=True,
    )


def run(manifest: dict, output_dir: Path, *, wall_timeout_s: float,
        goal_seconds: float | None, checkpoint: Path) -> dict:
    model = causal.runner.load_model(checkpoint)
    protocol = manifest["protocol"]
    for index, chart in enumerate(manifest["selected"], start=1):
        path = Path(chart["path"])
        if hash_file(path) != chart["sha256"]:
            raise RuntimeError(f"chart changed during evaluation: {path}")
        segment = None
        for policy, alpha in POLICIES:
            for arm in ARMS:
                destination = checkpoint_path(output_dir, index, chart["sha256"],
                                              policy, arm)
                saved = load_arm(destination, chart_sha=chart["sha256"],
                                 checkpoint_sha=protocol["checkpoint_sha256"],
                                 policy=policy, arm=arm)
                if saved is not None:
                    print(f"[{index}/{len(manifest['selected'])}] "
                          f"skip saved {policy}/{arm}: {chart['name']}", flush=True)
                    continue
                if segment is None:
                    segment = build_playable_segment(
                        compile_adofai(load_adofai(path))
                    )
                print(f"[{index}/{len(manifest['selected'])}] {chart['name']} "
                      f"{policy}/{arm} starting", flush=True)
                started = time.perf_counter()
                result = causal.execute(segment, model, policy, alpha, arm,
                                        wall_timeout_s=wall_timeout_s)
                elapsed_s = time.perf_counter() - started
                value = checkpoint_payload(
                    result, chart_sha=chart["sha256"],
                    checkpoint_sha=protocol["checkpoint_sha256"],
                    policy=policy, arm=arm, elapsed_s=elapsed_s
                )
                write_json_atomic(destination, value, create_only=True)
                print(f"  H={result['stats']['hits']}/{result['stats']['targets']} "
                      f"miss={result['stats']['misses']} "
                      f"X={result['stats']['x_accuracy_percent']}% "
                      f"TE={result['stats']['too_early']} "
                      f"OL={result['stats']['overloaded']} "
                      f"wall={elapsed_s:.2f}s; saved", flush=True)
                print_progress(publish_progress(manifest, output_dir, goal_seconds))
    return publish_progress(manifest, output_dir, goal_seconds)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart-root", type=Path,
                   default=Path("data/DMDOD-v140-tuf-p11-p20"))
    p.add_argument("--chart", type=Path, action="append", default=[])
    p.add_argument("--max-charts", type=int, default=3)
    p.add_argument("--checkpoint", type=Path, default=causal.CHECKPOINT)
    p.add_argument("--reference-replay", type=Path, default=causal.SAVED_REPLAY)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true",
                   help="resume from immutable run manifest, skipping saved arms")
    p.add_argument("--goal-seconds", "--target-seconds", type=float, default=None,
                   help="total accumulated arm evaluation time goal; never a cutoff")
    p.add_argument("--wall-timeout-s", type=float, default=600.0,
                   help="per-arm safety timeout; unrelated to --goal-seconds")
    args = p.parse_args()
    if args.wall_timeout_s <= 0 or not math.isfinite(args.wall_timeout_s):
        p.error("--wall-timeout-s must be finite and positive")
    if args.goal_seconds is not None and (
        not math.isfinite(args.goal_seconds) or args.goal_seconds <= 0
    ):
        p.error("--goal-seconds must be finite and positive")
    if args.resume and args.chart:
        p.error("--resume cannot add or change charts; chart selection is frozen")
    # A resumed run does not reselect charts. max-charts/chart-root are ignored.
    protocol = protocol_metadata(args.checkpoint, args.reference_replay)
    manifest = start_or_resume(
        args.output_dir, resume=args.resume,
        root=args.chart_root, charts=args.chart,
        max_charts=args.max_charts, protocol=protocol
    )
    print(f"locked chart sample: {len(manifest['selected'])} "
          f"from {manifest['selection']['selection_basis']}", flush=True)
    with exclusive_run_lock(args.output_dir):
        print_progress(publish_progress(manifest, args.output_dir, args.goal_seconds))
        try:
            finished = run(manifest, args.output_dir,
                           wall_timeout_s=args.wall_timeout_s,
                           goal_seconds=args.goal_seconds,
                           checkpoint=args.checkpoint)
        except KeyboardInterrupt:
            # Never discard already published per-arm checkpoints.
            stopped = publish_progress(manifest, args.output_dir, args.goal_seconds)
            print("\nStopped by Ctrl+C. Finished arms remain saved.", flush=True)
            print_progress(stopped)
            print("Resume: re-run with --resume and the SAME --output-dir",
                  flush=True)
            raise SystemExit(130) from None
        print_progress(finished)
        print("progress file:", args.output_dir / "progress.json", flush=True)
        if finished["completed_arms"] == finished["total_arms"]:
            report = {
                "format": FORMAT + "-complete",
                "generated_at_utc": utc(),
                "selection": manifest["selection"],
                "checkpoint": manifest["protocol"]["checkpoint_absolute_path"],
                "checkpoint_sha256": manifest["protocol"]["checkpoint_sha256"],
                "protocol": manifest["protocol"],
                "goal": {
                    "target_seconds": args.goal_seconds,
                    "measured_completed_arm_wall_seconds": finished[
                        "spent_completed_arms_s"
                    ],
                    "status": finished["goal_status"],
                    "counts_only_successful_finished_arm_time": True,
                },
                "charts": finished["charts"],
                "aggregate": finished["aggregate_completed_pairs_only"],
                "limitations": [
                    "Timing is still exact AutoPlay Oracle, not HUD-only Student.",
                    "Goal is a reporting target, NEVER an experimental cutoff.",
                    "Time goal sums finished arm wall times; interruptions and loading"
                    " overhead are excluded; ETA is a rough arm-average projection.",
                    "Immutable checkpoint/selection/protocol hashes guard resumed runs.",
                ],
            }
            write_json_atomic(
                args.output_dir / "multichart-summary.json",
                report, create_only=not args.resume
            )
            html_path = args.output_dir / "multichart-summary.html"
            html = multichart.render_html(report)
            if html_path.exists() and not args.resume:
                raise FileExistsError(html_path)
            fd, tmp_name = tempfile.mkstemp(
                prefix=".writing-", suffix=".html", dir=html_path.parent
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(html)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, html_path)
            finally:
                tmp.unlink(missing_ok=True)
            print("report:", args.output_dir / "multichart-summary.json", flush=True)
            print("viewer:", html_path, flush=True)


if __name__ == "__main__":
    main()
