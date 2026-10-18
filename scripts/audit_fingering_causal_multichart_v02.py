from __future__ import annotations

"""Frozen Causal Lead v0.2 on chart-heldout evaluation candidates.

Discover evaluation charts in Test/Holdout/Eval/Validation (never Train) or
accept explicit --chart paths. Deterministic selection happens *before* any
Motor execution; outcomes never influence chart selection.

For each chosen chart run fresh Free/Human-Proxy × baseline/causal_lead with
the same, unchanged checkpoint, oracle timestamps, Motor physics and Lead
constants as the confirmed bool experiment. Do not load any saved allocations.

This is chart generalization under privileged timing, NOT an independent
HUD-only Timing System or human physiology evaluation.
"""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_fingering_causal_early_v02 as causal
import build_fingering_replay_viewer_v01 as viewer

from dmdod.adofai.chart import load_adofai
from dmdod.adofai.playable import build_playable_segment
from dmdod.adofai.timing import compile_adofai


EVALUATION_SPLITS = ("test", "holdout", "evaluation", "eval", "validation")
REJECTED_TRAIN_NAMES = frozenset(("train", "training"))
PREVIOUS_BOOL_SHA256 = "04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1"
POLICIES = causal.POLICIES
ARMS = ("baseline", "causal_lead")


def split_name(path: Path) -> str:
    names = [part.lower() for part in path.parts[:-1]]
    if any(part in REJECTED_TRAIN_NAMES for part in names):
        return "train"
    for name in EVALUATION_SPLITS:
        if name in names:
            return name
    return "unspecified"


def equally_spaced(paths: list[Path], number: int) -> list[Path]:
    """Select deterministically without looking at score or chart content."""
    if number <= 0:
        raise ValueError("maximum chart count must be positive")
    count = min(number, len(paths))
    return [paths[(2 * i + 1) * len(paths) // (2 * count)]
            for i in range(count)]


def select_charts(root: Path, explicit: list[Path],
                  max_charts: int) -> tuple[list[dict], dict]:
    if max_charts < 1:
        raise ValueError("--max-charts must be >= 1")
    if explicit:
        ordered = [p.resolve() for p in explicit]
        basis = "explicit"
        allowed_splits = ("test", "holdout", "evaluation", "eval", "validation", "unspecified")
    else:
        if not root.is_dir():
            raise FileNotFoundError(
                f"evaluation dataset root missing: {root}. Use --chart PATH."
            )
        candidates = sorted((p.resolve() for p in root.rglob("*.adofai")
                             if p.is_file()), key=lambda p: str(p).lower())
        chosen_split = next((split for split in EVALUATION_SPLITS
                             if any(split_name(p) == split for p in candidates)), None)
        if chosen_split is None:
            raise FileNotFoundError(
                f"no recognized non-Train evaluation folders under {root}. "
                "Expected Test, Holdout, Eval, Evaluation or Validation; "
                "use --chart PATH to name a non-Train chart explicitly."
            )
        ordered = [p for p in candidates if split_name(p) == chosen_split]
        basis = f"automatic:{chosen_split}"
        allowed_splits = (chosen_split,)

    if not ordered:
        raise ValueError("no chart candidates supplied")
    checks = []
    seen_paths = set()
    for path in ordered:
        if not path.is_file() or path.suffix.lower() != ".adofai":
            raise FileNotFoundError(f"ADOFAI chart not found: {path}")
        split = split_name(path)
        if split == "train":
            raise ValueError(f"Train chart forbidden in independent evaluation: {path}")
        if split not in allowed_splits:
            raise ValueError(f"unexpected evaluation split: {path} ({split})")
        if path in seen_paths:
            raise ValueError(f"duplicate chart path in evaluation pool: {path}")
        seen_paths.add(path)
        checks.append({
            "path": str(path),
            "name": path.name,
            "split": split,
        })
    selected = (equally_spaced(checks, max_charts) if not explicit
                else checks[:max_charts])
    # Hash only preselected charts. A large chart corpus may contain thousands
    # of non-selected files, so hashing its entire contents is unnecessary.
    seen_hashes = set()
    for row in selected:
        digest = viewer.file_sha256(Path(row["path"]))
        if digest == PREVIOUS_BOOL_SHA256:
            raise ValueError(f"previously evaluated bool chart forbidden: {row['path']}")
        if digest in seen_hashes:
            raise ValueError(f"duplicate chart content in selected sample: {row['path']}")
        seen_hashes.add(digest)
        row["sha256"] = digest
        row["provenance"] = (
            "heldout_directory_name" if row["split"] != "unspecified"
            else "externally_supplied_unverified_split"
        )
    return selected, {
        "selection_basis": basis,
        "population_n": len(checks),
        "selection_max_charts": max_charts,
        "selected_n": len(selected),
        "full_ordered_candidate_manifest": checks,
        "selected_hashes": [row["sha256"] for row in selected],
        "selection_rule": ("equally_spaced_sorted_paths_before_scoring" if not explicit
                           else "explicit_paths_in_command_line_order_before_scoring"),
    }

def _summarize_run(run: dict) -> dict:
    stats = run["stats"]
    return {
        "hits": stats["hits"],
        "targets": stats["targets"],
        "misses": stats["misses"],
        "x_accuracy_percent": stats["x_accuracy_percent"],
        "timing_mae_ms": stats["timing_mae_ms"],
        "too_early": stats["too_early"],
        "overloaded": stats["overloaded"],
        "physical_keydowns": stats["physical_keydowns"],
        "decisions": len(run["decisions"]),
        "unassigned": sum(x["chosen"] is None for x in run["decisions"]),
        "termination_reason": run["termination_reason"],
        "elapsed_wall_s": run["elapsed_wall_s"],
        "actual_early_axis_steps": run["interventions"].get("actual_early_axis_steps", 0),
        "extra_lead_allocations": run["interventions"].get("extra_lead_allocations", 0),
    }


def compare_pair(baseline: dict, lead: dict) -> dict:
    a, b = baseline["stats"], lead["stats"]
    old = {int(d["event_index"]): d["chosen"] for d in baseline["decisions"]}
    changed = sum(d["chosen"] != old.get(int(d["event_index"]))
                  for d in lead["decisions"])
    def delta(key):
        return None if a[key] is None or b[key] is None else round(b[key] - a[key], 6)
    return {
        "baseline": _summarize_run(baseline),
        "causal_lead": _summarize_run(lead),
        "delta_accuracy_pp": delta("x_accuracy_percent"),
        "delta_timing_mae_ms": delta("timing_mae_ms"),
        "delta_hits": b["hits"] - a["hits"],
        "delta_misses": b["misses"] - a["misses"],
        "delta_too_early": b["too_early"] - a["too_early"],
        "delta_physical_keydowns": b["physical_keydowns"] - a["physical_keydowns"],
        "changed_finger_choices": changed,
        "common_decisions": sum(int(d["event_index"]) in old for d in lead["decisions"]),
        "future_saved_allocation_consulted": False,
    }


def aggregate(results: list[dict]) -> dict:
    output = {}
    for policy, _ in POLICIES:
        pairs = [r["policies"][policy] for r in results
                 if "policies" in r and "error" not in r["policies"].get(policy, {})
                 and policy in r["policies"]]
        if not pairs:
            output[policy] = {"completed_charts": 0}
            continue
        diffs = [p["delta_accuracy_pp"] for p in pairs
                 if p["delta_accuracy_pp"] is not None]
        def sums(arm, metric):
            return sum(p[arm][metric] for p in pairs)
        output[policy] = {
            "completed_charts": len(pairs),
            "total_targets_baseline": sums("baseline", "targets"),
            "total_hits_baseline": sums("baseline", "hits"),
            "total_hits_causal_lead": sums("causal_lead", "hits"),
            "total_misses_baseline": sums("baseline", "misses"),
            "total_misses_causal_lead": sums("causal_lead", "misses"),
            "total_too_early_baseline": sums("baseline", "too_early"),
            "total_too_early_causal_lead": sums("causal_lead", "too_early"),
            "accuracy_delta_macro_mean_pp": (
                round(sum(diffs) / len(diffs), 6) if diffs else None
            ),
            "changed_finger_choices": sum(p["changed_finger_choices"] for p in pairs),
            "full_hit_charts_baseline": sum(p["baseline"]["hits"] == p["baseline"]["targets"]
                                            for p in pairs),
            "full_hit_charts_causal_lead": sum(p["causal_lead"]["hits"] == p["causal_lead"]["targets"]
                                               for p in pairs),
        }
    return output


def evaluate(checkpoint: Path, selected: list[dict],
             wall_timeout_s: float) -> dict:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    ck_sha = viewer.file_sha256(checkpoint)
    model = causal.runner.load_model(checkpoint)
    output = []
    for ix, chart_info in enumerate(selected, start=1):
        path = Path(chart_info["path"])
        print(f"\n[{ix}/{len(selected)}] {path.name} ({chart_info['split']})", flush=True)
        entry = dict(chart_info)
        entry["policies"] = {}
        try:
            segment = build_playable_segment(compile_adofai(load_adofai(path)))
            entry["duration_s"] = segment.duration_s
        except Exception as exc:
            entry["chart_load_error"] = f"{type(exc).__name__}: {exc}"
            print(f"  chart load error: {entry['chart_load_error']}", flush=True)
            output.append(entry)
            continue
        for policy, alpha in POLICIES:
            try:
                baseline = causal.execute(segment, model, policy, alpha,
                                          "baseline", wall_timeout_s=wall_timeout_s)
                lead = causal.execute(segment, model, policy, alpha,
                                      "causal_lead", wall_timeout_s=wall_timeout_s)
                pair = compare_pair(baseline, lead)
                entry["policies"][policy] = pair
                print(f"  {policy}: baseline={pair['baseline']['hits']}/{pair['baseline']['targets']} "
                      f"X={pair['baseline']['x_accuracy_percent']}%; "
                      f"causal={pair['causal_lead']['hits']}/{pair['causal_lead']['targets']} "
                      f"X={pair['causal_lead']['x_accuracy_percent']}%; "
                      f"miss={pair['causal_lead']['misses']} "
                      f"TE={pair['causal_lead']['too_early']} "
                      f"OL={pair['causal_lead']['overloaded']}", flush=True)
            except Exception as exc:
                entry["policies"][policy] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  {policy} error: {type(exc).__name__}: {exc}", flush=True)
        output.append(entry)
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": ck_sha,
        "charts": output,
        "aggregate": aggregate(output),
    }


def render_html(report: dict) -> str:
    lines = []
    for chart in report["charts"]:
        for policy, _ in POLICIES:
            result = chart.get("policies", {}).get(policy, {})
            for arm in ARMS:
                a = result.get(arm, {})
                values = (
                    chart["name"], chart["split"], policy, arm,
                    a.get("hits"), a.get("targets"), a.get("misses"),
                    a.get("x_accuracy_percent"), a.get("timing_mae_ms"),
                    a.get("too_early"), a.get("overloaded"),
                    a.get("physical_keydowns"), a.get("unassigned"),
                    a.get("actual_early_axis_steps"),
                    result.get("changed_finger_choices"),
                    result.get("error", chart.get("chart_load_error")),
                )
                lines.append("<tr>" + "".join(
                    f"<td>{escape(str(v) if v is not None else '—')}</td>" for v in values
                ) + "</tr>")
    return (
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>Causal Lead — chart generalization</title>"
        "<style>body{max-width:1600px;margin:28px auto;padding:0 14px;"
        "font:14px/1.5 system-ui;background:#111825;color:#ecf2fc}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border-bottom:1px solid #3e4c63;padding:8px;text-align:left}"
        "th{position:sticky;top:0;background:#1d2d44}</style>"
        "<h1>Fingering v0.2 — 別譜面汎化評価</h1>"
        f"<p>事前選定: {escape(report['selection']['selection_basis'])}, "
        f"{report['selection']['selected_n']}譜面。学習済みモデルと制御定数は固定。</p>"
        "<table><thead><tr><th>譜面</th><th>分割</th><th>運指</th><th>条件</th>"
        "<th>Hit</th><th>Targets</th><th>Miss</th><th>X%</th><th>MAE ms</th>"
        "<th>TE</th><th>OL</th><th>KeyDown</th><th>未割当</th>"
        "<th>Early steps</th><th>選択変更数</th><th>エラー</th></tr></thead><tbody>"
        + "".join(lines) + "</tbody></table>"
        "<p>既知のbool譜面を排除。Test等のディレクトリ名は完全な分割独立性の証明ではない。"
        "AutoPlay OracleのTimingを使用しておりHUD-onlyの評価ではない。</p>"
        "</html>\n"
    )


def run(root: Path, explicit: list[Path], max_charts: int, checkpoint: Path,
        output_dir: Path, wall_timeout_s: float) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    selected, manifest = select_charts(root, explicit, max_charts)
    print(f"selection: {manifest['selection_basis']}; "
          f"{len(selected)} of {manifest['population_n']} chart(s)", flush=True)
    evaluated = evaluate(checkpoint, selected, wall_timeout_s)
    report = {
        "format": "dmdod-fingering-v02-frozen-causal-lead-multichart",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": manifest,
        "checkpoint": evaluated["checkpoint"],
        "checkpoint_sha256": evaluated["checkpoint_sha256"],
        "control_constants": {
            "decision_lookahead_ms": (causal.teacher.LEAD + causal.EXTRA_LEAD_S) * 1000,
            "baseline_lookahead_ms": causal.teacher.LEAD * 1000,
            "max_activation_extra_lead_ms": causal.EXTRA_LEAD_S * 1000,
            "activation_gate": causal.NEGATIVE_ACTIVATION_GATE,
            "control_dt_ms": causal.teacher.CTRL * 1000,
            "physics_dt_ms": causal.teacher.PHYS * 1000,
            "model_retrained": False,
            "Motor_constants_modified": False,
            "Timing_oracle_used": True,
            "HUD_only_test": False,
        },
        "charts": evaluated["charts"],
        "aggregate": evaluated["aggregate"],
        "limitations": [
            "The frozen Network was trained on synthetic pattern-series splits, not these whole charts.",
            "Directory naming and chart-hash exclusion do not prove absence of every possible training overlap.",
            "Selection occurs before evaluating, but scores must not be used to cherry-pick successes.",
            "Both baseline and Causal Lead are newly inferred on each chart using current Motor state.",
            "Future saved finger allocations are never supplied to the two arms.",
            "Known bool-chart SHA256 and all Train directories are excluded.",
            "Causal Lead changes both inference timing and Motor commands; not a pure controller-only ablation.",
            "Oracle event timestamps are privileged; HUD-only Timing generalization remains untested.",
            "Errors and partial chart failures are retained, not silently excluded from the report.",
        ],
    }
    html = render_html(report)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "multichart-summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with (output_dir / "multichart-summary.html").open("x", encoding="utf-8") as f:
        f.write(html)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chart-root", type=Path,
                   default=Path("data/DMDOD-v140-tuf-p11-p20"))
    p.add_argument("--chart", type=Path, action="append", default=[],
                   help="Explicit non-Train chart; repeat to test multiple charts")
    p.add_argument("--max-charts", type=int, default=3)
    p.add_argument("--checkpoint", type=Path, default=causal.CHECKPOINT)
    p.add_argument("--wall-timeout-s", type=float, default=600.0)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.wall_timeout_s <= 0.0:
        p.error("--wall-timeout-s must be > 0")
    result = run(args.chart_root, args.chart, args.max_charts,
                 args.checkpoint, args.output_dir, args.wall_timeout_s)
    print("\naggregate:", json.dumps(result["aggregate"], ensure_ascii=False))
    print("report:", args.output_dir / "multichart-summary.json")
    print("viewer:", args.output_dir / "multichart-summary.html")


if __name__ == "__main__":
    main()
