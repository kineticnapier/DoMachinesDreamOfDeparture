#!/usr/bin/env python3
"""D6 retrospective audit of four archived artifacts; no training is performed."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch

PROTOCOL = "docs/research-a-v2-l0-d6-r1-repeatability-protocol.md"
SOURCE_RUNS = {"r1": 41309837474, "d3": 41369778494,
               "d5a": 41380105087, "d5b": 41380334940}
SOURCE_ARTIFACTS = {"r1": 13084957586, "d3": 13109230783,
                    "d5a": 13112861470, "d5b": 13112708284}
D5_HEADS = {"d5a": "8df77ed43f993d5c187614c95966705b97f471b2",
            "d5b": "b118bc6ee596bf5dffb08d1d0300489ff46b08b7"}
MODEL_KEYS = [("a0", 0), ("a0", 1), ("a1", 0), ("a1", 1)]
MODEL_NAME = "research-a-v2-l0-a0-seed0-ep256.pt"


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def inventory(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory missing: {root}")
    result = {str(p.relative_to(root)).replace(chr(92), '/'): digest(p.read_bytes())
              for p in sorted(root.rglob('*')) if p.is_file()}
    if not result:
        raise ValueError(f"empty artifact: {root}")
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not {'arm', 'seed', 'policy', 'optimizer'} <= data.keys():
        raise ValueError(f"not an expected model checkpoint: {path}")
    return data


def tensor_differences(left: dict, right: dict) -> dict:
    if set(left) != set(right):
        raise ValueError("checkpoint tensor names differ")
    layers = {}
    total = changed = 0
    max_abs = 0.0
    for name in sorted(left):
        a, b = left[name], right[name]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            raise ValueError("non-tensor checkpoint value")
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"checkpoint tensor shape/dtype differs: {name}")
        n = int(a.numel())
        diff = int(torch.count_nonzero(a != b).item())
        max_layer = float((a.to(torch.float64) - b.to(torch.float64)).abs().max().item())
        layers[name] = {"elements": n, "changed": diff, "max_abs_difference": max_layer}
        total += n
        changed += diff
        max_abs = max(max_abs, max_layer)
    return {"elements": total, "changed": changed,
            "max_abs_difference": max_abs, "layers": layers}


def optimizer_differences(a: dict, b: dict) -> dict:
    if a['param_groups'] != b['param_groups'] or set(a['state']) != set(b['state']):
        raise ValueError("optimizer structure differs")
    counts = {}
    for index in sorted(a['state']):
        left, right = a['state'][index], b['state'][index]
        if set(left) != set(right):
            raise ValueError("optimizer state keys differ")
        for name in sorted(left):
            x, y = left[name], right[name]
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                if x != y:
                    raise ValueError("non-tensor optimizer state differs")
                continue
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError("optimizer tensor shape/dtype differs")
            counts[name] = counts.get(name, 0) + int(torch.count_nonzero(x != y).item())
    return {"changed_tensor_elements_by_state": counts,
            "all_states_equal": not any(counts.values())}


def parse_progress(log: str) -> list[dict]:
    rows = []
    for line in log.splitlines():
        if line.startswith("LOW_SPEED_PROGRESS "):
            rows.append(json.loads(line[len("LOW_SPEED_PROGRESS "):]))
    if [r['episode'] for r in rows] != list(range(32, 257, 32)):
        raise ValueError("D5 progress must cover episodes 32..256 exactly once")
    if any((r['arm'], r['seed']) != ('a0', 0) for r in rows):
        raise ValueError("D5 progress is not R1 A0 seed0")
    return rows


def progress_differences(archived: list[dict], observed: list[dict]) -> list[dict]:
    original = {x['episode']: x for x in archived}
    if len(archived) != 32 or len(original) != 32:
        raise ValueError("original R1 training history must contain 32 updates")
    result = []
    for row in observed:
        ep = row['episode']
        expected = original[ep]
        fields = {'entropy': expected['entropy'],
                  'training_hit_rate': expected['train_hit_rate']}
        for key, val in fields.items():
            if row[key] != val:
                result.append({"episode": ep, "field": key,
                               "original": val, "d5": row[key]})
    return result


def classify(*, d5_runs_equal: bool, d5_matches_r1: bool) -> str:
    if not d5_runs_equal:
        return "d5_runs_disagree"
    if d5_matches_r1:
        return "no_archived_r1_divergence"
    return "repeatable_archived_divergence"


def save_new_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write('\n')


def audit(r1: Path, d3: Path, d5a: Path, d5b: Path) -> dict:
    roots = {'r1': r1, 'd3': d3, 'd5a': d5a, 'd5b': d5b}
    files = {key: inventory(root) for key, root in roots.items()}
    source = read_json(r1 / 'full-results.json')
    replay = read_json(d3 / 'full-diagnostic.json')
    originals = {(m['arm'], m['seed']): m for m in source['models']}
    d3models = {(m['arm'], m['seed']): m for m in replay['models'] if m['mode'] == 'r1'}
    if set(originals) != set(MODEL_KEYS) or set(d3models) != set(MODEL_KEYS):
        raise ValueError("R1 or D3 does not contain exactly the four original models")
    r1_replay = []
    for arm, seed in MODEL_KEYS:
        name = f'research-a-v2-l0-{arm}-seed{seed}-ep256.pt'
        a = f'checkpoints/{name}'
        b = f'replayed-checkpoints/r1/{name}'
        original, traced = originals[(arm, seed)], d3models[(arm, seed)]
        if not traced['original_reproduced'] or original['checkpoint_sha256'] != traced['replay_checkpoint_sha256']:
            raise ValueError(f"D3 did not reproduce original R1 {arm}/{seed}")
        if files['r1'][a] != files['d3'][b] or files['r1'][a] != original['checkpoint_sha256']:
            raise ValueError(f"D3/R1 checkpoint byte mismatch: {name}")
        r1_replay.append({"arm": arm, "seed": seed, "sha256": files['r1'][a],
                          "byte_exact": True})

    d5a_summary = read_json(d5a / 'baseline/summary.json')
    d5b_summary = read_json(d5b / 'baseline/summary.json')
    for item in (d5a_summary, d5b_summary):
        if (item['mode'] != 'baseline' or item['r2_arm'] != 'a0' or item['r2_seed'] != 0 or
                item['r2_episodes'] != 8 or item['r2_ppo_updates'] != 1 or
                item['original_stage0_difference'] is not None or
                item['original_update1_difference'] is not None):
            raise ValueError("D5 baseline R2 archive comparison invalid")
    checkpoint_rel = 'preceded/r1-replayed-checkpoints/' + MODEL_NAME
    original_rel = 'checkpoints/' + MODEL_NAME
    original_ckpt = load_checkpoint(r1 / original_rel)
    d5_ckpt = load_checkpoint(d5a / checkpoint_rel)
    if (original_ckpt['arm'], original_ckpt['seed']) != ('a0', 0) or (d5_ckpt['arm'], d5_ckpt['seed']) != ('a0', 0):
        raise ValueError("R1/D5 model identity mismatch")
    policy = tensor_differences(original_ckpt['policy'], d5_ckpt['policy'])
    optimizer = optimizer_differences(original_ckpt['optimizer'], d5_ckpt['optimizer'])
    progress_a = parse_progress((d5a / 'preceded.log').read_text(encoding='utf-8'))
    progress_b = parse_progress((d5b / 'preceded.log').read_text(encoding='utf-8'))
    changes = progress_differences(originals[('a0', 0)]['training'], progress_a)
    all_files_equal = files['d5a'] == files['d5b']
    unequal = sorted(k for k in set(files['d5a']) | set(files['d5b'])
                     if files['d5a'].get(k) != files['d5b'].get(k))
    if progress_a != progress_b and all_files_equal:
        raise AssertionError("identical D5 artifacts yielded different progress")
    d5_matches_r1 = files['d5a'][checkpoint_rel] == files['r1'][original_rel]
    return {
        "schema": "research-a-v2-l0-d6-archived-r1-repeatability-v1",
        "protocol": PROTOCOL, "type": "retrospective_archived_audit",
        "not_blind_or_new_intervention": True, "new_training_episodes": 0,
        "source_runs": SOURCE_RUNS, "source_artifacts": SOURCE_ARTIFACTS,
        "d5_heads": D5_HEADS,
        "classification": classify(d5_runs_equal=all_files_equal,
                                   d5_matches_r1=d5_matches_r1),
        "d5_all_files_byte_equal": all_files_equal,
        "d5_files_compared": len(files['d5a']),
        "d5_unequal_files": unequal,
        "artifact_file_sha256": files,
        "r1_vs_d3_all_four_checkpoints_exact": True,
        "r1_d3_checkpoints": r1_replay,
        "r1_original_checkpoint_sha256": files['r1'][original_rel],
        "d5_checkpoint_sha256": files['d5a'][checkpoint_rel],
        "r1_vs_d5_policy": policy,
        "r1_vs_d5_optimizer": optimizer,
        "d5_progress_points": [r['episode'] for r in progress_a],
        "d5_first_observed_progress_difference": changes[0] if changes else None,
        "d5_all_observed_progress_differences": changes,
        "r2_baseline_first_update_exact": True,
        "r2_baseline_physical_keydowns": d5a_summary['physical_keydowns'],
        "r2_d3_stage0_difference": d5a_summary['d3_stage0_difference'],
        "r2_d3_update1_difference": d5a_summary['d3_update1_difference'],
        "r2_historical_gradients_verified": False,
        "first_divergent_ppo_update_known": False,
        "cpu_or_order_cause_identified": False,
        "teacher_free_fingering_acquired": False,
        "formal_study_002_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r1', 'd3', 'd5a', 'd5b'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('refuse to overwrite existing D6 report')
    torch.set_num_threads(1)
    try:
        result = audit(args.r1, args.d3, args.d5a, args.d5b)
    except Exception as exc:
        save_new_json(args.out, {"schema": "research-a-v2-l0-d6-archived-r1-repeatability-v1",
                                 "status": "failed", "error_type": type(exc).__name__,
                                 "error": str(exc), "new_training_episodes": 0,
                                 "source_runs": SOURCE_RUNS, "formal_study_002_approved": False})
        raise
    save_new_json(args.out, result)
    print('D6_ARCHIVED_AUDIT', json.dumps({
        'classification': result['classification'],
        'd5_all_files_byte_equal': result['d5_all_files_byte_equal'],
        'd5_files_compared': result['d5_files_compared'],
        'r1_vs_d5_policy': {k: v for k, v in result['r1_vs_d5_policy'].items() if k != 'layers'},
        'first_observed_progress_difference': result['d5_first_observed_progress_difference'],
        'r2_baseline_physical_keydowns': result['r2_baseline_physical_keydowns'],
        'new_training_episodes': 0,
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
