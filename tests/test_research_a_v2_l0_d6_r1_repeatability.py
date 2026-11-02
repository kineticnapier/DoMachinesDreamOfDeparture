"""D6 archival audit must not invent causality or overwrite existing results."""
from pathlib import Path
import json
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from a002v2_l0_d6_r1_repeatability import (
    SOURCE_RUNS, MODEL_KEYS, classify, digest, inventory,
    parse_progress, progress_differences, save_new_json, tensor_differences,
)


def test_fixed_sources_and_models():
    assert SOURCE_RUNS == {'r1': 41309837474, 'd3': 41369778494,
                           'd5a': 41380105087, 'd5b': 41380334940}
    assert MODEL_KEYS == [('a0', 0), ('a0', 1), ('a1', 0), ('a1', 1)]


def test_classification_requires_both_equal_archives():
    assert classify(d5_runs_equal=True, d5_matches_r1=False) == 'repeatable_archived_divergence'
    assert classify(d5_runs_equal=False, d5_matches_r1=False) == 'd5_runs_disagree'
    assert classify(d5_runs_equal=True, d5_matches_r1=True) == 'no_archived_r1_divergence'


def test_exact_tensor_counts_and_max_difference():
    a = {'weight': torch.tensor([0., 1., 2.]), 'bias': torch.tensor([4.])}
    b = {'weight': torch.tensor([0., 1.25, 2.]), 'bias': torch.tensor([5.])}
    diff = tensor_differences(a, b)
    assert diff['elements'] == 4 and diff['changed'] == 2
    assert diff['max_abs_difference'] == 1.0
    assert diff['layers']['weight']['changed'] == 1


def test_progress_uses_only_logged_32_episode_boundaries():
    rows = [{'episode': i, 'arm': 'a0', 'seed': 0, 'entropy': float(i),
             'training_hit_rate': 0.0, 'eval_hit_rate': None}
            for i in range(32, 257, 32)]
    log = '\n'.join('LOW_SPEED_PROGRESS ' + json.dumps(row) for row in rows)
    parsed = parse_progress(log)
    assert len(parsed) == 8
    source = [{'episode': i, 'entropy': float(i), 'train_hit_rate': 0.0}
              for i in range(8, 257, 8)]
    assert progress_differences(source, parsed) == []
    source[15]['entropy'] = -1.0
    assert progress_differences(source, parsed)[0]['episode'] == 128


def test_progress_missing_row_rejected():
    with pytest.raises(ValueError):
        parse_progress('LOW_SPEED_PROGRESS ' + json.dumps({'episode': 32, 'arm': 'a0', 'seed': 0}))


def test_inventory_hashes_file_bytes(tmp_path):
    (tmp_path / 'x.txt').write_bytes(b'abc')
    assert inventory(tmp_path) == {'x.txt': digest(b'abc')}


def test_no_overwrite(tmp_path):
    p = tmp_path / 'result.json'
    save_new_json(p, {'status': 'first'})
    with pytest.raises(FileExistsError):
        save_new_json(p, {'status': 'second'})
    assert json.loads(p.read_text()) == {'status': 'first'}
