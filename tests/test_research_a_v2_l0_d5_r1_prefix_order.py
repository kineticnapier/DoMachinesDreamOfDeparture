"""D5: fixed process-order comparison boundaries and conservative classification."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from a002v2_l0_d5_r1_prefix_order import (
    ARMS_SEEDS, R1_RUN, R2_RUN, D3_RUN, classify,
    physical_summary, save_new_json,
)


def test_sources_and_fixed_predecessor_order():
    assert (R1_RUN, R2_RUN, D3_RUN) == (
        41309837474, 41312136897, 41369778494)
    assert ARMS_SEEDS == [
        ("a0", 0), ("a0", 1), ("a1", 0), ("a1", 1)]


def test_classification_requires_original_baseline():
    assert classify(
        baseline_matches_original=False, predecessor_exact=True,
        initial_digest_equal=True, first_batch_equal=False,
        preceded_matches_d3=True) == "invalid_diagnostic"


def test_classification_requires_four_exact_r1_and_initial_digest():
    for predecessor_exact, digest_equal in ((False, True), (True, False)):
        assert classify(
            baseline_matches_original=True,
            predecessor_exact=predecessor_exact,
            initial_digest_equal=digest_equal, first_batch_equal=False,
            preceded_matches_d3=True) == "invalid_diagnostic"


def test_exact_match_does_not_claim_effect():
    assert classify(
        baseline_matches_original=True, predecessor_exact=True,
        initial_digest_equal=True, first_batch_equal=True,
        preceded_matches_d3=True) == "no_r1_predecessor_effect_observed"


def test_effect_without_d3_reproduction_is_not_causal_conclusion():
    assert classify(
        baseline_matches_original=True, predecessor_exact=True,
        initial_digest_equal=True, first_batch_equal=False,
        preceded_matches_d3=False) == (
            "r1_predecessor_effect_but_d3_not_fully_reproduced")


def test_exact_d3_match_requires_different_batches():
    assert classify(
        baseline_matches_original=True, predecessor_exact=True,
        initial_digest_equal=True, first_batch_equal=False,
        preceded_matches_d3=True) == (
            "r1_predecessor_state_reproduces_archived_d3_divergence")


def test_physical_keydown_accounting():
    recorder = SimpleNamespace(events=[
        {"hit": True, "onset_step": 2, "onset_reused": False},
        {"hit": False, "onset_step": None, "onset_reused": False},
        {"hit": False, "onset_step": 7, "onset_reused": True},
    ])
    assert physical_summary(recorder) == {
        "total": 3, "hits": 1, "extras": 2,
        "unmatched_onsets": 1, "reused_onsets": 1}


def test_no_overwrite(tmp_path):
    path = tmp_path / "diagnostic.json"
    save_new_json(path, {"status": "first"})
    assert json.loads(path.read_text()) == {"status": "first"}
    with pytest.raises(FileExistsError):
        save_new_json(path, {"status": "second"})
