from __future__ import annotations

import random
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import audit_dmdogo_h20_human_plan_alignment as h20


def brute_force_cost(h, ref, offset, gap=2.75, tol=.05, sigma=.028):
    n, m = len(h), len(ref)
    dp = [[float("inf")] * (m+1) for _ in range(n+1)]
    dp[0][0] = 0.
    for i in range(1, n+1):
        dp[i][0] = i*gap
    for j in range(1, m+1):
        dp[0][j] = j*gap
    for i in range(1, n+1):
        for j in range(1, m+1):
            score = min(dp[i-1][j]+gap, dp[i][j-1]+gap)
            diff = h[i-1] - ref[j-1] - offset
            if abs(diff) <= tol:
                score = min(score, dp[i-1][j-1] + (diff/sigma)**2)
            dp[i][j] = score
    return dp[n][m]


def test_alignment_matches_dense_dynamic_programming():
    rng = random.Random(42)
    for _ in range(60):
        h = sorted(rng.random() for _ in range(rng.randint(1, 15)))
        d = sorted(rng.random() for _ in range(rng.randint(1, 15)))
        offset = rng.uniform(-.05, .05)
        cost, pairs = h20.sparse_align(h, d, offset)
        assert cost == pytest.approx(brute_force_cost(h, d, offset), abs=1e-7)
        assert len(set(i for i,j in pairs)) == len(pairs)
        assert len(set(j for i,j in pairs)) == len(pairs)
        assert pairs == sorted(pairs)


def test_raw_log_keeps_fast_same_key_input():
    log = "# Fingering sequence\n# Active input mapping:\n"
    log += "time_ms    hits\n"
    log += "73783.1 K10:RM\n73795.0 K10:RM\n"
    output = h20.parse_human(log)
    assert len(output) == 2
    assert output[0]["hand"] == "R"
    assert output[1]["time_s"] - output[0]["time_s"] == pytest.approx(.0119)


def test_reject_nonhand_or_out_of_order_input():
    with pytest.raises(ValueError, match="unmapped"):
        h20.parse_human("1.0 K01:F1")
    with pytest.raises(ValueError, match="not chronological"):
        h20.parse_human("2.0 K09:RI\n1.0 K04:LI")
    with pytest.raises(ValueError, match="invalid log"):
        h20.parse_human("nonsense")


def test_skipping_unmatchable_is_not_miss_claim():
    cost, matched = h20.sparse_align([100., 101.], [1., 2.], 0.)
    assert matched == []
    assert cost == 11.
