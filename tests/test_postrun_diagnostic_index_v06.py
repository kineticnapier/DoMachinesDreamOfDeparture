from __future__ import annotations

import random
from math import nextafter

from dmdod.planning.postrun_diagnostic_index_v06 import PostrunDiagnosticIndex


def legacy(keydowns, trace, axes, ordinals, due_ms):
    nearby = [value for value in keydowns
              if value["axis"] in axes and abs(value["time_ms"] - due_ms) <= 35.0]
    matched = [min((value for value in nearby if value["axis"] == axis),
                   key=lambda value: abs(value["time_ms"] - due_ms))
               for axis in axes if any(value["axis"] == axis for value in nearby)]
    early = [value for value in trace
             if value["target_index"] in ordinals and value["margin"] is None
             and value["time_s"] * 1000.0 < due_ms]
    return matched, len(early)


def test_duplicate_axes_ties_and_boundary_rounding():
    due = 100.1
    keydowns = [
        {"axis": "a", "time_ms": due + 10.0, "contact": "first"},
        {"axis": "a", "time_ms": due - 10.0, "contact": "second"},
        {"axis": "b", "time_ms": nextafter(due + 35.0, float("inf")), "contact": "edge"},
        {"axis": "b", "time_ms": due + 35.0, "contact": "within"},
        {"axis": "c", "time_ms": float("nan"), "contact": "nan"},
    ]
    trace = [
        {"target_index": 2, "margin": None, "time_s": .1},
        {"target_index": 2, "margin": None, "time_s": .101},
        {"target_index": 3, "margin": "hit", "time_s": .099},
        {"target_index": 3, "margin": None, "time_s": .099},
    ]
    idx = PostrunDiagnosticIndex(keydowns, trace)
    for axes, ordinals in ((["a", "b", "a", "c"], [2, 2, 3]),
                           (["b"], [2]), ([], [])):
        assert (idx.match_keydowns(axes, due), idx.count_too_early(ordinals, due)) == legacy(
            keydowns, trace, axes, ordinals, due)


def test_randomized_old_new_diagnostic_equivalence():
    rng = random.Random(212)
    axes = ("a", "b", "c", "d")
    keydowns = [dict(axis=rng.choice(axes), time_ms=round(rng.uniform(-10, 1200), 3), contact=i)
                for i in range(1000)]
    trace = [dict(target_index=rng.randrange(60), margin=rng.choice((None, None, "hit")),
                  time_s=round(rng.uniform(-.01, 1.2), 6)) for _ in range(900)]
    idx = PostrunDiagnosticIndex(keydowns, trace)
    for _ in range(500):
        chosen_axes = [rng.choice(axes) for _ in range(rng.randrange(5))]
        ordinals = [rng.randrange(60) for _ in range(rng.randrange(5))]
        due = round(rng.uniform(-50, 1250), 3)
        assert (idx.match_keydowns(chosen_axes, due),
                idx.count_too_early(ordinals, due)) == legacy(
                    keydowns, trace, chosen_axes, ordinals, due)


def test_empty_inputs_and_strict_early_boundary():
    idx = PostrunDiagnosticIndex([], [{"target_index": 1, "margin": None, "time_s": .1}])
    assert idx.match_keydowns(["a"], 100.0) == []
    assert idx.count_too_early([1, 1], 100.0) == 0
    assert idx.count_too_early([1], 100.001) == 1
