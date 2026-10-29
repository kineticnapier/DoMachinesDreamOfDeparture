from __future__ import annotations

from itertools import permutations

from dmdod.planning.judgement_label_orders import (
    MAX_ORDER_EXPANSIONS,
    bounded_unique_orders,
    tied_input_orders_exceed_budget,
)


def test_unique_orders_match_complete_small_permutation_set():
    for labels in ((), ("a",), ("b", "a", "a"), ("b", "a", "c", "a")):
        orders = bounded_unique_orders(labels)
        assert orders is not None
        assert orders == tuple(sorted(set(permutations(labels))))


def test_twelve_identical_visible_labels_have_one_order():
    labels = ("perfect",) * 12
    assert bounded_unique_orders(labels) == (labels,)


def test_budget_and_tied_input_preflight_abstain_without_expansion():
    assert bounded_unique_orders(tuple("abcdefg")) is None  # 7! > 4096
    assert bounded_unique_orders(("perfect",) * 65) is None
    assert bounded_unique_orders(("a", "b"), limit=1) is None
    assert bounded_unique_orders(("a", "b"), limit=2) is not None
    assert tied_input_orders_exceed_budget(8, 1)
    assert not tied_input_orders_exceed_budget(4, 3)
    assert MAX_ORDER_EXPANSIONS == 4096
