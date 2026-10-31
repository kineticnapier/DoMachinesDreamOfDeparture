from __future__ import annotations

"""Bounded deterministic order enumeration for HUD-only judgement research."""

from collections import Counter
from math import comb, factorial

# These limits change the estimator's abstention policy, not Stock judgement.
MAX_VISIBLE_EVENTS = 64
MAX_ORDER_EXPANSIONS = 4096


def bounded_unique_orders(
    labels: tuple[str, ...], *, limit: int = MAX_ORDER_EXPANSIONS
) -> tuple[tuple[str, ...], ...] | None:
    """Enumerate every distinct order, or abstain before excessive allocation."""
    if len(labels) > MAX_VISIBLE_EVENTS or limit < 1:
        return None
    count_seen = 0
    ways = 1
    for count in Counter(labels).values():
        ways *= comb(count_seen + count, count)
        count_seen += count
        if ways > limit:
            return None
    values = sorted(labels)
    result: list[tuple[str, ...]] = []
    while True:
        result.append(tuple(values))
        pivot = len(values) - 2
        while pivot >= 0 and values[pivot] >= values[pivot + 1]:
            pivot -= 1
        if pivot < 0:
            return tuple(result)
        successor = len(values) - 1
        while values[successor] <= values[pivot]:
            successor -= 1
        values[pivot], values[successor] = values[successor], values[pivot]
        values[pivot + 1:] = reversed(values[pivot + 1:])


def tied_input_orders_exceed_budget(
    count: int, label_order_count: int, *, limit: int = MAX_ORDER_EXPANSIONS
) -> bool:
    """Preflight factorial permutations of distinct, tied input IDs."""
    return factorial(count) * label_order_count > limit
