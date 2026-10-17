from __future__ import annotations

"""Deterministic chart-disjoint partitions for post-checkpoint JRP research."""


PARTITION_SIZE = 20
PARTITIONS = ('development', 'qualification', 'reserve')


def select_unseen_charts(train_charts, trained_hashes, partition: str):
    """Exclude checkpoint Train20, sort by immutable SHA, then slice 20/20/20."""
    if partition not in PARTITIONS:
        raise ValueError('partition must be development, qualification, or reserve')
    trained = set(trained_hashes)
    remaining = sorted(
        (chart for chart in train_charts if chart.content_sha256 not in trained),
        key=lambda chart: chart.content_sha256)
    if len(remaining) != 3 * PARTITION_SIZE:
        raise RuntimeError(f'expected exactly 60 checkpoint-unseen Train charts, got {len(remaining)}')
    if trained & {chart.content_sha256 for chart in remaining}:
        raise RuntimeError('checkpoint Train chart leaked into JRP unseen split')
    start = PARTITIONS.index(partition) * PARTITION_SIZE
    selected = tuple(remaining[start:start + PARTITION_SIZE])
    if len({chart.content_sha256 for chart in selected}) != PARTITION_SIZE:
        raise RuntimeError('duplicate chart in JRP unseen split')
    return selected
