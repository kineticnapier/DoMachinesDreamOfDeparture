from dataclasses import dataclass

import pytest

from dmdod.evaluation.jrp_unseen_split_v2 import select_unseen_charts


@dataclass(frozen=True)
class Chart:
    content_sha256: str


def test_split_excludes_trained_and_is_disjoint_deterministic_20s():
    trained={f'{i:03d}' for i in range(20)}
    charts=[Chart(f'{i:03d}') for i in reversed(range(80))]
    dev=select_unseen_charts(charts,trained,'development')
    qual=select_unseen_charts(charts,trained,'qualification')
    reserve=select_unseen_charts(charts,trained,'reserve')
    assert [x.content_sha256 for x in dev]==[f'{i:03d}' for i in range(20,40)]
    assert [x.content_sha256 for x in qual]==[f'{i:03d}' for i in range(40,60)]
    assert [x.content_sha256 for x in reserve]==[f'{i:03d}' for i in range(60,80)]
    assert not (set(dev)&set(qual) or set(dev)&set(reserve) or set(qual)&set(reserve))


def test_split_fails_closed_on_wrong_cardinality_or_name():
    with pytest.raises(RuntimeError,match='exactly 60'):
        select_unseen_charts([Chart(f'{i:03d}') for i in range(59)],set(),'development')
    with pytest.raises(ValueError,match='partition'):
        select_unseen_charts([],set(),'final')
