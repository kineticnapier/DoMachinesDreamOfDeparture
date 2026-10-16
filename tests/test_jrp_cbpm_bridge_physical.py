"""Motor acts on DMDOD physics; overlay observations are text only."""
from __future__ import annotations
from scripts.probe_jrp_cbpm_physical_v1 import probe


def test_real_accepted_hit_with_one_frame_late_jrp_update_anchors():
    result=probe(2.24)
    assert result['hits']==5
    assert result['second_hook'][0] == 251
    assert any(abs(due-3.)<.00005 for _,due,_ in result['anchors'])


def test_real_late_hit_does_not_invent_earlier_anchored_epoch():
    result=probe(2.28)
    assert result['hits']==5
    assert result['second_hook'][0] == 255
    assert not any(abs(due-3.)<.00005 for _,due,_ in result['anchors'])
