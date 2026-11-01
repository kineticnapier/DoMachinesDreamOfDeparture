from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dmdod.planning.gesture_chunk_planner_v08 import GestureChunk, GestureEvent, GesturePlan
from run_dmdogo_h23_frozen_hands import propose_axes, create_intervened_plan, gap_bin, FROZEN_SWITCH


def events(times):
    return tuple(GestureEvent(i,t,1) for i,t in enumerate(times))


def test_frozen_bins_and_training_source():
    assert [gap_bin(v) for v in (15, 30, 61, 121, 182, 400)] == list(FROZEN_SWITCH)
    assert list(FROZEN_SWITCH.values()) == [False,False,True,True,False,False]


def test_no_reuse_under_80ms_for_fast_same_hand_sequence():
    e=events([1,1.01516,1.03032,1.04548,1.06064,1.0909])
    axes,reasons=propose_axes(e,minimum_reuse_s=.080,variant='interval_only')
    keys=[x[0] for x in axes]
    for i,key in enumerate(keys):
        assert all(e[i].due_s-e[j].due_s >= .080-1e-8 for j in range(i) if keys[j]==key)
    assert len(set(keys[:5])) >= 4
    assert 'physical_cooldown_forced_other_hand' in reasons


def test_94ms_switch_and_182ms_stay():
    e=events([1,1.094,1.276])
    axes,reasons=propose_axes(e,minimum_reuse_s=.080,variant='interval_only')
    assert axes==(('right_1',),('left_1',),('left_1',))


def test_predeclared_reset_only_changes_long_rest():
    e=events([1,1.094,1.8])
    no,_=propose_axes(e,minimum_reuse_s=.080,variant='interval_only')
    yes,_=propose_axes(e,minimum_reuse_s=.080,variant='right_after_long_rest')
    assert no[-1]==('left_1',)
    assert yes[-1]==('right_1',)


def test_reject_chords_and_unsorted_events():
    with pytest.raises(ValueError,match='single-contact'):
        propose_axes((GestureEvent(0,1.,2),),minimum_reuse_s=.080,variant='interval_only')
    with pytest.raises(ValueError,match='strictly increasing'):
        propose_axes(events([1,1]),minimum_reuse_s=.080,variant='interval_only')


def test_intervention_retains_original_segmentation_and_marks_assumptions():
    e=events([1,1.094,1.276])
    base=GesturePlan(True,(('left_1',),('right_1',),('right_1',)),(
        GestureChunk(0,1,2,'regular','start','interleaved',(('left_1',),('right_1',))),
        GestureChunk(2,2,1,'regular','gap','single',(('right_1',),))),
        None,10,{'beam_cost':2},{'source':'original'})
    alt=create_intervened_plan(e,base,minimum_reuse_s=.080,variant='interval_only')
    assert alt.assignments==(('right_1',),('left_1',),('left_1',))
    assert [(x.start_event_index,x.end_event_index,x.event_count) for x in alt.chunks]==[(0,1,2),(2,2,1)]
    assert alt.model_contract['human_data_on_target_chart_used'] is False
    assert alt.model_contract['offline_oracle_timing_used'] is True
    assert alt.model_contract['physical_motor_verified'] is False
