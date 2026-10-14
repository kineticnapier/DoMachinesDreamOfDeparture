from __future__ import annotations

from types import SimpleNamespace

import pytest

from dmdod.training import prestige1 as p
from dmdod.training import prestige1_alignment_audit as audit


def targets(*times):
    return [SimpleNamespace(episode_time_s=t, floor_index=1) for t in times]


def test_event_target_matches_unresolved_before_any_hit():
    data = targets(.1, .2)
    result = audit.compare_targets(data, [False]*2, [False]*2, .05, 1)
    assert result.kind == "aligned"
    assert result.legacy_index == result.unresolved_index == 0
    assert result.legacy_labels == result.unresolved_labels
    assert result.legacy_mask and result.unresolved_mask


def test_early_scored_note_remains_legacy_schedule_target():
    data = targets(.12, .21)
    result = audit.compare_targets(
        data, [True, False], [False, False], .1, 1)
    assert result.kind == "legacy_points_behind_unresolved"
    assert result.legacy_points_to_resolved
    assert result.legacy_index == 0 and result.unresolved_index == 1
    assert result.time_delta_ms == pytest.approx(90)
    assert result.legacy_labels[0] == pytest.approx(.02)
    assert result.unresolved_labels[0] == pytest.approx(.11)
    assert result.legacy_labels[1] == result.unresolved_labels[1] == 1.


def test_legacy_can_skip_still_unresolved_overdue_note():
    data = targets(.10, .5)
    result = audit.compare_targets(
        data, [False, False], [False, False], .12, 1)
    assert result.kind == "legacy_skips_unresolved"
    assert result.unresolved_overdue
    assert result.unresolved_mask is False
    assert result.legacy_index == 1 and result.unresolved_index == 0
    assert result.time_delta_ms == pytest.approx(-400)


def test_unresolved_chord_shrinks_after_one_early_hit():
    data = targets(.15, .15, .3)
    result = audit.compare_targets(
        data, [True, False, False], [False]*3, .11, 1)
    assert result.legacy_index == 0
    assert result.unresolved_index == 1
    assert result.legacy_labels[2] == 1.
    assert result.unresolved_labels[2] == 0.


def test_no_future_target_after_all_resolved():
    data = targets(.05, .07)
    result = audit.compare_targets(
        data, [True, True], [False, False], .1, 1)
    assert result.kind == "aligned"
    assert result.legacy_index is result.unresolved_index is None
    assert result.legacy_mask is False
    assert result.unresolved_mask is False


def test_refuse_existing_output_before_touching_sources(monkeypatch, tmp_path):
    config = p.PrestigeConfig(dataset="bad", output_dir=str(tmp_path),
                              core_path="missing", old_best_path="missing",
                              device="cpu")
    dest = audit.transfer.output_dir(config)
    dest.mkdir(parents=True)
    path = dest / audit.REPORT
    path.write_text("do not replace")
    def unexpected(*args, **kwargs):
        raise AssertionError("should not read any sources")
    monkeypatch.setattr(p, "_protect_old_best", unexpected)
    with pytest.raises(FileExistsError):
        audit.run_train_alignment_audit(config)
    assert path.read_text() == "do not replace"
