import pytest

from dmdod.curriculum import curriculum_start_s, note_curriculum


def test_note_curriculum_builds_progressive_stages():
    assert note_curriculum(1) == (1,)
    assert note_curriculum(4) == (1, 2, 4)
    assert note_curriculum(16) == (1, 2, 4, 8, 16)
    assert note_curriculum(10) == (1, 2, 4, 8, 10)
    assert note_curriculum(32) == (1, 2, 4, 8, 16, 32)


def test_curriculum_start_shortens_small_stages():
    assert curriculum_start_s(1) == pytest.approx(0.450)
    assert curriculum_start_s(2) == pytest.approx(0.500)
    assert curriculum_start_s(4) == pytest.approx(0.600)
    assert curriculum_start_s(8) == pytest.approx(0.750)


def test_curriculum_rejects_non_positive_counts():
    with pytest.raises(ValueError):
        note_curriculum(0)
    with pytest.raises(ValueError):
        curriculum_start_s(0)
