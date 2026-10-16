from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_nerfed_fingering_v06 import choose_swap
from dmdod.motor.contact_16k import ContactAction, Coupled16ContactMotor, PHYSICAL_AXES


def test_candidate_scores_record_selection_without_mutation():
    motor = Coupled16ContactMotor()
    before = motor.observe()
    action = ContactAction((0.,) * len(PHYSICAL_AXES))
    scores = []
    chosen, evaluated, rejected = choose_swap(
        motor, action, {"left_1": (0,)}, {0: .05, 1: .10},
        "left_1", candidate_scores=scores)
    assert motor.observe() == before
    assert evaluated + rejected > 0
    assert scores
    assert all("axis" in score and "first_error_ms" in score for score in scores)
    if chosen:
        selected = [s for s in scores if s["selected"]]
        assert len(selected) == 1
        assert selected[0]["axis"] == chosen.candidate_axis
        assert "second_error_ms" in selected[0]
