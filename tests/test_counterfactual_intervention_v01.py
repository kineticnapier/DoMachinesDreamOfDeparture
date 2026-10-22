from dmdod.research.counterfactual_intervention_v01 import (
    InterventionClass,
    InterventionEvidence,
    classify_intervention,
)


def evidence(**changes):
    values = dict(
        name="candidate", oracle_event89_recovered=True,
        action_information_visible_by_deadline=True,
        robust_all_hypotheses_safe=True, motor_envelope_valid=True,
        baseline_hits_preserved=True, runtime_parity_confirmed=True,
    )
    values.update(changes)
    return InterventionEvidence(**values)


def test_only_complete_evidence_can_be_hud_safe() -> None:
    assert classify_intervention(evidence()).classification is InterventionClass.HUD_ONLY_SAFE


def test_oracle_success_without_causal_or_robust_evidence_stays_oracle_only() -> None:
    result = classify_intervention(evidence(
        action_information_visible_by_deadline=False,
        robust_all_hypotheses_safe=None,
        motor_envelope_valid=None,
        baseline_hits_preserved=None,
        runtime_parity_confirmed=False,
    ))
    assert result.classification is InterventionClass.ORACLE_ONLY
    assert len(result.reasons) == 5


def test_reference_failure_is_ineffective_not_safe() -> None:
    assert classify_intervention(evidence(oracle_event89_recovered=False)).classification is InterventionClass.INEFFECTIVE
