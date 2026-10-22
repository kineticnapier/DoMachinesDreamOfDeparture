from __future__ import annotations

"""Offline classification of predeclared counterfactual interventions."""

from dataclasses import dataclass
from enum import Enum


class InterventionClass(str, Enum):
    HUD_ONLY_SAFE = "hud_only_safe_intervention"
    ORACLE_ONLY = "oracle_only_intervention"
    INEFFECTIVE = "ineffective"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True, slots=True)
class InterventionEvidence:
    name: str
    oracle_event89_recovered: bool
    action_information_visible_by_deadline: bool | None
    robust_all_hypotheses_safe: bool | None
    motor_envelope_valid: bool | None
    baseline_hits_preserved: bool | None
    runtime_parity_confirmed: bool


@dataclass(frozen=True, slots=True)
class InterventionAssessment:
    classification: InterventionClass
    reasons: tuple[str, ...]


def classify_intervention(evidence: InterventionEvidence) -> InterventionAssessment:
    reasons: list[str] = []
    if not evidence.oracle_event89_recovered:
        return InterventionAssessment(InterventionClass.INEFFECTIVE, ("event89_not_recovered_in_reference",))
    if evidence.action_information_visible_by_deadline is not True:
        reasons.append("action_not_causally_selectable_by_deadline")
    if evidence.robust_all_hypotheses_safe is not True:
        reasons.append("not_safe_in_all_retained_hypotheses")
    if evidence.motor_envelope_valid is not True:
        reasons.append("motor_envelope_not_validated_for_actual_snapshot")
    if evidence.baseline_hits_preserved is not True:
        reasons.append("baseline_hit_preservation_unproven")
    if not evidence.runtime_parity_confirmed:
        reasons.append("live_stock_runtime_parity_unverified")
    if reasons:
        return InterventionAssessment(InterventionClass.ORACLE_ONLY, tuple(reasons))
    return InterventionAssessment(InterventionClass.HUD_ONLY_SAFE, ())
