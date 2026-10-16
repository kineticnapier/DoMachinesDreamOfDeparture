# Prestige I — Stage C FingeringPlanner v0.2 reservation-swap forecast

The evaluator-only `forecast_reservation_swap` function connects a proposed
physical axis allocation to the live `Coupled16ContactMotor` 12-axis/16-contact
physical state and `ContactOracleTeacher.reservations` token structure.

It clones the identical motor state, releases one reserved source axis,
presses a currently unreserved compatible candidate axis, and runs the existing
1ms physical model for a fixed horizon. The outputs carry the real emitted
logical KeyDown order (including same-ms ties), reserved note ordinal,
missing proposed ordinals, unexpected KeyDowns, ordinal inversions and
timing error. Single-contact and coupled two-contact swaps are supported
only when both axes have equal contact multiplicity.

**Important limits:** This is a local open-loop forecast, not a scored ADOFAI
counterfactual or a complete planner; it does not simulate replanning, check
future reservations outside the horizon, or replace the authoritative evaluator.
A physically feasible swap is not automatically a valid ADOFAI Hit. Privileged
note times are allowed only in this offline oracle evaluator. No 263D HUD,
Student, frozen Stage A gate, MaleCNS4096, original teacher, scorer or
Validation/FINAL data is changed.

Run tests: `uv run pytest -q tests/test_fingering_reservation_counterfactual.py`.

Next step: invoke this helper at selected actual teacher reservation boundaries
and compare the local forecast to the teacher's actual KeyDown trace, before
attempting any full-chart alternative-fingering strategy.
