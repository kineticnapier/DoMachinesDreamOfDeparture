# Stage C FingeringPlanner v0.1 — Physical counterfactual primitive

Added `dmdod.planning.fingering_counterfactual` as an independent, non-learning
experimental primitive. For each candidate physical axis, clone an identical
12-axis Motor state, apply the same background commands with that axis pressed,
and run the existing 1ms Motor for a fixed horizon. Report first physical
KeyDown, absolute difference from supplied event time, other physical KeyDowns,
and whether the axis was already pressed.

This is **not** a complete FingeringPlanner or full-chart counterfactual:
it does not alter reservations, forecast future decisions, score multiple logical
contacts, emulate true Multipress, evaluate hand motions anatomically, or
establish whole-chart success. The 16-contact thumb/edge mode does not change.
No attempt is made to treat all physical KeyDowns as credited ADOFAI notes.
Candidate ranking is a deterministic descriptive sorting only; ties and equal
outcomes are not a learned cost function.

The due time can be oracle-truth **only in this evaluator namespace**. It must
never enter HUD-only Student input. Existing Stage A FAIL, 263D HUD, MaleCNS4096,
8-key Student, original Teacher, scorer, Validation and FINAL are untouched.

Smoke tests:
```powershell
uv run pytest -q tests/test_fingering_counterfactual.py
```

Next: test alternative whole-note reservations with closed-loop scorer,
including 2-contact coupling and arrival-order ties; compare against the fixed
2794/2794 U2 oracle baseline. Human 12K demonstrations should only be used as
separate descriptive data, not privileged Student inputs.
