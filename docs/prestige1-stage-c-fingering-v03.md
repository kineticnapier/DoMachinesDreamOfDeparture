# FingeringPlanner v0.3 — oracle-run reservation diagnostics

`scripts/diagnose_nerfed_fingering_reservations.py` is a standalone,
read-only *short-segment* diagnostic. It reuses the existing
`u1_keydown_attribution_audit.audit` scorer trace and the unchanged
`TieAwareJointTeacher` action. On the first few newly created
reservations, it clones the same 12-axis physical motor state and checks
physically compatible, currently unreserved alternative axes using the
v0.2 reservation-swap helper. The *real* run still uses the original
teacher action.

Actual baseline keydowns are joined to the sampled reserved ordinals
only **after** the run, so samples can be compared without affecting
control. Reports are write-once and include chart SHA256 and explicit
limits. The candidate projections are not whole-chart scored
counterfactuals and cannot establish alternative SAFE play.

Run on Windows (from repository root):

```powershell
uv run pytest -q tests/test_fingering_reservation_v03.py
uv run python -u scripts/diagnose_nerfed_fingering_reservations.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --end-s 15 --output diagnostics/fingering-v03-first15s.json
```

Frozen Stage A FAIL, Student, scorer, Teacher, checkpoints and
Validation/FINAL unchanged.
