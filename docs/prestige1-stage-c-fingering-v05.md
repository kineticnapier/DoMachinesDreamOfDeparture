# Stage C v0.5 — axis-specific recovery surrogate

This experimental script is a fork of v0.4 and leaves the v0.4 comparison
unchanged. For each locally feasible replacement physical axis, the planner
combines 7-frame projected timing error with a candidate-specific physical
reset penalty if the next unreserved target arrives within 110ms.

For each candidate, a cloned 12-axis Motor receives the candidate press
and current concurrent actions, then a release. The penalty is applied only
when the candidate's physical keyboard has not reset before that next target.
This can change ranking across axes; it is not an anatomical recovery model,
a full future reallocation, an assured feasible second stroke, or learned
fingering. The physical simulation and scorer still decide actual Hits.

Command from repository root:

```powershell
git pull --ff-only
uv run pytest -q tests/test_fingering_reservation_v05.py
uv run python -u scripts/compare_nerfed_fingering_v05.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --after-s 58 --end-s 65 --max-swaps 12 `
  --output diagnostics/fingering-v05-58to65s.json
```

The frozen Stage A gate, teacher, Student, scorer, checkpoints,
Validation/FINAL remain unchanged. Compare against the saved v0.4 run
rather than asserting v0.5 performance before Windows execution.
