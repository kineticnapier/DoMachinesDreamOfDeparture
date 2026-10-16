# Stage C v0.6 Candidate Ranking Diagnostics

Adds candidate-by-candidate diagnostic rows to the *existing* v0.6 experiment
without changing its ranking or physical scoring behavior. Every accepted
local forecast now records first error, second-event error, next ordinal,
chosen hypothetical second axis, combined error and selection reason.
Locally infeasible candidates are also recorded. The JSON includes this
information under each `swaps[].candidate_scores`.

This is **not evidence** that two-note lookahead improves selection;
it is meant to diagnose why v0.6 and v0.5 picked identical axes.

Run after updating:
```powershell
uv run pytest -q tests/test_fingering_candidate_scores.py
uv run python -u scripts/compare_nerfed_fingering_v06.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --after-s 58 --end-s 65 --max-swaps 12 `
  --output diagnostics/fingering-v06-scored-candidates-58to65s.json
```

The old v0.6 report is retained. Frozen Oracle, Motor, scorer and Student
are not changed.
