# FingeringPlanner v0.6: two-event physical lookahead (experimental)

Standalone fork of v0.5. Candidate first-stroke reservations are forecast
using the unchanged 12-axis motor. For each locally feasible candidate,
the routine clones the physical model again, runs the first press, releases,
and tries the next unreserved note using either that same axis or another
currently idle single-contact axis. It records best second-KeyDown error
and ranks candidates by first-event plus second-event timing errors.

Important limits: this is a **two-event surrogate**, not a committed two-note
assignment. It assumes a fixed 44ms lead for the next press, does not
score the hypothetical second note, does not adapt other active motors,
and does not model 2-contact allocation for that next note. It still
makes actual changes only to current teacher reservations and evaluates
the ensuing closed-loop real-chart score using the unchanged scorer.

Windows command:
```powershell
git pull --ff-only
uv run pytest -q tests/test_fingering_reservation_v06.py
uv run python -u scripts/compare_nerfed_fingering_v06.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --after-s 58 --end-s 65 --max-swaps 12 `
  --output diagnostics/fingering-v06-58to65s.json
```
