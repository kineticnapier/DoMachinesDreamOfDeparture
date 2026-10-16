# Stage C FingeringPlanner v0.4 — Scored closed-loop reservation swaps

The independent diagnostic script `scripts/compare_nerfed_fingering_v04.py`
runs two simulations over the same playable chart segment. First, an unchanged
TieAwareJointTeacher establishes the reference scorer result. Second, a
subclass may replace at most N *newly created, active* physical-axis reservations
with a compatible currently unreserved physical axis. It ranks all locally
eligible 12-axis candidates by short-horizon physical forecast timing error,
rather than selecting the first two in key-name order. Each changed reservation
is carried forward into the subsequent teacher control ticks. The actual
motor/scorer then runs normally, and the script reports scored Hit/Miss,
TooEarly, overload, reservation matching and the resulting attributed KeyDown.

For a first smoke test, change **one reservation**, not all of them:

```powershell
git pull --ff-only
uv run pytest -q tests/test_fingering_reservation_v04.py
uv run python -u scripts/compare_nerfed_fingering_v04.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --end-s 15 --max-swaps 1 `
  --output diagnostics/fingering-v04-first15s-one-swap.json
```

`--after-s 45 --end-s 65` targets the denser passage while still running
from time zero: it changes reservations only after 45 seconds, and should
therefore not be described as a 20-second isolated simulation.

The experiment uses a privileged Oracle with chart-truth times and the existing
simplified scorer. It **does not** train FingeringPlanner, calibrate anatomy or
prove real ADOFAI Multipress correctness. Baseline teacher, frozen Student,
Stage A gate, checkpoints, and Validation/FINAL are untouched.
