# Stage C v0.6.1 — second-stroke timing diagnostic

Isolated diagnostic fork of v0.6, preserving the scorer, frozen motor, teacher
and Student. First stroke replay now releases the original source axis,
records the true physical DOWN event and the containing 10ms controller tick,
then tests release and each eligible next-stroke axis. Every trial logs
actual control command start, end, key state, and timestamped physical events.

This experiment deliberately uses a 44ms lead assumption, rather than
claiming an optimal scheduled second stroke. Failed trials retain a distinct
null second error; for ranking compatibility only they receive a 1000ms
surrogate. The candidate-selection result MAY change. Test in a standalone
output; never overwrite previous JSON.

```powershell
git pull --ff-only
uv run pytest -q tests/test_fingering_v061.py
uv run python -u scripts/compare_nerfed_fingering_v061.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --after-s 58 --end-s 65 --max-swaps 12 `
  --output diagnostics/fingering-v061-58to65s.json
```
