# v0.6 lookahead diagnostic, ranking frozen

A separate copy of v0.6 records second-press candidate attempts under
`swaps[].candidate_scores[].rollout_attempts`, including whether the axis
was still pressed before the hypothetical second command and whether a
keydown happened inside the fixed forecast horizon. No selection parameters,
physical constants, scorer, or Student are changed.

**Important:** The v0.6 rollout advances in 10ms control chunks. Its
`second_error_ms=1000` is a sentinel, not a measured 1000ms error.
This diagnostic preserves that behavior for reproducibility. Failure labels
`not_reset` and `no_keydown_within_horizon` are preliminary and do not
prove the underlying cause: there may be onset scheduling or competing
reservations. Full correction requires a separately tested physical rollout
whose step origin and release timing match real teacher execution.

Run:
```powershell
git pull --ff-only
uv run pytest -q tests/test_fingering_v06_lookahead_debug.py
uv run python -u scripts/compare_nerfed_fingering_v06_lookahead_debug.py `
  "C:\\Users\\Kinetic Napier\\Downloads\\命日 (Nerfed) - LeaF (1)\\nerfed.adofai" `
  --after-s 58 --end-s 65 --max-swaps 12 `
  --output diagnostics/fingering-v06-lookahead-debug-58to65s.json
```
