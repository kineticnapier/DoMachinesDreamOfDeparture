# Stage C v0.6.2 — causal source of the 14ms mismatch

**Case:** 命日 (Nerfed), 58.400s, #614–615, left_5 -> left_6.
The frozen current-command forecast predicted a logical left_7/left_8 DOWN
at 58,455ms. In actual alternate-Teacher execution it occurred at 58,469ms.

The saved TieAware inversion diagnostic reports a **left_6 delay at 58,410ms**
for target #614. Replaying the actual 15 control commands into a cloned
12-axis/16-contact Motor reproduces 58,469ms exactly. Replaying the first
command unchanged predicts 58,455ms. Changing only the 58,410ms left_6
command to +1 within the recorded sequence produces 58,451ms.

More decisively, a snapshot of the **teacher and environment together**,
with the same initial selected reservation, re-runs teacher.action each
10ms and reproduces the actual full command sequence and all 1ms-timestamped
physical/logical events exactly. Thus the 14ms mismatch is explained by
subsequent control replanning, not inconsistent cloned Motor physics.

The one-tick intervention is a *fixed-follow-up-actions* counterfactual;
it is not a second closed-loop run with teacher replanning.

This is an isolated 15-tick case study and not proof that two-note scoring
can choose among multiple feasible reservations. No Student, Motor, scorer,
or frozen TieAware implementation is changed.

To reproduce with a locally available chart:
```powershell
uv run python -u scripts/trace_swapped_teacher_causality_v062.py "PATH_TO_NERFED.adofai" --output diagnostics/swapped-teacher-causal-v062.json
```
