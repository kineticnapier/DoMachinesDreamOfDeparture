# Phase 16 — Event 89 causal diagnosis

## Result

`CAUSE_CONFIRMED_SAFE_FIX_BLOCKED`

Event 89 is not an isolated Timing-ID or Motor failure.  The candidate enters
a one-event head-of-line backlog much earlier, at 3.261 s:

- Event 37 is due at 3.220588272 s.
- Timing ID 37 first becomes actionable at 3.190 s, leaving 30.588 ms.  The
  frozen 44 ms Motor lead gate correctly abstains (`insufficient_observable_lead`).
- Event 36 has expired by 3.261 s, but Event 37's repository-scorer late window
  remains open until 3.285588272 s.
- Timing ID 38 is correctly forecast for 3.264705920 s, assigned to `right_2`,
  and physically/logically presses at 3.261 s.  The FIFO scorer therefore
  consumes it as Event 37.

The one-event offset continues for every emitted KeyDown through Timing ID 89.
ID 89 is correctly forecast near 5.250000044 s, assigned to `right_3`, and
physically/logically presses at 5.253 s; Event 88 remains unresolved until
5.270882397 s, so that press is credited to Event 88.  Event 89's own late
window closes at 5.315000044 s.  The next press at 5.319 s therefore expires
Event 89 first and Timing ID 90 is again credited to Event 90.  This is the
exact resynchronization point.

The expanded 0–6 s replay is evaluator-only evidence.  Target ordinals,
truth times, judgement traces and late windows are joined only after each
complete policy/physics replay; they never enter Timing, Fingering or Motor.

## Responsibility

| Component | Finding |
| --- | --- |
| Timing | Correct due and stable ID for the examined events. Confirmed causal limitation: ID 37 arrives too late for the frozen 44 ms guarantee. |
| Fingering | Not the direct cause. Every assigned reservation has one stable choice; ID 37 is not presented as assignable. |
| Motor | Not the direct cause. Reserved axes generate matching physical/logical edges; lifecycle and reservation violations are zero. |
| Scorer | Confirmed attribution mechanism in this repository: expire overdue targets, then score against the oldest unresolved target. Whether its exact late-window behavior matches the installed Stock game remains unverified. |
| Combined | Confirmed: late-observed abstention plus FIFO late-window consumption creates the backlog that reaches Event 89. |

The scorer behavior is not labeled a bug.  It is an established repository
contract and may represent sequential Stock play, but no independent live-game
evidence in this task proves exact Stock parity.

## Why no production fix was made

Three apparent fixes are unsafe with current evidence:

1. issuing a best-effort late press for ID 37 relaxes the frozen 44 ms Motor
   safety contract and changes the scheduler policy;
2. delaying ID 38 until Event 37 expires requires scorer-window knowledge and
   verified Stock parity, and can make ID 38 late;
3. assigning the 5.253 s press directly to Event 89 changes the evaluator to
   fit the desired result.

None follows solely from HUD observations.  The fixed checkpoint, Fingering,
Motor constants, scorer, and acceptance gate therefore remain unchanged.
The measured A/B remains 207/220 versus 162/220 with the single lost Hit at
Event 89, so the current candidate remains
`MOTOR_AB_REGRESSION_OR_INCOMPLETE`.

## Next bounded evidence

Before implementing recovery, obtain one of:

- live-game evidence for how a late input is assigned when the previous tile
  is unresolved and adjacent late windows overlap; or
- a separately approved, opt-in best-effort scheduling experiment that keeps
  the 44 ms guaranteed metric distinct and evaluates whether an extra physical
  input causes TooEarly/Overload or other regressions.

No Validation20, FINAL, Train16-wide Motor A/B, training, Hold vision, or
checkpoint update was performed.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -u scripts/run_hud_timing_phase16_motor_ab_v01.py `
  --chart-name "01645 - First town of this Journey remix - Irin.adofai" `
  --diagnostic-center-s 3.0 --diagnostic-radius-s 3.0 `
  --output reports/phase16-firsttown12-event89-causal-20261021-r2.json

.\.venv\Scripts\python.exe scripts/audit_phase16_event89_causal_v01.py `
  --source reports/phase16-firsttown12-event89-causal-20261021-r2.json `
  --output reports/phase16-event89-causal-diagnosis-20261021-r1.json
```

The raw expanded replay is 4.5 MB and remains a local, SHA-pinned generated
artifact; the compact causal audit JSON is the committed result.
