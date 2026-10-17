# Stage T — checkpoint-unseen JRP hit-epoch v2

## Split contract

The frozen `prestige_v2.pt` stores exactly 20 Train chart hashes.  Those hashes
are excluded from the dataset's Train80.  The remaining 60 are ordered by
content SHA256 and split into Development20, Qualification20, and Reserve20.
Only Development was opened for this result; Qualification and Reserve remain
unobserved.

The fixed window is the middle eight seconds.  Hidden and JRP-visible runs use
the same checkpoint, Motor, chart, initial recurrent state, 10ms control and
1ms physics.  JRP never enters the checkpoint's 263D tensor.

## Development result: FAIL

Run PID 51612, 2026-10-17 11:22:56–11:25:29 JST, exit code 1.
Full output: `reports/prestige1-jrp-unseen-v2-development-27fb0c0-r2.json`.

| Metric | Timing v1.4 hidden | Hit-epoch JRP v2 |
|---|---:|---:|
| targets | 1340 | 1340 |
| matched | 1310 (97.7612%) | 1310 (97.7612%) |
| false | 3 | 4 |
| JRP-only added | 0 | 1 (0 correct, 1 false) |
| Motor Hit / Miss | 224 / 1096 | 224 / 1096 |
| Too Early / Overload | 62 / 1 | 62 / 1 |

The sole added JRP forecast had evidence confidence 0.3214.  Correct JRP
candidates existed, but v1.4 already contained them, so they were suppressed as
duplicates and did not improve coverage.  The v2 hit-epoch ordering removed the
known high-confidence Final association errors but was not sufficient for a
positive Development result.

## One bounded Development choice

The evidence confidence is quantization confidence multiplied by
`1 - observed_lag/(2*control_dt)`.  A minimum of 0.5 has a direct causal
interpretation: accept only the first half of the two-frame association window.
It rejects the sole Development false addition (0.3214).  This is one bounded,
predeclared Development selection, not a sweep.  Qualification must be run only
after the 0.5 gate and its tests are committed and pushed.

Qualification and Reserve metrics must not be used to change this threshold.
