# Hold integration H6 quality report

## Final decision

`HOLD_PARTIAL_INTEGRATION`

The implementation safely covers code-audited timing intervals, a deterministic
judgement state machine, a causal Short/Long icon observation, and a physical
Motor hold overlay.  It does not meet `HOLD_INTEGRATION_PASS`: positive Hold
durations remain indistinguishable in the verified icon-only HUD, and neither
live-game judgement margins nor an end-to-end HUD-only full-chart Hold run have
been demonstrated.

## Component decisions

| Component | Result | Evidence boundary |
| --- | --- | --- |
| Stock Timing | Code parity PASS; live game UNVERIFIED | 25 pinned Train Holds; per-action removal counterfactual; max error `1.887379141862766e-14 s` |
| Hold judgement | Deterministic contract PASS; live margins UNVERIFIED | Mode/state/input unit tests; ambiguous decompiled continuous-Hold branch excluded |
| HUD recognition | Partial; exact duration BLOCKED | Four audited stock icon classes; duration 1 and 3 yield identical observations but different releases |
| Motor operation | Deterministic overlay PASS; full-chart HUD-only BLOCKED | Real 12-axis/16-contact simulation with concurrent ordinary tap; unverified timing rejected |
| Existing Timing | Regression PASS | Phase 14 Train16 1067/1068, reference5 200/200, FP 0, regression 0 |

These results are separate.  In particular, a correct privileged
`PlayableHoldInterval` does not prove that the Student can observe its release,
and a deterministic Motor overlay does not prove a stock Hold judgement.

## API and quality checks

H6 checks:

- stable H1 `PlayableHoldInterval` API separate from ordinary targets;
- Student Hold fields contain no duration, target time, chart time, or absolute
  floor index;
- deterministic double compilation of all three pinned Hold charts;
- reset, missing/stale frames, ID reuse, axis collision, and invalid input
  tests;
- Hold-free exact fixture tests and full frozen Phase 14 regression;
- fixed-three double-compile bounded-time and peak-memory smoke.

The performance check is an absolute safety bound, not a historical speedup or
formal performance-regression claim because no pre-Hold memory/time baseline
was recorded.

The focused Hold/HUD/Timing/Composite/Motor contract suite passes 245/245.
The repository-wide suite, rerun with a writable `--basetemp`, reports
1878 passed and 9 failed.  All nine failures are outside the files changed by
this program: seven are a test-side missing `cos` import in the old half-turn
audit, one is an older diagnostic mock that does not accept the existing
`legacy_bpm_encoding` keyword, and one is an older overwrite-guard test whose
empty `SimpleNamespace` reaches `config.output_dir` first.  They are reported,
not modified or hidden.  The first full-suite attempt without `--basetemp`
also produced 183 sandbox Temp setup errors; those disappear with the writable
workspace basetemp and are not code failures.

## Unsupported or unverified cases

- live-game parity against the exact installed ADOFAI build;
- exact stock start/release judgement margins;
- the contradictory-looking `CheckPreHoldFail` branch in the decompiled source;
- consecutive-Hold boundary hand-off;
- simultaneous key callback ordering in the live game;
- robust Hold mesh loop-count recognition under camera/skin/occlusion changes;
- HUD-only release prediction for positive duration;
- full-chart Hold scoring and Motor control without privileged timing;
- real-chart Hold combined with same-floor SetSpeed, Pause, MultiPlanet, or
  Midspin (not present in the pinned 25; synthetic timing tests exist);
- Hold mode observability from pixels.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_adofai_hold_timing.py tests/test_hold_judgement_v01.py tests/test_hold_hud_v4.py tests/test_hold_motor_overlay_v01.py tests/test_audit_hold_h5_v01.py tests/test_audit_hold_h6_v01.py -q

.\.venv\Scripts\python.exe scripts/audit_hold_h5_v01.py --output reports/<new-h5-name>.json

.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase14_twirl_v01.py --manifest reports/hud-timing-phase11-selection-20261019-r1.json --output reports/<new-phase14-name>.json --trace-output reports/<new-phase14-trace-name>.json

.\.venv\Scripts\python.exe scripts/audit_hold_h6_v01.py --h5 reports/hold-h5-train-audit-20261020-r1.json --phase14 reports/hold-h5-phase14-regression-20261020-r1.json --output reports/<new-h6-name>.json
```

Writers use exclusive creation; existing evidence is never overwritten.
