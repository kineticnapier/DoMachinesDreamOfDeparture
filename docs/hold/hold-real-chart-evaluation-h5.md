# Hold real-chart evaluation and regression audit (H5)

## Fixed Train-only corpus

The audit pins the three Train80 charts and SHA256 values first recorded in H0.
No Validation20, Reserve20, or FINAL data is opened.

| Chart | SHA256 | Holds |
| --- | --- | ---: |
| `04349 - ヒアソビ (feat. 初音ミク) - Zedwards.adofai` | `baaf6bff3cd012d8b0e4f123f58868c1cd4f262a35394e32eb2ebcba898caab9` | 2 |
| `10907 - Remote Access Trojan - 0mni & Trippy.adofai` | `d7e5a1cb5abc8ea1104116ac154f5f1e0b9fd889fdf2650ed1fe1d27510f4b82` | 22 |
| `13131 - 『계획 이상 有!』 (feat. 重音テト) - Team Name.adofai` | `167e781dca406fcec3a0ee754a173d40a50b23f1714ee93758ca3eaf53564d64` | 1 |

## Timing parity method

For each of 25 active non-final Holds, the audit compiles the original chart
and a counterfactual copy with only that Hold action removed.  It checks:

- Hold start time does not move;
- compiled additional time equals `duration * 2 * 60 / BPM_at_hold`;
- the Hold floor exit shift equals the additional time;
- the following floor target shift equals the additional time;
- episode duration is chart duration divided by pitch ratio.

Tolerance is 1 microsecond, chosen to be much larger than the audited stock
single-precision pi constant's nanosecond-scale numerical residue while far
below the 1ms physics step.  This is source-code/formula parity, not a claim of
live-game capture parity.  The JSON result explicitly records live comparison
as `UNVERIFIED`.

All 25 Holds pass code parity; maximum error is
`1.887379141862766e-14 s`.  The duration distribution is 0: 8, 1: 8, 2: 6,
3: 3.  One same-floor Twirl is present.  No same-floor SetSpeed, Pause,
MultiPlanet, or Midspin occurs in these 25 actions, so the synthetic H1 tests,
not this real-chart audit, remain the only evidence for those combinations.

## Input and judgement evidence

The deterministic state-machine audit covers:

- normal start, unrelated extra key, and valid release;
- early release in Normal mode;
- `CanHitEnd` continuous hold then end-key completion;
- `NoHoldNeeded` release then end-key completion;
- missed start and missed release;
- separate ordinary-Hit and Hold-completion credits.

These pass the audited H0 control-flow contract.  They remain code-parity
tests: no current live-game capture establishes exact margins or resolves the
decompiled continuous-Hold branch ambiguity.  Consecutive-Hold boundary
semantics remain fail-closed.

## Frozen Timing regression

The unchanged Phase 14 full audit is rerun after Hold changes.  The acceptance
criteria are:

- Train16 candidate clean: 1067/1068;
- reference5 candidate clean: 200/200;
- false-positive IDs: 0;
- existing-success regressions: 0;
- Midspin/Twirl opt-in checks unchanged;
- future SetSpeed gate remains closed.

Hold metrics are reported separately and are not added to Phase 14 coverage.

The rerun result is `TWIRL_CAUSAL_RECOVERY_PASS`: Train16 remains 1067/1068,
reference5 remains 200/200, false-positive IDs remain 0, and existing-success
regressions remain 0.  The Phase 12 fixed 13 remain 13/13 clean and at least
9/13 Motor-ready.  No Hold code is enabled in this frozen regression path.
