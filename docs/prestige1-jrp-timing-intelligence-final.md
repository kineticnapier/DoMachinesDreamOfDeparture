# DMDOD Prestige I — JRP Timing Intelligence final report

## Outcome

**No measured improvement was demonstrated.**  The work is complete under the
explicit failure clause: the causal implementation, matched evaluations,
physical Motor loop, counterexamples and causes are recorded without promoting
a failed candidate.

Frozen Timing v1.4 remains the adopted system.  Stage A remains FAIL and
`may_run_stage_c=false`.  No old checkpoint, dataset role, scorer, Motor physics
or 263D Student input was overwritten or redefined.

## Implemented causal system

- Exact rounded visible JRP TBPM/CBPM/KPS strings, normal and optional Jongyeol
  pseudo lifecycle, Hit-triggered updates and display hold behavior.
- Conservative CBPM anchor bridge, hit-epoch association v2, frozen confidence
  gate v3, and isolated rejected phase-continuation v4.
- Frozen `prestige_v2.pt` adapter.  The neural model receives exactly its old
  263D HUD.  It emits 8-key actions plus causal relative timing, event
  confidence and chord size.  JRP remains a separate versioned Timing stream.
- Evaluator truth—target timestamps, floor IDs, score and renderer diagnostic
  metadata—is inaccessible to policy calls and used only after rollout.
- Same-checkpoint JRP-hidden/JRP-visible closed-loop evaluation with the real
  repository 8-Key Motor dynamics (10ms control, 1ms physics).

## Primary held-out Final10 result

Locked before opening Final: all ten Final charts, middle eight seconds, same
checkpoint/Motor/initial state, only JRP visibility changed.

| Metric | Timing v1.4 | JRP visible |
|---|---:|---:|
| targets | 862 | 862 |
| matched | 718 (83.2947%) | 719 (83.4107%) |
| false | 0 | 2 |
| JRP-only | — | 3: 1 correct, 2 false |
| matched-event MAE | 0.0000042ms | 0.0000042ms |
| Motor Hit / Miss | 147 / 715 | 147 / 715 |
| Too Early / Overload | 29 / 0 | 29 / 0 |
| physical keydowns | 176 | 176 |

The slight tracking increase is invalid as an improvement because false
detections increased.  The two false predictions were 15.625ms and 31.250ms
from the nearest target despite evidence confidence near 0.999.

## Checkpoint-unseen Development20

The checkpoint's saved Train20 hashes were excluded.  The remaining Train60
were SHA-sorted into Development20 / unopened Qualification20 / unopened
Reserve20.

| Candidate | Matched / 1340 | Combined false | JRP additions |
|---|---:|---:|---:|
| v1.4 baseline | 1310 | 3 | — |
| hit-epoch v2 | 1310 | 4 | 1: 0 correct |
| confidence v3 | 1310 | 3 | 0 |
| phase v4 | 1310 | 48 | 45: 0 correct |

All Development arms had identical physical results: 224 Hit, 1096 Miss,
62 Too Early, one Overload, 286 keydowns.  Qualification was intentionally not
opened after every Development candidate failed the improvement condition.

## Failure cases and causes

1. **One-frame association ambiguity.** At high event rates, a prior Hit's text
   update and the next HUD floor transition arrive in one 10ms observation.
   A Student cannot causally know which event generated the changed string.
2. **Confidence miscalibration.** Quantization uncertainty says nothing about
   event/text identity; the first system was confidently wrong.
3. **Display hold is not cadence.** CBPM describes the interval following the
   floor that last refreshed JRP.  Ordinary ADOFAI floor angles can change with
   no special marker, so held CBPM cannot be extrapolated periodically.
4. **Miss feedback sparsity.** A Miss generates no JRP Hit refresh.  Precisely
   when baseline recovery is most useful, the overlay may be stale.
5. **Information overlap.** Safe JRP one-step candidates were usually already
   present in v1.4.  Unsafe relaxations created false events rather than useful
   extra coverage.

## Edge-condition evidence

Tests cover Miss, no-Hit display hold, 0/10/30ms visibility delay, one-frame
early/late updates, pseudo KPS rejection, multiple events per frame, invalid
numbers/HUD, Pause, SetSpeed, Twirl, full-turn aliases, reset and physical Hit
hook behavior.  The final related suite is 68/68 PASS on CPU.  Native Unity/JRP
hook parity remains outside this simulator evidence and is not claimed.

## Artifacts

- `reports/prestige1-jrp-final-v1-983c396-r4.json` — original Final FAIL.
- `reports/prestige1-jrp-final-v1-314784d-r5.json` — evaluator-hardening rerun,
  identical FAIL.
- `reports/prestige1-jrp-unseen-v2-development-27fb0c0-r2.json`.
- `reports/prestige1-jrp-unseen-v3-development-4f9afdf.json`.
- `reports/prestige1-jrp-unseen-v4-development-73f17ae.json`.
- `docs/prestige1-stage-t-jrp-final-v1.md` and
  `docs/prestige1-stage-t-jrp-unseen-v2.md` — detailed analyses.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --basetemp=artifacts/pytest-jrp-phase-v4 `
  tests/test_jrp_cbpm_phase_v4.py `
  tests/test_jrp_cbpm_anchor_v3.py `
  tests/test_jrp_unseen_split_v2.py `
  tests/test_jrp_cbpm_anchor_v2.py `
  tests/test_jrp_closed_loop_v1.py `
  tests/test_jrp_cbpm_bridge_physical.py `
  tests/test_jrp_cbpm_anchor_v1.py `
  tests/test_jrp_overlay_diagnostic.py `
  tests/test_jrp_overlay_v2.py `
  tests/test_jrp_bpm_v1.py

.\.venv\Scripts\python.exe -u scripts/audit_prestige1_jrp_unseen_v2.py `
  --config configs/training/prestige1.toml `
  --checkpoint checkpoints/prestige1/prestige_v2.pt `
  --device cuda --partition development --bridge-version v3 `
  --output reports/NEW-jrp-v3-development.json `
  --status-file logs/NEW-jrp-v3-development/status.json
```

Audits refuse to overwrite outputs.  Do not run Qualification as a tuning set.
