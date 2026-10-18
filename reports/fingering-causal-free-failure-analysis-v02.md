# Fingering Causal Lead v0.2 — Free failure analysis

## Verdict

The three reported Misses are reproducible and are not simple late presses by
the fingers named on the failed decision rows.  They are sequence-alignment
failures caused by reusing a physical axis whose earlier stroke is still in
flight after its reservation is removed.  The evaluator consumes any physical
KeyDown against the next unresolved target, while the planner removes a
reservation as soon as that target ordinal is consumed by *any* contact.  This
can make an unpressed but positively moving axis look available for a later
event.  The next physical stroke is then missing or arrives after the target's
pass window.

The fixed Early Decision control proves that earlier Network decisions alone
are sufficient to reproduce failures 507 and 559.  The adaptive early-action
extension is therefore not the sole cause.  Event 642 additionally depends on
the Causal Lead trajectory: Early Decision hits 642, while Causal Lead leaves a
thumb stroke in flight and later retriggers it.

This is a diagnosis, not a policy fix.  Free, Human-Proxy, the checkpoint,
Motor constants, and Validation20 parameters were not changed.

## Fixed provenance and reproduction

- Chart: `14483 - QuomodocunquizE - 없는닉이없네.adofai`
- Chart SHA256: `b640ed3a8cf098d748d9661292ae590fa9b6b445bad299b8aec78dd022850c69`
- Checkpoint SHA256: `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`
- Saved source: `artifacts/fingering_network_v02/validation20-80-fast-r3/arms/`
- Diagnostic code commit: `f0c0787`
- Diagnostic run: PID 10036, 2026-10-18T06:03:42Z to 06:04:53Z,
  exit code 0
- Raw diagnostic:
  `artifacts/fingering_network_v02/validation20-80-fast-r3/diagnostic-run-20261018-r2/diagnostic.json`
- Raw diagnostic SHA256:
  `7bcd7419f15d4e82dfe6a2409de2cdf70200cca4a6e9edef49809939f056a847`

The probe observes the existing execution loop after decisions/actions are
formed.  It does not alter policy inputs or commands.  Saved Baseline and
Causal Lead statistics were required to match the rerun exactly.

| Policy / arm | Hit / target | Miss | TooEarly | physical KeyDown | X accuracy | Timing MAE |
|---|---:|---:|---:|---:|---:|---:|
| Free / Baseline | 1717 / 1717 | 0 | 0 | 1717 | 99.4671% | 11.8461 ms |
| Free / Early Decision | 1712 / 1717 | 5 | 1 | 1713 | 98.7427% | 12.5319 ms |
| Free / Causal Lead | 1714 / 1717 | 3 | 1 | 1715 | 99.5634% | 6.6815 ms |
| Human-Proxy / Causal Lead | 1717 / 1717 | 0 | 0 | 1717 | 99.9563% | 3.2874 ms |

## Important attribution rule

`decisions[].outcome` is derived from the event's evaluator target ordinal. It
does not prove that the KeyDown produced by that decision hit the same event.
The scorer is an ordered stream: every KeyDown is offered to the next
unresolved target regardless of the planned contact.  Therefore the event 507
row can be `Miss` even though its selected `left_3` physically went down near
the burst; that KeyDown was consumed by target 504.

## Event 507

- Target: ordinal/event 507, due 65909.375 ms.
- Free/Baseline: decision 65870 ms (39.375 ms lead), `right_2`, no extra
  lead. Four burst KeyDowns were 65919 `left_3`, 65920 `right_1`, 65923
  `right_3`, 65924 `right_2`; target 507 received the last one and Hit.
- Free/Causal Lead: decision 65850 ms (59.375 ms lead), `left_3`, 5.963 ms
  extra lead. KeyDown 65909 `left_3` was scored against target 504, followed
  by 65913 `right_3` -> 505 and 65916 `right_1` -> 506. There was no fourth
  KeyDown. Target 507 became Miss in the 65970--65980 ms control step.

The initiating conflict is earlier than the row marked Miss.  Free/Causal
Lead selected `left_5/tip` for event 502 and again for event 505.  At 65840 ms,
when the second selection was made, the thumb was not yet pressed but was
already moving toward actuation:

| state at 65840 ms | position | velocity | activation | fatigue | pressed |
|---|---:|---:|---:|---:|---|
| Free/Causal `left_5` | 0.001528450 m | +0.118970028 m/s | +0.132447762 | 0.060611765 | false |
| Human/Causal `left_5` | -0.000315939 m | +0.003685581 m/s | -0.015164787 | 0.000983918 | false |

The old Free thumb stroke went down at 65844 ms and did not go up until
65908 ms, 0.708 ms after event 505's due time.  It could not generate another
KeyDown for event 505.  Its prior target reservation had already disappeared
because target 502 was consumed by a different contact at 65838 ms.

Human-Proxy avoids this exact state history: events 500--503 use ordinary
axes (`left_1`, `left_2`, `right_2`, `left_4`), and events 504--507 use
`right_1`, `left_3`, `right_3`, then the rested `left_5`.  Its four KeyDowns
at 65907/65909/65910/65922 ms hit targets 504--507.

## Event 559

- Target: ordinal/event 559, due 67409.376 ms.
- Free/Baseline: decision 67370 ms (39.376 ms lead), `left_2`, no extra
  lead; target 559 Hit from `left_2` at 67432 ms.
- Free/Causal Lead: decision 67350 ms (59.376 ms lead), `left_3`, 6.000 ms
  extra lead. KeyDowns 67410 `left_3`, 67413 `right_3`, 67414 `right_1`
  were consumed by targets 556--558. Target 559 became Miss in the
  67470--67480 ms control step.

This is the same repeated chart pattern and the same mechanism.  The earlier
`left_5` stroke went down at 67343 ms and up at 67409 ms.  At 67340 ms the
axis was still nominally unpressed but in flight: position 0.001684497 m,
velocity +0.117382256 m/s, activation +0.139446933, fatigue 0.071817051.
Free selected it again for event 557, so no second thumb KeyDown was available
inside that burst.

Human-Proxy again keeps the thumb rested before assigning it only to the last
event.  KeyDowns 67409 `right_1`, 67409 `right_3`, 67410 `left_3`, and 67424
`left_5` hit targets 556--559.

## Event 642 and the TooEarly order

- Target: ordinal/event 642, due 70314.584 ms.
- Free/Baseline: decision 70280 ms (34.584 ms lead), `left_3`; target 642 Hit
  from `left_3` at 70331 ms.
- Free/Causal Lead: decision 70250 ms (64.584 ms lead), `right_5/tip`, zero
  extra lead because thumb axes are excluded from the adaptive lead rule.
- Human-Proxy/Causal Lead: decision 70250 ms, `right_4`, 22.594 ms extra
  lead; target 642 Hit from `left_1` at 70318 ms in the evaluator's ordered
  stream.

Free had already selected `right_5/tip` for event 637.  At the event 642
decision, the axis was not yet pressed but was visibly in flight: position
0.001887050 m, velocity +0.088443199 m/s, activation +0.086272149, fatigue
0.073093699.  It went down at 70252 ms and was consumed by target 639.  It
released at 70303 ms, then went down again at 70390 ms.

The recorded order is unambiguous:

1. Target 642 became Miss in control step 70370--70380 ms.
2. At 70390 ms `right_5` went down.
3. The evaluator was already waiting for target 643 (due 70500 ms), so the
   70390 ms KeyDown was classified TooEarly for 643.

Thus one delayed physical retrigger explains both the third Miss and the sole
TooEarly, in that order.  Human-Proxy does not select the moving thumb: at
70250 ms its `right_5` was essentially at rest (position -0.000000348 m,
velocity +0.000002696 m/s, activation -0.000009855) and event 642 used an
ordinary axis instead.

## Early Decision control and causal conclusion

Early Decision moves decisions to the same 69 ms horizon but retains the
original 44 ms action window.  It produced five Misses and one TooEarly
overall.  It reproduced Misses 507 and 559, with `left_2` as their final
selected axes, because the earlier thumb-reuse/reservation sequence still
removed one KeyDown from each four-note burst.  It did not miss 642; its
different earlier choices avoided that particular moving-thumb conflict.

The supported causal claims are therefore:

1. Earlier decision timing changes the Network choices and is sufficient for
   the first two failures.
2. Removing a reservation when its target is consumed, rather than when its
   physical stroke is finished or canceled, permits in-flight axis reuse.
3. Causal Lead's earlier actuation changes the state trajectory and is needed
   to reproduce the event-642 counterexample, but event 642 itself received no
   thumb extra-lead command.
4. Human-Proxy succeeds here because its frozen style choices avoid these
   specific in-flight thumb reuses.  This does not establish that
   Human-Proxy is universally safer or anatomically human.

## Minimal reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_diagnose_fingering_causal_failure_v02.py `
  tests/test_fingering_causal_early_v02.py -q

.\.venv\Scripts\python.exe scripts/diagnose_fingering_causal_failure_v02.py `
  --output artifacts/fingering_network_v02/validation20-80-fast-r3/diagnostic-run-REPRO/diagnostic.json `
  --wall-timeout-s 120
```

The output path is write-once.  The second command reruns only this chart and
four arms; it does not rerun Validation20 × 80.

## Scope and protection

- No checkpoint, policy weight, Human-Proxy alpha, Causal Lead parameter, or
  Motor constant was changed.
- No Qualification, Reserve, or Final chart was read.
- Stage A remains FAIL.
- Protected SHA256 values after the experiment:
  - Stage A: `c4f71aadc5dc6cca49679a9902bcb7583a0f89b40f92b7fb283a50c3faad0558`
  - MaleCNS4096: `a9d00a26c24ad19e815a0d768851f8040e00748e3a6474a31ce786576f53c710`
  - Fingering checkpoint: `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`
  - Dirty `prestige1.toml`: `869f114044d9af2185fa26a9525f825a82e7db6d30b4ff94d7fea5ab620bdb7f`

