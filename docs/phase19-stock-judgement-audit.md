# Phase 19 — Stock ADOFAI judgement audit

## Scope and evidence grading

This audit inspected the existing local reverse-compiled snapshot in
`F:\adofai-code`.  The prior IL audit identifies its source assembly as
`Assembly-CSharp.dll` SHA-256
`0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD`.
The source snapshot does not expose a trustworthy product version and no live
Stock run was performed in Phase 19.  Consequently, **VERIFIED** below means
verified for this audited DLL control flow, not every Stock release or the
currently installed executable.

Relevant reverse-compiled source fingerprints:

| File | SHA-256 |
|---|---|
| `scrMisc.cs` | `5DD7696235ED26476A01D9417ACEE4A7BE91D19567A9BD83FA510DC4225F62A5` |
| `scrPlayer.cs` | `21326189390DB431A4BBE2EA9E608D67048509AA2138526E5AD5179DA08EFC47` |
| `scrPlanet.cs` | `2250FEB39218F386233227E1BD2007BD1A9360C62C12749EBE0B8EFC07618C0E` |
| `scrMarginTracker.cs` | `A7E181E0F22CE32A99D21B1D0A0D4E0121ABB8B0066AC978BD4BDC4BCDB7A1BF` |
| `HitMargin.cs` | `650B1BE9D1C6020BECD767347DA59F9987EE2A247B2E18327EC4D845EC208998` |

Definitions used here:

- **VERIFIED**: direct control flow or value in the audited DLL snapshot.
- **INFERRED**: implication of several code paths or existing observations,
  without a matching live Stock capture.
- **UNKNOWN**: the available evidence cannot establish the behavior.

Public Mod cross-check: the maintained
[AdofaiTweaks repository](https://github.com/PizzaLovers007/AdofaiTweaks)
documents judgement-visual and restricted-judgement hooks, and the
[ADOFAIMacro repository](https://github.com/adofaiex/ADOFAIMacro) documents a
direct path that calls the game's `Hit()` separately from real key injection.
These are useful corroboration that the named judgement/input seams are used by
real Mods. Their public documentation does **not** specify exact window
boundaries, current-floor allocation, automatic-Miss thresholds, or render
latency, so it does not promote any UNKNOWN item below to VERIFIED.

## Findings

| Question | Classification | Evidence and conclusion |
|---|---|---|
| Judgement categories | VERIFIED | `HitMargin.cs:1-14` defines TooEarly, VeryEarly, EarlyPerfect, Perfect, LatePerfect, VeryLate, TooLate, Multipress, FailMiss, FailOverload, Auto, and OverPress. |
| Window construction | VERIFIED | `scrMisc.GetAdjustedAngleBoundaryInDeg`, `scrMisc.cs:178-212`, combines 30/45/60 degree bases with minimum-time radii. Desktop counted minima are Lenient 91 ms, Normal 65 ms, Strict 40 ms; EP is 30 ms and Pure is 20 ms, then each has an absolute 25 ms floor. Mobile uses 90/70/50 ms. Speed-trial, effective BPM/speed, pitch, and tile `marginScale` affect the final angular boundary. It is not universally a fixed 25/30/65 ms window. |
| Boundary conditions | VERIFIED | `scrMisc.GetHitMargin`, `scrMisc.cs:233-266`, starts at TooEarly and applies ordered strict `>` comparisons at `-Counted`, `-Perfect`, `-Pure`, `+Pure`, `+Perfect`, `+Counted`. Equality stays in the earlier comparison's category: exact `-Counted` is TooEarly; exact `-Perfect` is VeryEarly; exact `-Pure` is EarlyPerfect; exact `+Pure` is Perfect; exact `+Perfect` is LatePerfect; exact `+Counted` is VeryLate. Float/angle conversion still applies. |
| Which tile receives a KeyDown | VERIFIED | `scrPlayer.UpdateHoldKeys`, `scrPlayer.cs:693-739`, removes a queued key and calls `Hit`. `scrPlanet.SwitchChosen`, `scrPlanet.cs:630-654`, evaluates the current planet against its current `targetExitAngle` and assigns only `currfloor.nextfloor` when valid. It does not search an arbitrary set of unresolved tiles. This is a sequential current-floor state machine, not proof of repository cursor parity. |
| TooEarly/TooLate behavior | VERIFIED | When no valid next floor is selected, `scrPlanet.cs:744-816` records the computed invalid margin and returns the same planet. The ordinary invalid input does not scan forward to another target. Death/fail-bar and special Hold/checkpoint/no-fail branches can alter the visible outcome. |
| Automatic Miss transition | VERIFIED | `scrPlayer.CheckPostHoldFail`, `scrPlayer.cs:453-475`, calls `Die` only when the signed passed angle is strictly greater than `max(pi, 2 * minAngleMargin)` (or a smaller no-fail/safe threshold). `scrPlayer.Die`, `:324-371`, normally ends play; in no-fail/safe/cooldown it temporarily sets `noFailInfiniteMargin` and calls `Hit`. `scrPlanet.SwitchChosen`, `scrPlanet.cs:649-652`, turns that forced transition into FailMiss. This is materially different from repository `_expire_misses` at `due + pass_s`. |
| Multiple inputs in one frame | VERIFIED (mechanism), UNKNOWN (all outcomes) | `scrPlayer.HitAutoFloors`, `scrPlayer.cs:505-523`, counts valid presses and appends one `keyTimes` item per press. `UpdateHoldKeys` consumes a queue item and `AsyncInputUtils.WhileFloorNotChange`, `AsyncInputUtils.cs:30-38`, reruns an action when the floor changes. Multipress/OverPress branches exist in `scrPlanet.cs:819-850`. Exact outcomes depend on floor length, average frame time, limiter, Hold, co-op, and settings; no live parity matrix was run. |
| Internal judgement counters | VERIFIED | `scrMarginTracker.AddHit`, `scrMarginTracker.cs:102-108`, appends the enum and increments `hitMarginsCount[(int)hit]`. Valid hits call it at `scrPlanet.cs:873-885`; invalid inputs at `:764-790`. Reset clears every count (`scrMarginTracker.cs:85-100`), and checkpoint rollback decrements removed history (`:32-60`). |
| Visible judgement counter surface used in Phase 17 | UNKNOWN for Stock | The audited DLL has hit text (`scrHitTextManager.ShowHitText`, `scrHitTextManager.cs:77-110`) and internal margin counts, but Phase 19 did not prove that Stock exposes the exact cumulative counter schema, timing, reset behavior, or Miss visibility simulated by Phase 17. |
| Display latency/order | UNKNOWN | `ShowHitText` proves a display call, not the exact rendered-frame capture latency or ordering of two texts/counters. No live capture tied render frames to input ticks. |
| Runtime identity of the audited DLL | UNKNOWN | The snapshot is hash-anchored but has no reliable product version and was not matched to a running installation in this phase. |

## Repository parity decision

Repository `RealChartEnv` uses `_next_unresolved_cursor`, offers each KeyDown to
the first unresolved target, resolves TooLate immediately, and auto-expires at
`target time + pass_s` (`src/dmdod/envs/real_chart.py:188-197,220-253,272-293`).
The audited Stock code instead operates on current-floor/next-floor state and
uses a later angular post-floor failure transition.  Both are sequential, but
that shared shape is insufficient to claim identical FIFO semantics.

Therefore:

- Stock window formula and boundary comparisons: **VERIFIED for audited DLL**.
- Stock current/next-floor allocation: **VERIFIED for audited DLL**.
- Repository FIFO equivalence: **not verified; important auto-Miss mismatch is
  directly observed in code**.
- Exact Phase 17 counter parity, render latency, and current runtime build
  identity: **UNKNOWN**.

Phase 19's estimator results must consequently be described as
repository-model results.  The non-FIFO/alternative-window runs are independent
countermodels, not claims about Stock.
