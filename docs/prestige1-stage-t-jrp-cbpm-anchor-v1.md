# Stage T: JRP CBPM anchored continuation v1 (read-only)

**Independent research only.** No Student weights, Motor, frozen Stage A v2
FAIL / `may_run_stage_c=false`, original 263D HUD, v1.4 Timing source,
oracles, or checkpoints are modified.

## Causal contract

Source JRP: `Jongye0l/JipperResourcePack` commit
`861a1ff6d12ef1c785ca16fb980a7e9292cfee4f`. Text is produced by
the already separately tested `JrpOverlayRenderer`, with `Bpm.OnHit` and
`Overlay.Show` update semantics and optional Jongyeol pseudo group handling.

The **input** to `JrpCbpmAnchoredTracker.observe` is:

1. the already visible **263D HUD** frame, sampled every 10ms;
2. the **three rounded display strings** TBPM/CBPM/KPS (possibly stale);
3. `CausalTimingTracker`'s already elapsed HUD-predicted event records.

Not allowed: chart floor ordinal, target timestamps, exact JRP renderer BPM,
`JrpDisplayedHud.source_floor`, pseudo group count, OnHit callback flags,
or a privileged chart clock. The separate renderer and test evaluator can
read those values, never the planner.

Because `CBPM = 60 / next_floor_gap` (for normal mode), **CBPM supplies
an interval, not an absolute next-note time**. The planner needs an
independent *anchor*. It accepts exactly one interval when:

- the current visible floor window demonstrably advanced by **one** floor;
- exactly one recent independently tracked v1.4 event is available;
- the visible JRP text changed in the same or immediately adjacent 10ms
  frame (allowing physical actuation jitter);
- the previous/current visible floor geometry/markers and one further
  neighboring tile agree across the shifted window, and do **not** also
  match a two-step shift;
- the rounded KPS agrees with rounded CBPM/60, rejecting grouped pseudo
  BPM. This is a conservative text-only classifier, NOT secret JRP mode data;
- quantized CBPM introduces at most 0.75ms computed time uncertainty.

A Miss, identical rendered strings, a multiple-floor jump, a late update
beyond one adjacent frame, an ambiguous pseudo BPM, or a non-finite or
unreliable number all cause **abstention**. Only the next **single** event
is predicted. No future floor timing is extrapolated beyond that interval.

`CausalJrpTimingBridge` merges this independent candidate with v1.4
**pending** event times and suppresses duplicates to within 0.75ms.
This does NOT modify v1.4's identity reconciliation or create a
Fingering/Motor command. Student training is not performed.

## CPU audit (two fixed known charts, synthetic hypothetical Hit hooks)

**Chart labels are accessed only by the renderer to generate hypothetical
on-Hit text and by the evaluator afterwards**. The proposed predictor uses
only the three causal streams above. This is NOT an autonomous Student
rollout, not real ADOFAI hook parity, and not a holdout qualification.

| Chart/window | v1.4 matching | additional CBPM targets | combined | emitted CBPM mismatches |
|---|---:|---:|---:|---:|
| 命日 (Nerfed), 65s, 890 targets | 887 | **+2** | **889/890** | **0** |
| level-recovered, 50s, 863 targets | 852 | **+2** | **854/863** | **0** |

Matching is strictly one-to-one within **50µs**, so two close notes do
not both count as successes from one prediction. A first, unsafe version
produced a 5.43ms false prediction in the second chart when **two floors
advanced in a single 10ms tick**. It was rejected, and the observable
one-floor shift guard removed the false event on re-audit. Only the
corrected results above are adopted.

**Jongyeol pseudo** mode, 命日 65s: combined **889/890**, 0 CBPM mismatches
on supported text, with grouped KPS rejected.

**Miss counterexample:** skip the two hypothetical source Hit hooks that
previously generated the extra forecasts (source target ordinals 395 and
800). The additional two CBPM predictions both disappear; combined
falls safely back to **887/890**, 0 CBPM mismatches. This is not a model
of actual missed physical input, but isolates stale overlay behavior.

**Actual virtual Motor**, tiny 120 BPM fixture: 5 physical scored hits.
One OnHit display update occurred a frame after the visible event and
recovered the next event correctly (2.99999998s). Intentionally
4-frames-late accepted update did not anchor the earlier event.
The virtual scoring/HUD was unchanged; native ADOFAI game-hook parity
remains unverified. One-on-one Motor tests are diagnostic, not trained
Student performance.

## Tests and commands

```
PYTHONPATH=src python -m pytest -q \
  tests/test_jrp_cbpm_anchor_v1.py \
  tests/test_jrp_cbpm_bridge_physical.py \
  tests/test_jrp_overlay_v2.py \
  tests/test_jrp_overlay_diagnostic.py \
  tests/test_jrp_bpm_v1.py \
  tests/test_hud_bpm_unclipped.py

PYTHONPATH=src python scripts/audit_jrp_cbpm_anchor_v1.py \
  PATH_TO_NERFED.adofai --end-s 65 \
  --output diagnostics/jrp-cbpm-anchor-nerfed.json

PYTHONPATH=src python scripts/audit_jrp_cbpm_anchor_v1.py \
  PATH_TO_NERFED.adofai --end-s 65 --skip 395 --skip 800 \
  --output diagnostics/jrp-cbpm-anchor-miss.json
```

All related CPU regressions: **68/68 PASS**. Scenarios passed on the
CPU source snapshot and verified their exact committed source hashes.

## Important unresolved constraints

- First countdown near chart start still has no confirmed anchor, so
  the last missing event of 命日 is not recovered.
- The pseudo detection heuristic is conservative and not a proof of native
  JRP mode identity, especially at extreme display rounding.
- Visually identical successive floors and many very fast intervals
  cause abstentions. This cannot simply be loosened without wrong
  event associations.
- `Overlay.Show` on resets, replay edits, variable framerate / game
  OnHit-before-scoring, and native `scrFloor` angle equivalence need parity
  tests against the installed JRP/ADOFAI build.
- Integration into **trained Student**, especially the Motor feedback
  loop after an actual Miss, is still future work. No Stage A gate lift.
