# Phase 20A — Stock Judgement Reference Model

## Scope and evidence

`dmdod.adofai.stock_judgement_reference` is a stateless research reference for
the portion of the audited Stock DLL that Phase 19 could verify.  Its evidence
base is the hash-anchored DLL snapshot and source positions recorded in
`docs/phase19-stock-judgement-audit.md`:

- `scrMisc.GetAdjustedAngleBoundaryInDeg` (`scrMisc.cs:178-212`) for boundary
  construction;
- `scrMisc.GetHitMargin` (`scrMisc.cs:233-266`) for the ordered category
  comparisons;
- `HitMargin.cs:1-14` for category names.

The audited DLL SHA-256 is
`0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD`.
Phase 19 did not establish the product version or match this snapshot to a live
installation, so this module is not a claim of universal Stock parity.

## Reproduced contract

`stock_angle_boundaries` accepts BPM and dimensionless speed, pitch,
difficulty, speed-trial, and normalized `margin_scale` inputs.  It returns
positive angle radii in degrees.  Angular speed is
`3 * bpm * speed * pitch` degrees per second.  The 30/45/60 degree bases are
scaled by `margin_scale` and widened by the audited minimum-time radii.

Desktop counted minima are 91/65/40 ms for Lenient/Normal/Strict; the
Early/Late Perfect and Pure minima are 30/20 ms.  Desktop speed-trial divides
these values before the 25 ms absolute floor.  The audited mobile branch uses
90/70/50 ms directly and does not apply speed-trial division.

`classify_stock_angle` separately accepts a signed angle error (negative is
early, positive is late) and the three boundaries.  It reproduces the DLL's
ordered strict `>` comparisons.  Consequently exact negative boundaries stay
in the earlier category, while exact positive boundaries stay in the inner
category.

## Repository comparison

The reference does not call or alter the production/simulator Scorer.

| Case | Audited Stock reference | Repository `rules.classify_timing` |
|---|---|---|
| Center | Perfect | Perfect |
| Exact `-Pure` | EarlyPerfect | Perfect |
| Exact `-Perfect` | VeryEarly | EarlyPerfect |
| Exact `-Counted` | TooEarly | VeryEarly |
| Exact positive boundaries | Inner category | Same result |
| Mobile with speed-trial | Fixed 90/70/50 ms minima | Divides mobile minima by speed-trial, then applies 25 ms floor |

The fixed 65 ms Normal counted minimum in the repository is only one input to
its boundary calculation; it is not treated here as a complete Stock-compatible
judgement rule.

Automatic Miss and target/floor assignment are `NOT_COMPARABLE` in this
stateless reference.  Phase 19 found that Stock uses a sequential current-floor
transition and angle-based post-hold failure path, whereas `real_chart.py` uses
an unresolved FIFO cursor and expires at `due + pass_s`.  Those state machines,
multiple-input behavior, fail branches, counter/display timing, and runtime DLL
identity are deliberately not reproduced.

## Verification

`tests/test_stock_judgement_reference.py` covers the Perfect center, both sides
of every category boundary, exact equality, parameter effects, the mobile
speed-trial difference, and direct comparisons with the repository classifier.
No production integration, training, Validation20, FINAL, or Train16 run is
part of Phase 20A.
