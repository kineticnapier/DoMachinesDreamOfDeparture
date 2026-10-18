# HUD-only Timing v1.4 × Fingering: first Windows r1 triage

Observed on Windows 2026-10-18, as reported by the user. This report
summarizes the *existing saved r1 data* and does not claim new local
simulation or completed fixes. The script outputs remain local; the
GitHub repository has the integration and diagnostic sources.

## Initial integration

- Ordinary 120/240 BPM: Oracle, HUD Network and HUD CenterFirst all
  48/48 Hit, 0 Miss, 0 TooEarly, no overload.
- Ordinary 480 BPM: Oracle 48/48, HUD Network 47/48, HUD CenterFirst
  48/48, no TooEarly or overload in any arm.
- SetSpeed speed_double: Oracle 32/32, HUD Network 29/32,
  HUD CenterFirst 29/32. No TooEarly/overload.
- Twirl speed fixture: 32/32 Hit for all three arms.
- Pause fixture: Oracle Network 31/32, HUD Network 31/32,
  HUD CenterFirst 32/32. No TooEarly/overload.
- Train bool first 5 seconds: Oracle 9/9, HUD Network 7/9,
  HUD CenterFirst 7/9, no TooEarly/overload.

The declared integration gates correctly FAIL overall. They must not
be weakened to hide these results.

## Saved r1 forecast/decision-only analysis

The r1 JSON did not include actual missed target ordinals, so the
following local indices describe forecast observations, NOT proven
missed events.

- Ordinary 480 HUD Network: zero unmatched target forecasts, zero
  insufficient lead and 48 assigned decisions. CenterFirst all Hit.
  Most informative next check: selected physical axis, recent actuation
  history, KeyDown delay, cancellation vs reset and timing difference.
- SetSpeed HUD Network and CenterFirst: 2 targets (ordinals 5 and 6)
  appear in insufficient lead, with 30 assigned decisions and 3 misses.
  A third Miss is not assigned to a cause by the saved aggregate. Check
  whether the third is cascading physical reservations or a distinct
  Timing/physical failure.
- Pause: both Network arms have one Miss, while CenterFirst gets all
  hits. Oracle's Miss prevents explaining this as solely a HUD Timing
  estimator defect; compare actual selected physical axes and timing.
- Train bool 5s: HUD arms each have 2 Miss. Target ordinal 0 has no
  forecast within 20ms; ordinal 1 has insufficient lead. Actual
  missed ordinals were not stored in r1 and must be confirmed.

## Differential follow-up (predeclared, no hyperparameter search)

Run exactly four cases (ordinary 480, speed double, pause, Train bool 5s)
through three frozen comparison arms. Keep current Network checkpoint,
style alpha, timing horizon, physical motor constants, and scorer fixed.
Only add *posthoc* physical timing and target-outcome diagnostics.

Use:

    .venv\Scripts\python.exe -m pytest tests/test_diagnose_hud_v14_failed_cases_v02.py -q

    .venv\Scripts\python.exe -u scripts/diagnose_hud_v14_failed_cases_v02.py --output-dir artifacts/fingering_network_v02/hud-v14-failure-crosscheck-r1

The script creates separate create-once arm JSON files and case-level
comparisons; interrupted runs preserve completed arm files. Comparison
of nearby KeyDown times is temporal association, not proof of target/
physical-finger causality.

Do not change frozen Timing v1.4, retrain Fingering, or use Validation20
to improve scores before examining the physical timeline. If a condition
is genuinely unobservable or the available lead is shorter than the
Motor's physical response time, report abstention/miss instead of
filling in Oracle true time.
