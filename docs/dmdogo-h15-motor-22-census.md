# DMDOGO H1.5 — 22-candidate physical Motor census

## Objective

Two H1.4 pilots have already produced complete full-hit and exact-plan-adoption Motor results:

- ALL MY FELLAS, 1,540 events, alternative e142af91b178-1089-1090 (8 changed).
- Regular Dan 2nd Dan, 3,637 events, alternative 2ada1d0a06b5-1237-1238 (2,385 changed).

H1.5 **reuses** these two H1.4 files and executes only the remaining 20 independent H1.1 alternatives. Every candidate gets a separate physical episode. Never merge 22 alternative plans into one fictitious physical session.

## Evidence boundary

- Exactly three frozen Train charts (10/10/2 candidates), 10,231 distinct original source events, 22 independent alternative episodes.
- Recompute the full H1.3 report from H1.1 bytes; check source SHA256, Train manifest, original verified full-chart physical report and frozen v0.8 code digest.
- Run each new episode through the existing H1.4 worker. The original Planner, Motor, Timing, physics, selection and fallback logic are unchanged.
- Only previously selected Train cases. No Validation20/FINAL, training, new chart selection, human labels or HUD-only Student.

## Strict status and replay

The two earlier H1.4 outputs are protected by their previously observed SHA256s: 5cb3cb69b791d2da6fd630342575901a04cc37fa815f860f51da0542153e2e57 (1089) and 0bf2d8bd91c83a5d79cc1be245455db993953c5409a445870ec6166f41c0c244 (1237). The prior status file must independently report success and match the exact output bytes.

Every new result must pass the same full evidence checks: H1.4 executed format, unchanged frozen-source and input identities, event count, original baseline fields, selected event identities, every desired axis from H1.1, chosen axes, actual adoption, local replans, scorer Hit/Miss/TooEarly/Overload and actual KeyDown count. A forged, truncated, or partial output cannot count as success.

Physical completion and physical success are DISTINCT. The stricter all_hit_and_exact_plan requires chart_complete, all Hit, zero Miss/TooEarly/Overload, full event selection, all desired H1.1 axes actually adopted and zero local replans. Deferrals are allowed if later resolved.

All verified physical episodes remain part of the census even if they have misses, overload or local replans. They must not be selectively discarded. H1.5 reports recorded runtime fingerprint equality separately; it does not assume equality.

## Windows instructions

Stay on develop/dmdogo. Preserve local edits. Never reset, clean, stash, rebase, force push, or overwrite earlier evidence.

### Focused tests

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h15_motor_census.py
~~~

### Initial read-only audit (does not run Motor)

~~~powershell
uv run python scripts/run_dmdogo_h15_motor_census.py `
  --manifest reports/dmdogo-v08-train-selection-r2.json `
  --verified-report reports/dmdogo-v08-train-report-r4.json `
  --h11-report reports/dmdogo-h11-fingering-pairs-r1.json `
  --h13-report reports/dmdogo-h13-automated-quality-r1.json `
  --results-root artifacts/dmdogo-v08/train-generalization-r2
~~~

Expected initial count: 2 verified reused pilots, 20 pending, full_census_verified=false, provided both original file/status pairs still match their SHA256s.

### Launch remaining cases automatically

Append to the same command:

~~~powershell
  --execute --max-new 20
~~~

Or use --execute --max-new 3 for a staged run. Calls are sequential, foreground processes on the user's PC; the script checks the already completed candidates and does not rerun them. Each candidate has an independent 1,200-second H1.4 watchdog by default; 20 runs may take substantial time. The first failed worker stops the batch.

Each fresh candidate saves a create-once pair:

- artifacts/dmdogo-h15/motor-r1/<candidate-id>.json
- logs/dmdogo-h15/motor-r1/<candidate-id>.status.json

The original two H1.4 files stay where they are, untouched. Incomplete status or mismatched bytes stop safe resumption; never silently retry, retune, or replace the failure.

### Final report (only when all 22 episodes verified)

Append to the initial read-only command:

~~~powershell
  --complete-report reports/dmdogo-h15-motor-22-census-r1.json
~~~

That option exclusively creates a final JSON once the full 22-case evidence census is verified, including unsuccessful physical cases. With fewer than 22 verified episodes the report is NOT created. The normal read-only command prints progress without needing a report file.

## Limits

Twenty-two counterfactual episodes span only three source charts, not 22 independent chart samples. All use Oracle exact due times, not HUD-only Student perception. Motor feasibility does not establish human-preferred fingering, physical human comfort, stamina or anatomy. A full-hit episode with local fallback is not evidence that the specified alternate fingering itself was executed.

GitHub implementation alone is not a passed Windows test or an executed 22/22 Motor census. Do not claim these outcomes without the local output.
