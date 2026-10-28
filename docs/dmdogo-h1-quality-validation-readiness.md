# DMDOGO H1.0 — Fingering quality evidence register (readiness gate)

## Purpose and research stopping decision

After H0.15–H0.17 did not obtain an early certified release in the
same 34 synthetic targets, stop refining that lower-bound sequence.
H1 returns to the *original question*: does an alternate low-cost
fingering correspond to a playable and human-preferred fingering?

H1.0 is NOT a validation result. It is a conservative, source-pinned
**inventory of what is already measured versus what remains unknown**.
No new play session, Motor replay, chart scanning, human log
interpretation, model learning, Validation20 or FINAL is performed.

## Existing historical source and provenance

Read only `reports/dmdogo-v09-h06-improved-suffix-r2.json`,
committed Git blob SHA1:

`67d8f63086f59c086c3724bc3284fe89178f85f2`.

It is the frozen v0.8, untrained H0.6 conditional forced-suffix
experiment, not the later H0.8 full quota8 search. The source
contains 3 already-selected Train charts, 10,231 events and
22 independently targeted seam counterfactuals. Seven have a
strictly lower **frozen six-term heuristic cost**, fifteen do
not. The complete exact alternative assignment sequence is
**not stored** in the H0.6 report: only certain join-axis
descriptors, cumulative metrics and changed-region summaries.

These 22 target interventions are alternative worlds, not 22
independent physical runs or 22 disjoint charts. In particular
the seven wins must not be combined into a playable chart.

## Validation ladder and evidence types

| Claim | Evidence available from H0.6 | H1.0 conclusion |
| --- | --- | --- |
| Alternate lower frozen objective | 7/22 | Descriptive yes, for those seven |
| Abstract >=80ms same-axis reuse | Checked on saved assignments | Abstract-only yes |
| Original v0.8 physical feasibility | Reported in separate previously verified Train runs | Not independently reverified here |
| Alternate physical Motor feasibility | No alternate Motor run | **Unknown** |
| Human key/finger observations | None linked in H0.6 | **Unknown** |
| Human preference / ergonomics | No aligned independent judgments | **Unknown** |
| HUD-only Student ability | Planner uses Oracle times | **Not tested** |

The existing human-recorder parser produces **declared key mappings**
from a legacy session and does not independently verify which finger
pressed which key. Its clock and oracle events are not aligned.
Treating this data as human ground truth would be invalid.

Do not label a smaller cost, a removed Quick flag, a smaller
hand-switch count, a longer reuse interval, or a physical
v0.8 Hit rate as *human-preferred*. These are separate hypotheses.

## H1.0 output

`scripts/audit_dmdogo_h1_quality_evidence.py`:

1. Requires H0.6's exact source blob and original frozen commit.
2. Rechecks all 3 chart SHA256s, 22 unique event-pair IDs,
   7 strict improvements and original 10,231-event census.
3. Rechecks six-term cost arithmetic for every target; compares
   quick/rapid flag additions/removals, changed assignment
   coverage and the >=80ms abstract reuse gate.
4. Writes one record for **every** target: 7 better and 15 not
   better, with declared status of unobserved alternate Motor
   and unobserved human quality.
5. Labels every candidate `blind_review_ready=false` because
   a fair human A/B assessment cannot be made from the partial
   original-vs-counterfactual summaries alone.
6. Does not change how H0.6, H0.8, v0.8, frozen Timing or
   physical Motor computes candidates or score.

## Follow-on H1 design — evidence required, no run authorized yet

**H1.1: paired schedule reconstruction.** Independently rebuild
the complete ordered source and alternative event-to-axis
assignments for each of 22 independent H0.6 counterfactual
worlds, using the original exact Train files and H0.6 source
checks. Fail closed if any source SHA or candidate path differs.
Do not use the already inspected artificial H0.12–H0.17
trials as supposed human observations.

**H1.2: blinded qualitative evaluation.** Only with genuine
human review, present the same chart span (including sufficient
downstream changed context) for A/B in a counterbalanced order,
without revealing objective costs. Predeclare a rubric that
distinguishes preference, physically possible hand movements
and uncertainty. No labels are manufactured or inferred from
legacy K01..K16 mappings.

**H1.3: offline physical feasibility.** When explicitly
authorized, run separately verified frozen Motor episodes
for each alternative, using exactly the same chart and
source-policy identity. Report every failure, local replan,
deferral, overload, timing error and exact final key assignment.
Original v0.8 complete runs cannot substitute for alternate
run results. No FINAL/Validation20 unless separately authorized.

Keep the experiment, quality rubric, any future human inputs
and all benchmark outcomes versioned separately. Human
preference must never be optimized and evaluated on the same
demonstrations without an independent split.

## Run on Windows PowerShell

Stay on `develop/dmdogo`. Preserve all local edits, notably
`configs/training/prestige1.toml`, and every saved report.
Only fast-forward pull; do not reset, clean, stash, rebase or
force-push. The script reads only its source JSON.

~~~powershell
cd F:\dev\DoMachinesDreamOfDeparture
git pull --ff-only
uv run pytest -q tests/test_dmdogo_h1_quality_evidence.py

uv run python scripts/audit_dmdogo_h1_quality_evidence.py `
  --h06-report reports/dmdogo-v09-h06-improved-suffix-r2.json `
  --output reports/dmdogo-h1-quality-evidence-r1.json
~~~

The report uses exclusive creation (`open('x')`). Change
revision filename on a rerun, never overwrite prior evidence.

## Stop/go decision

H1.0 can complete with all evidence gates marked unknown.
That outcome is intentional; it means the project has the
right inputs to plan quality validation without pretending
to have measured actual human preferences. A shift to
H1.1 should be explicitly treated as **offline path
reconstruction**, not human-like fingering validation.
