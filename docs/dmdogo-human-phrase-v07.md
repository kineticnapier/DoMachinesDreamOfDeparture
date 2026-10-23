# DMDOGO v0.7 — Rhythm-phrase fingering hypothesis

## Motivation

v0.6 proved that three-key regular schedules could complete all 1,122
Train bool targets with zero Miss/TooEarly/Overload, but the full physical
run used 233 local replans, 626 deferred events and X Accuracy 92.362%.
An abstract three-key loop does not explain human chart parsing or hand
gesture selection.

**v0.7 does not claim to be human-like yet.** The repository contains
a human-log comparison script and a prior *Human-Proxy* experiment, but no
verified aligned human demonstration sequence on this branch. The prior
Human-Proxy was an ordinal style prior, not learning from actual players.

The testable v0.7 step is to separate:
1. **Rhythm analysis**: discover repeated inter-onset interval motifs.
2. **Phrase planning**: choose a named, auditable gesture template for a
   bounded timing-derived planning window.
3. **Physical verification**: route proposed ordinary axes through the
   unchanged v0.6 live Motor eligibility and v0.4 lead compensator.
4. **Empirical human validation (not implemented yet)**: compare against
   permissioned, time-aligned human input traces and validate out of sample.

## Source/observability contract

The analyzer reads **exact compiled Oracle event times and chord sizes**.
These are privileged offline labels, not HUD-only observations. No Network
scores, future Motor states, human logs, future physical KeyDowns or future
judgements enter the planner. This is an offline fingering research arm,
**not** a claim that a Student can infer the same plan from visible HUD.

There is no fingertip fatigue, anatomy, individual preference or keyboard
finger calibration encoded. Eight identifiers are the existing eight
ordinary physical Motor axes, not calibrated human joints. Thumb/side
contacts and coupled 16-contact relationships are not bypassed.

## Rhythm vocabulary

For each timing-derived planning window, inspect the inter-onset gaps:

- `even_dense` / `even_spaced`: a repeated single gap;
- `alternating_gaps`: distinct short/long two-gap recurrence;
- `periodic_3`, `periodic_5` etc.: repeated 3- to 8-gap motifs;
- `chord`: any event with more than one required contact;
- `irregular`: no supported repeated motif;
- `short_or_isolated`: insufficient gaps to classify.

Periodic gap recognition is an observed property of the compiled timing
sequence. It does **not** mean a player will actually use the corresponding
finger pattern. The current 16-event window size, 450ms pause threshold
and 6% timing tolerance are declared researcher-selected conventions; do
not interpret a window boundary as a measured musical phrase boundary.

## Explicit gesture hypotheses

The v0.7 planner offers three physical-axis gesture templates:

- `interleaved`: L1 R1 L2 R2 L3 R3 L4 R4;
- `paired_hand`: L1 L2 R1 R2 L3 L4 R3 R4;
- `hand_roll`: L1 L2 L3 L4 R1 R2 R3 R4.

Both hand mirrors and eight cyclic phase choices are evaluated for each
window. The motif class orders these templates. A bounded deterministic
beam retains abstract reuse-safe alternatives across window boundaries.
It is **not a global optimum** or an imitation-trained policy.
A fixed ≥80ms same-axis constraint is an abstract assumption, not a
physiological measurement. The live Motor must still check pressed,
reservation, coupling, and actual timing.

## Reproducible off-line bool audit

The received v0.6 full replay has chart SHA256
`04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1`,
1,122 single-contact onset events. On this source, the offline v0.7
research prototype produced:

- 71 bounded planning windows;
- 4 alternating-gap, 12 periodic-3, 30 periodic-5,
  24 even-dense, 1 short/isolated windows;
- 15 changes of preferred gesture family;
- zero abstract same-axis reuse violations;
- minimum abstract axis reuse 187.468767ms;
- 982/1,122 different proposed allocations from v0.6's saved physical
  allocations, **not** 982 physical improvements.

**The v0.6 scorer's 1,122 Hits, 92.362% X and 13.363ms MAE cannot be
attributed to v0.7.** The offline audit deliberately outputs
`v07_hits: null` and `v07_physical_motor_evaluated: false`.

Run against the existing saved v0.6 replay:

```powershell
uv run python scripts/audit_human_phrase_v07.py `
  --input-replay artifacts/dmdogo-v06/bool-full-r1/replay-v06.json `
  --output reports/dmdogo-human-phrase-v07-bool-r1.json
```

Only a new file path is accepted; existing reports are never overwritten.

## Physical study — opt-in and separate from v0.6

The existing v0.6 command is unchanged by default: `--planning-mode
pattern_v06`. To run the new **research hypothesis** in an independent
Motor environment, specify `--planning-mode phrase_v07`.

First run focused tests and a small diagnostic prefix:

```powershell
git status --short --untracked-files=no
git pull --ff-only
uv run pytest -q tests/test_human_phrase_planner_v07.py tests/test_human_phrase_audit_v07.py tests/test_human_phrase_physical_v07.py tests/test_pattern_motor_v06.py

uv run python scripts/run_dmdogo_v06_pattern_first.py `
  --planning-mode phrase_v07 `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai" `
  --output-dir artifacts/dmdogo-v07/bool-prefix-r1 `
  --status-file logs/dmdogo-v07-bool-prefix-r1.status.json `
  --stop-after-s 10 --wall-timeout-s 180
```

The `git pull --ff-only` is appropriate only if local changes do not
overlap the updated runner. Preserve all unstaged changes; never reset,
stash, clean, branch, merge or rebase to force this update.

After examining prefix failure and saved telemetry, the optional
full-chart command is:

```powershell
uv run python scripts/run_dmdogo_v06_pattern_first.py `
  --planning-mode phrase_v07 `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai" `
  --output-dir artifacts/dmdogo-v07/bool-full-r1 `
  --status-file logs/dmdogo-v07-bool-full-r1.status.json `
  --wall-timeout-s 1200
```

Output: `summary.json`, `telemetry-v07.json`,
`replay-v07.json`, `replay-v07.html`. Prefix and full output names are
create-once; no prior artifacts are overwritten.

Do **not** run Validation20 or FINAL.

## Completion definition

Current status: phrase recognizer, hypothesis planner, offline audit,
opt-in Motor runner and focused regression tests added; the latter still
need execution on the user's Windows/physics environment.

The next meaningful human-like result requires a **real human
demonstration with a verified key-to-physical-axis mapping and stable
event alignment**, followed by held-out evaluation. Matching template
counts or improving Hit alone cannot establish human imitation.
