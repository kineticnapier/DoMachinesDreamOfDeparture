# DMDOGO v0.6 — Pattern-first physical Motor handoff

## Scope

The abstract scheduler remains independent from Network, fingers, fatigue and
Motor mechanics. The new research-only adapter and runner test whether its
periodic logical-key schedule can be executed by the existing frozen
12-axis / 16-contact physical Motor.

The uploaded unfinished adapter and runner were used as the starting point.
The GitHub copies add a bounded physical selection decision, explicit deferral
reasons, create-once replay outputs, and a testable CLI. No source charts,
checkpoints, Motor constants or Oracle timing were changed.

### Discrete-to-physical handoff

The default fixed mapping uses eight **ordinary** physical axes:

- key_0 → left_1; key_1 → right_1
- key_2 → left_2; key_3 → right_2
- key_4 → left_3; key_5 → right_3
- key_6 → left_4; key_7 → right_4

These are mathematical labels, not verified human anatomical bindings.
The adapter rejects duplicate physical axes, invalid axis names and
duplicate-key chords. It does not treat the other eight logical contacts
as independent physical actuators.

Each event begins consideration within the existing 90 ms Motor prediction
window. The mapped allocation is always attempted first:

1. If **live eligible**, commit the desired ordinary-axis allocation.
2. If currently unavailable and the event is still more than 44 ms away,
   **defer**, recording the physical eligibility reason. Do not substitute
   a different axis yet. Retry at the next control frame.
3. At or inside the existing 44 ms legacy lead window, try the desired
   allocation followed by a bounded, deterministic list of ordinary-axis
   alternatives. Every alternative is checked against **live eligibility**.
4. If no alternative is eligible, make no unsafe physical commitment,
   record the reason, and allow a later retry; unresolved event indices are
   reported explicitly if the run ends without an assignment.

This does **not** prove that abstract 80 ms reuse spacing is physically
sufficient. A chosen substitute can break the original three-key cycle.
The physical episode and its telemetry are the source of truth for Hits.

The runner uses the existing v0.4 state-aware Motor lead compensator,
without Network scoring, Teacher candidate ranking or future physical
observations. Physical Judge data and the Motor are unchanged.

## Run on the user's Windows checkout

All original user-local untracked v0.6 files are a potential **git pull
collision** once these same paths exist in the remote commit.

Do **not** run `git clean` or `git reset`.

Safely retain the originals outside the repository first. The backup
commands below copy rather than delete any original file:

```powershell
$backup = Join-Path $env:TEMP ("dmdogo-v06-handoff-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $backup | Out-Null
Copy-Item scripts/run_dmdogo_v06_pattern_first.py $backup/
Copy-Item src/dmdod/planning/pattern_motor_adapter_v06.py $backup/
Copy-Item tests/test_pattern_motor_v06.py $backup/
```

To permit `git pull --ff-only` without overwriting the old untracked
files, relocate the originals to those backups:

```powershell
Remove-Item scripts/run_dmdogo_v06_pattern_first.py
Remove-Item src/dmdod/planning/pattern_motor_adapter_v06.py
Remove-Item tests/test_pattern_motor_v06.py
git pull --ff-only
```

Only do this after verifying that **all three files were successfully
backed up** in `$backup`. These removals are of the copied original local
untracked paths, not project history. If preserving the working-tree
copy without any deletion is required, use `Move-Item` instead of the
copy/removal sequence.

Focused test:

```powershell
uv run pytest -q tests/test_pattern_scheduler_v06.py tests/test_pattern_motor_v06.py
```

Small diagnostic fixture / chart prefix:

```powershell
uv run python scripts/run_dmdogo_v06_pattern_first.py `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai" `
  --output-dir artifacts/dmdogo-v06/bool-prefix-r1 `
  --status-file logs/dmdogo-v06-bool-prefix-r1.status.json `
  --stop-after-s 10 --wall-timeout-s 180
```

Full chart after the focused tests and prefix pass:

```powershell
uv run python scripts/run_dmdogo_v06_pattern_first.py `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/04042 - bool - vo[ICE].adofai" `
  --output-dir artifacts/dmdogo-v06/bool-full-r1 `
  --status-file logs/dmdogo-v06-bool-full-r1.status.json `
  --wall-timeout-s 1200
```

Both output and status paths are create-once. Use new revision numbers
if they already exist. Outputs: `summary.json`, `telemetry-v06.json`,
`replay-v06.json`, and the self-contained `replay-v06.html`.

## Verification status / limitations

Local isolated Motor-mapping adapter tests: 5 passed. They do **not**
test the complete DMDOD simulation. Full repository pytest and physical
full-chart replay have **not** been executed in this environment. The
uploaded physical runner was unfinished and should not be represented as
an already verified full-chart implementation.

The replay's chosen-axis nearest physical KeyDown is a diagnostic
association only; physical outcome and X Accuracy use the unchanged
scorer. For comparison with v0.5, obtain a separate independent Motor
episode on the same chart SHA; no v0.5 metric is inferred from the
abstract three-key result.
