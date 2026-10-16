# Stage T — real deterministic simulator HUD audit through 20,000 BPM

## Exact scope

This new experiment uses the actual DMDOD ADOFAI chart compiler,
playable target scheduler, N-key physical simulation, and 263D HUD
encoder. It does not invent synthetic orbit angles.

Fixed BPM values:

    600, 1920, 2380, 2999, 3000, 3001, 4000,
    6000, 8000, 10000, 12000, 16000, 20000

Each chart is checked both CW and CCW (a Twirl event on floor 1
enables CCW), giving 26 nonrandom, predetermined fixtures.
They use straight 180-degree floors and v2 unclipped HUD BPM.

There is NO learning, hyperparameter sweep, Motor policy, user
dataset, Validation or FINAL evaluation.

## Actual renderer and physical-clock measurements

1. Compile ordinary JSON ADOFAI charts through parse_adofai_text,
   compile_adofai, and build_playable_segment.
2. Run an actual 8-key DiagnosticHudNKeyRealChartMotorEnv with
   physics/control sampling 1ms, zero physical key commands.
3. Record actual HUD-rendered orbit and 263D input per 1ms step.
   Take every tenth frame to represent the normal 10ms observation.
4. Run the unmodified v2 RPM-guided projective estimator on 10ms
   frames, and the same read-only estimator at explicit 1ms sample
   rate for a diagnostic high-resolution angular recovery probe.
5. ONLY AFTER predictions are made, consult compiled floor labels to
   assess same-floor signed angle reconstruction, and compiled target
   timestamps to assess 10ms target crossing count / conditional
   next-target Timing error / unsafe multi-target candidate claims.

Report both correct angular steps and abstentions, rather than
confusing candidate coverage with proof of gameplay success.

## Observability limit

At 20,000 BPM, ordinary 180-degree floor spacing means one target
every 3ms. The 10ms Student interface skips about 3.33 targets per
sample. A correct aggregate angular speed does not uniquely identify
which intermediate notes must be pressed or their exact order.

At 1ms, 20,000 BPM moves 60 degrees. For a pair truly on the same
ordinary floor, this can be recovered from 2x-angle RP1 together
with unclipped Tile BPM. But 1ms HUD observation remains diagnostic:
the original Student still controls and observes at 10ms.

The report flag unsafe_for_adoption indicates a phase error OR any
multi-target crossings in ordinary 10ms observations. A flag does
not imply that all conditional Timing predictions are incorrect.

## Preservation

Original Stage A FAIL, frozen Geometry/Hybrid and all neural checkpoints
are untouched. No new training, key presses, Validation or FINAL.
Only a distinct report file is created; replacement is refused.

## Windows

~~~powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_autoplay_high_bpm_simulator_audit.py
uv run python -u scripts/audit_prestige1_autoplay_high_bpm_simulator.py --config configs/training/prestige1.toml
~~~

Output:

    diagnostics/prestige1/stage_timing_simulator_20000bpm_readonly.json

Passing test fixtures does NOT establish full 20,000 BPM Motor playability.
