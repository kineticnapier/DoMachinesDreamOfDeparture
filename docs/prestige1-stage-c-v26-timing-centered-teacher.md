# DMDOD Prestige I — Stage C v2.6 Timing-Centered Actuator Teacher

## Why

v2.4 reached Train20 SAFE=20/20, Hits=850/850, TooEarly=0, Miss=0,
Overload=0, with 136 safety-braked commands. Stage C v2.5 exposed:

- all 850 valid Hits occurred early (signed mean -58.957ms, MAE 58.957ms);
- only 225/850 (26.47%) were in the Perfect judgement family;
- legacy chronological target labels differed from the physical unresolved
  target in 4,575/15,386 control frames (29.73%).

Safety is not the same as timing precision. This is a **new fixed Train20-only
Teacher candidate**, not Student training or a resumption of old trust-DAgger.

## Controller design

1. Run the **unchanged** v2.4 `ActuatorAwareTeacherSafety` first to suppress
   the original Teacher's forecast TooEarly KeyDowns.
2. Copy all 8-key motor state (including activation, velocity, fatigue, keyboard
   press flags, hand/bilateral coordination) and physical target state.
3. Simulate two choices at 1ms physics resolution for `lead_s + 10ms`:
   - the v2.4 safety action held throughout;
   - one key changed from `+1` to neutral `0` for **exactly the next 10ms**,
     with the original v2.4 safety action held afterward.
4. Keep the deferred action only when:
   - predicted Hit **target identity set is unchanged** (later targets cannot
     disappear to fake better MAE);
   - predicted TooEarly, Miss, TooLate, and no-target KeyDown counts do not grow;
   - no existing Hit worsens by more than one 1ms physics tick (1.01ms);
   - total predicted absolute error improves by at least 1ms.
5. Greedily choose the largest improvement among the available positive
   fingers, then optionally defer another key within the same frame.
6. Timing changes are **neutral pulses only**. No extra `+1 → -1` timing
   braking is introduced; the existing v2.4 shield remains free to brake
   when necessary for safety.

This horizon is an approximation, **not** an exact full Teacher policy
counterfactual: future 10ms Teacher decisions can differ after deferral.
The true Train20 simulator, never the lookahead, determines the outcome.

Note: preserving only the forecasted Hit set does not prove the entire
episode will be SAFE; hence a strict report-level physical gate.

## Immutable reference and report gates

Inputs are frozen v2.3, v2.4, v2.5 Train20 JSON files, original Stage A/C
provenance, Train20 manifest and protected checkpoints. Their hashes must
remain unchanged. The Stage A held-out Train gate remains **FAIL**.

The candidate is compared with v2.4's known 850/850 Hit and v2.5's
58.957ms Hit-only MAE; per-target paired error differences distinguish
shifted/missing targets from timing improvements.

The read-only v2.6 report records:

- full physical aggregate, Hit-only timing precision and Perfect-family rate;
- every anchor's baseline/candidate stats and physical Hit events;
- paired targets with counts improved, worsened and unchanged;
- v2.4 shield interventions, extra neutral defers, command sign transitions
  and direction reversals measured on the same v2.6 replay state;
- three gates: SAFE850 preserved, timing improved without Perfect-family
  regression, and eligibility for **later design review only**.

No automatic adoption occurs, including when both gates pass.
No changed model or checkpoint is saved. Validation and FINAL are never read
or executed. No hyperparameter sweep is performed.

## Windows commands

In the existing branch `develop/v0.1-simulator`, from
`F:\dev\DoMachinesDreamOfDeparture`:

```powershell
git status --short
git pull --ff-only

uv run pytest -q `
  tests/test_prestige1_target_contract.py `
  tests/test_prestige1_actuator_teacher.py `
  tests/test_prestige1_teacher_quality.py `
  tests/test_prestige1_timing_teacher.py

# Only after the CPU tests pass, read-only Train20 candidate:
uv run python -u scripts/run_prestige1_timing_comparison.py `
  --config configs/training/prestige1.toml
```

Output `diagnostics/prestige1/stage_c_v26_timing_centered_teacher.json`
(refuses overwrite). If a local uncommitted change conflicts with pull, do
not reset, stash, clean or discard it.

The new timing Teacher is experimental and is **not** an instruction to run
learning, Validation or FINAL. Decide what to do next using the v2.6 report.
