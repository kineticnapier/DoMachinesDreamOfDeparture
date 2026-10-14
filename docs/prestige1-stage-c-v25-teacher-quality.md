# DMDOD Prestige I — Stage C v2.5 Teacher Timing Quality & Label Alignment

## Scope

This is a **read-only, Train20-only** replay of the fixed v2.4
`ActuatorAwareTeacherSafety`. It is not a new Student training experiment,
nor a Validation or FINAL test. The original Stage A v2 **FAIL** gate,
`gru_near_event_mae_not_20pct_better_than_constant_and_single`, is
unchanged. No checkpoints, original Teacher source, Stage A labels or
263D HUD input are altered.

Before replay the runner verifies:
1. Fixed Train20 manifest, chart hashes, config hash and failed Stage A gate.
2. Stage A/C provenance, old Best SHA256 and other protected checkpoints.
3. v2.3 and v2.4 Train reports and the SHA256 of all originally protected files.
4. That the new v2.5 output file does **not** exist.

For each Train anchor, the v2.4 Teacher is re-run from reset; its actual
physics counts and intervention counts must **exactly match** the immutable
v2.4 report, or the audit stops without writing a report. Input file
digests are compared again after replay, before creating the new JSON.

## Timing quality — real 1ms KeyDown

Use each recorded physical KeyDown's **evaluator-selected next unresolved
target**, not the nearest chronological note. Classify the signed
`keydown_time - target_time` using the physical simulator's own BPM,
pitch, difficulty and judgement method.

For actual Hits only, report:
- mean signed error (early/late bias), MAE, median absolute error,
  95th-percentile absolute error (linear interpolation at `(n-1)*.95`);
- count and fraction of **pure Perfect** hits and **Perfect-family**
  hits (`perfect`, `early_perfect`, `late_perfect`);
- full judgement distribution and each individual Hit's key/index/time/error.

The v2.3 baseline Hit MAE is computed from its **existing physical event
trace**, while v2.4 is independently replayed. **These are different
physical trajectories**: the v2.3 set includes early Overload terminations.
Do not interpret a difference in conditional Hit MAE as a matched
precision improvement without considering survival bias. In particular,
an anchor with no Hits has no numeric MAE, rather than zero MAE.

## Target alignment (v2.2 definitions)

On every 10ms control frame of the v2.4 trajectory, call the original
`compare_targets` without changing either Stage A labels or Motor labels.
Store:
- `legacy_index` from chronological future-note search and
  `unresolved_index` from physical used/missed state;
- matched/mismatched frame counts and mismatch kinds;
- counts when the chronological label points to an already-resolved note;
- original/physical 200ms-near flag and time disagreements when both
  labels are observable, plus observability-mask disagreements.

These counts must be compared to v2.2's 3,389/13,681 **with their frame
denominators**: the v2.4 corrected replay intentionally takes a different
path and lasts longer on the previous Overload charts. A reduced
mismatch rate would not by itself prove the earlier Stage A gate passed.

## Actuator command audit

For every 10ms frame save:
- the unmodified center-first Teacher proposal (8 values);
- the v2.4 corrected action (8 values);
- modified key names;
- original future-event index and physically unresolved index.

Compute action-sign transitions, immediate `+1 ↔ -1` reversals,
reversals across zero-valued frames and changes in positive instruction
count. These values are obtained by comparing unmodified and corrected
proposals **at the same v2.4 physical state**; they are not a
counterfactual playback of the original v2.3 Teacher trajectory.

The safety correction acts only on the Teacher's offline labels. No
Student action is passed physical target index, judgement, miss state,
timing truth or any future note timestamp.

## Files and commands

New:
- `src/dmdod/training/prestige1_teacher_quality.py`
- `scripts/run_prestige1_teacher_quality.py`
- `tests/test_prestige1_teacher_quality.py`

From `F:\dev\DoMachinesDreamOfDeparture`:

```powershell
git status --short
git pull --ff-only
uv run pytest -q tests/test_prestige1_target_contract.py tests/test_prestige1_actuator_teacher.py tests/test_prestige1_teacher_quality.py

# Run only after CPU tests pass:
uv run python -u scripts/run_prestige1_teacher_quality.py `
  --config configs/training/prestige1.toml
```

Output: `diagnostics/prestige1/stage_c_v25_teacher_quality.json`.
The runner refuses to overwrite that file. Do not use reset, clean,
stash or new branches to accommodate a dirty worktree.

## Next decision

Review **physical** precision, label mismatch rate, and additional
sign reversals together. A Teacher can score 850/850 Hits on Train20 while
still producing early-biased, coarse timing labels. The v2.4 Train20
success is a strong safety result, **not proof that a Student can
infer the same timing from visible HUD alone**.

No Student training will start until the user provides the v2.5 report
and explicitly agrees to a subsequent training-stage design.
