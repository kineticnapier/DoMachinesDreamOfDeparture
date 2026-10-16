# Stage T — 8K vs 16K *simulated actuator* capacity

Status: **isolated, read-only Motor-only oracle diagnostic**. No Student,
263D HUD controller, checkpoint, training, Stage A gate, Geometry/Hybrid,
Validation or FINAL changes.

## Key limitation: 16K ≠ 16 biological fingers

The current `NKeyBody(16)` creates **16 independently controlled physical
actuators**, eight per hand. It is **not** a representation of a player using
16 keyboard keys with a smaller set of human fingers (e.g. rolling or
alternating between keys). Real 16K fingering cannot be inferred from these
results. The frozen per-hand force budget is shared across actuators, and
the coordination weighting **also changes** when the number of modeled
actuators changes. Even 16-actuator mode limited to its central eight
channels differs physically from `NKeyBody(8)`.

## Three predeclared experimental conditions

1. **8k_motor_8k_allowed:** eight independent simulated actuators; all usable.
2. **16k_motor_8k_allowed:** the 16-actuator body but only its inner eight
   keys may receive nonzero actions. A geometry/coordination control.
3. **16k_motor_16k_allowed:** all 16 independently simulated actuators usable.

Use the **same exact chart bytes** and fixed `CenterFirstNKeyTeacher`
reservation/release algorithm, with a restricted inner-tier selector in
condition 2. Both bodies use the unchanged physical profile: **10ms
control-action hold, 1ms physical substeps**, same global hand-capacity
parameters and the same **44ms** lead. Independent cold calibrations confirm
this lead for both counts. No physical reset between notes.

### Oracle warning

This is *deliberately not a causal HUD-only* planner: the teacher sees the
**exact future note schedule**, as in the project's existing
`diagnose_n_key_teacher_capacity` tool. It is a diagnostic of the combination
of a simple oracle-assisted scheduler and simulated Motor. It is **not an
upper bound on all possible controllers**, nor a Student/16K gameplay result.

## Results (fixed straight-floor real-chart fixtures)

| BPM | Notes | 8-body/8 allowed | 16-body/8 allowed | 16-body/16 allowed |
|---:|---:|---:|---:|---:|
| 1,200 | 144 | 144 Hit | 144 Hit | 144 Hit |
| 2,000 | 144 | 144 Hit | 144 Hit | 144 Hit |
| 4,000 | 144 | 120 Hit | 123 Hit | **144 Hit** |
| 4,000 | 576 | 472 Hit | 479 Hit | **576 Hit** |

At 4,000 BPM, the fully available 16-actuator model used only **10 distinct
actuators** during both 144- and 576-note fixtures, rather than all 16.
This result suggests that the extra recoverable actuators matter under
this scheduler's fixed reservation/release scheme.

**Timing precision is not solved**: at 4,000 BPM / 576 notes, X-Accuracy was
34.37%, 34.79%, and 51.72% respectively. Higher Hit count must not be
reported as perfect timing or as a clear of an unseen chart.

The 16-body/8-allowed geometry control yields 123 vs 120 Hit (144 notes)
and 479 vs 472 Hit (576 notes), confirming that comparing 8 vs 16 bodies
without accounting for geometry/coordination differences is confounded.

## Reproduce

```powershell
uv run pytest -q tests/test_prestige1_autoplay_8k16k_motor_capacity_audit.py
uv run python -u scripts/audit_prestige1_autoplay_8k16k_motor_capacity.py --config configs/training/prestige1.toml
```

Output refuses overwriting earlier reports:

    diagnostics/prestige1/stage_t_motor_only_8k_vs_16k_capacity_readonly.json

## Next step

A physically meaningful *16 keyboard keys / eight finger actuators* model
would need an explicit finger-to-key mapping and per-finger ability to reach,
press, release and transition between more than one key; this is **not**
implemented here. Separately, the frozen 8-Key Student can be investigated
with joint multi-finger trajectory planning, but must not inherit privileged
teacher target timestamps from this Motor-only experiment.
