# Stage T — 12-actuator / 16-contact Motor in the real-chart evaluator

**Scope:** isolated, read-only **Motor-only oracle** experiment. This is **not** a trained Student or a 16K HUD feature contract. Frozen 8-Key / 263D Student, Stage A FAIL, Geometry/Hybrid, MaleCNS4096, original Motor physics, Validation and FINAL are untouched.

## Integration

The new `CoupledContactRealChartEnv` subclasses the existing `RealChartMotorEnv` **without modifying** its scoring rules. A `Coupled16ContactMotor` has 12 physical command axes (four fingers, one shared thumb, one hand edge per hand) and generates 16 logical contact switches. It sends physical KeyDown/KeyUp events through the **normal complete `env.step()` lifecycle**: overload gauge progression, miss expiry, hit classification, reward, termination, and 10ms control/1ms physical integration. The observation includes the separate 12-axis physical state, 16 logical press flags, and renderer-visible chart geometry. It does **not** provide an absolute floor index or target timestamps to a policy.

The fixed `ContactOracleTeacher` uses **privileged target times and ordinals intentionally** to verify capacity and integration. It assigns each upcoming event to a free motor axis. An axis with two coupled contacts is only reserved for two consecutive targets at most 20ms apart, to avoid intentionally producing single-note contact surplus at low BPM. Reservations persist as the one physical Motor continues throughout the chart. It uses a predeclared 44ms lead, round-robin finger allocation, and fixed thumb posture for each episode (4K / 5K / 6K / 7K / 8K per hand). This is a simplistic deterministic teacher, **not optimal**.

The 16 contact labels are not 16 independently movable fingers. Flat thumb produces the two contacts `hand_5,hand_6` on a **single** actuator; the hand edge produces `hand_7,hand_8` simultaneously on a **single** actuator. Real human finger geometry, pose-switching dynamics and fatigue are not validated.

## Fixed complete-chart runs

**All input charts are generated, compiled ADOFAI-format straight-floor fixtures, not held-out songs.** Each row plays to episode completion using the same unchanged evaluator and Motor dynamics. All runs use the same evaluator difficulty.

| BPM | Playable targets | 4K per hand | 5K | 6K | 7K | 8K |
|---:|---:|---:|---:|---:|---:|---:|
| 600 | 24 | 24 Hit | 24 Hit | 24 Hit | 24 Hit | 24 Hit |
| 1,200 | 48 | 48 Hit | 48 Hit | 48 Hit | 48 Hit | 48 Hit |
| 4,000 | 72 | 60 Hit | 72 Hit | 72 Hit | 72 Hit | 72 Hit |

At **4,000 BPM**, the 6K teacher assigned **12** two-contact thumb reservations; 7K assigned **10** edge-pair reservations; 8K assigned **18** mixed two-contact reservations. This demonstrates that the physical adapter's paired contacts are actually being used in a continuous chart environment. The 5K mode gets full Hits without pair contacts under this generous oracle/normal-judgement benchmark, so the results do **not** establish the necessity of 16 contacts.

**Important evaluator limitation:** the current DMDOD scorer can credit two simultaneous physical KeyDowns to **different successive targets** (even 15ms apart at 4,000 BPM). Multipress/OverPress queue semantics are not fully reproduced. Thus, counts can be optimistic. In particular a 72/72 Hit result is *not* perfect X-Accuracy, a human 16K clear, Student transfer, or evidence that the physical body can make two independent timestamped strikes with the thumb/edge. Record the total contact KeyDown count separately from Hit count and never treat them interchangeably.

### End-to-end regression guards

- At 600 BPM a *single* edge actuator produces **two simultaneous real KeyDowns** through `env.step()`, scoring **1 Hit + 1 TooEarly** against the original real-chart evaluator.
- At 4,000 BPM the same two contacts can be scored **2 Hits** at an identical timestamp because the current scorer lacks complete Multipress behaviour.
- The 12-physical-axis state and 16 logical press flags are distinct; no target oracle is included in `ContactChartObservation`.
- Five per-hand static contact modes each execute to completion using the **same continuously simulated Motor**, without an intermediate reset.
- Re-running the report refuses overwrite. Existing 8K Student and test fixtures are not modified.

## Run

```powershell
uv run pytest -q tests/test_prestige1_autoplay_coupled_chart_audit.py
uv run python -u scripts/audit_prestige1_autoplay_coupled_chart.py --config configs/training/prestige1.toml
```

Output (new, immutable):

`diagnostics/prestige1/stage_t_coupled_16contact_real_chart_teacher_readonly.json`

## Next separate step

Once the current same-timestamp limitations are accepted as an experimental constraint, measure **true planned-target vs physical KeyDown timing**, unexpected coupled contacts and TE/Overload per note under mixed patterns. Then consider a **16-contact HUD observation design** and a deliberately new Student. Do **not** overwrite the fixed 263D/8K learning configuration or present privileged teacher performance as Student ability.
