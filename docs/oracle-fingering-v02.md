# DMDOD — Oracle-assisted 12-axis / 16-contact Physical Fingering Teacher v0.2

## Purpose

Experimental **physical fingering teacher**, not a learned Fingering Network. Exact AutoPlay timestamps are available only in the privileged teacher and physical scoring. The original HUD-only Student, timing network, checkpoints, and Stage A gate are not modified.

v0.2 expands the previous 8-key v0.1 pilot to the experimental `Coupled16ContactMotor`: 12 coupled physical axes, 16 logical key contacts. Names in candidate logs refer to *physical axes* (`left_1`…`left_6`, `right_1`…`right_6`), not actual keyboard bindings. Each hand has four ordinary fingers, a thumb axis (`tip`=1 contact / `flat`=2 contacts), and a coupled edge axis (2 contacts). The model is **simplified human anatomy**, not validated human motions.

## Evaluation method

- 10ms control, 1ms physical simulation using the **existing frozen motor profile**.
- When an AutoPlay event enters the 44ms motor lead window, enumerate physically eligible first-event finger allocations.
- For each first allocation, explore at most three diverse second-event continuations and one deterministically selected third-event continuation. Lookahead is at most 3 events and 155ms beyond the first event.
- For each complete proposal, **deep-copy the identical live physical Motor state**, simulate all its assignments to the final due time + 35ms, and measure true simulated physical logical KeyDown times, missing presses, extra presses, and absolute due-time error.
- Cost is `150 * missing + 120 * extra + sum(abs(KeyDown_time - due))` in ms-equivalent units. Penalties are exploratory heuristics, not a learned reward and not the native ADOFAI accuracy function.
- Pick the lowest-cost first allocation (deterministic tie-break), **commit only that first assignment**; future assignments are hypothetical and recomputed on the next control steps.
- The real continuing coupled Motor and the unchanged real-chart scorer determine hits/misses. Separate counterfactuals never modify the live state. Only the teacher and evaluator access target timestamps.
- The event-level report lists *every tested first allocation*, its rank and best physical continuation, and the actual adopted finger and KeyDown. The `*.html` viewer includes a clickable twelve-axis timeline; CSV lists the adopted sequence; JSON preserves all shadow candidate outcomes.

## Windows run

Copy the new files to their corresponding paths inside the DMDOD repository:

- `scripts/audit_oracle_fingering_v02.py`
- `tests/test_oracle_fingering_v02.py`
- `docs/oracle-fingering-v02.md`

From the repository root:

```powershell
uv run pytest -q tests/test_oracle_fingering_v02.py

uv run python -u scripts/audit_oracle_fingering_v02.py `
  "F:\path\to\命日.adofai" `
  --end-s 35.6 --focus-from-s 35.0 --depth 3 `
  --output diagnostics/oracle_fingering_u9_35to35_6_v02.json
```

The experiment replays the chart physically from t=0. Before `focus-from-s`, it uses a **simple deterministic center-first prefix** to establish the state; that prefix can generate hits or misses and should not be credited to the optimized teacher. `--focus-from-s 0` optimizes from the start. Output `.json`, `.html`, `.csv` are **write-once** and must not replace existing files. Open `.html` in any browser; no server needed.

## Local evidence and limitations

- U9 `命日.adofai` first 15 seconds: 37/37 Hit, all single-event lookahead because note gaps are >155ms.
- U9 physical rollout 0..35.6 seconds with optimization only after 35.0: 203/204 full-prefix Hit, with 1 miss **before** the optimized interval; all 15 audited high-density events hit. 11 were evaluated at full 3-event depth, 3 at 2-event depth, and 1 at 1-event depth.
- These are **oracle-assisted simulation observations**, not a full-chart U9 clear or a comparison to the 99.83% historical best. No learned Fingering Network has been trained.
- The beam and contact topology are simplified. Ranking is *not an exhaustive global optimum*. Third-event branch choice uses one deterministic continuation. Longer-range coordination, real biological finger mapping and native Unity Multipress are not established.
- No prior Student checkpoint was read or modified, no frozen Stage A gate bypassed, and no Qualification20/Reserve20/Final selection was used.
