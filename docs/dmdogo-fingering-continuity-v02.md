# DMDOGO Fingering Continuity v0.2

## Scope

This change keeps Fingering Network v0.1, its checkpoint, AutoPlay event
times, candidate generation, Motor physics and physical eligibility frozen.
It adds an opt-in post-network selector and does not train a model or run a
Teacher counterfactual rollout.

`free` remains the default and is the exact eligible Network-score argmin.
`continuity` only reorders candidates that are already eligible.

## Selector contract

For each event, the frozen Network score and the continuity cost are converted
to ranks within the current eligible set. The fixed blend is 65% Network rank
and 35% continuity rank. This is one predeclared setting, not a hyperparameter
search. Tied costs receive tied ranks so the implementation does not silently
become a candidate-list-order prior.

The causal continuity cost uses only committed earlier choices and the current
event time:

- at gaps of 35 ms or less, continue an observed same-hand 1--4 rolling step;
- discourage immediate reuse of the same physical axis;
- prefer adjacent ordinary axes to large jumps or unnecessary hand changes;
- keep thumb (axis 5) and side (axis 6) finite, usable alternatives;
- treat a multi-contact allocation atomically;
- abstain if no candidate is physically eligible.

The 35 ms boundary already existed in replay diagnostics. It is a diagnostic
definition, not an anatomical claim. Axis names are Motor proxies and do not
establish real keyboard bindings or calibrated human anatomy.

## Metrics

`comparison.json` records physical Hit/Miss/TooEarly/Overload, X accuracy and
Timing MAE separately for both arms. The continuity metrics are:

- `left_right_switches`: hand changes between adjacent single-axis decisions;
- `axis_reuse_interval_ms`: count/minimum/mean/median interval per reused axis;
- `same_part_reinputs_le_35ms`: same-axis reuse at or below 35 ms;
- `thumb_uses` / `side_uses`: allocation counts for axes 5 / 6;
- `changed_from_free_rate`: decisions whose allocation differs from the Free
  arm;
- `fast_pattern_stability`: within uninterrupted fast, same-hand ordinary-axis
  runs, the fraction of transition steps matching either of the preceding two
  step classes. This accepts steady rolls and two-step alternation. It is not a
  claim that fewer finger changes are always better.

The comparison also reports every baseline Hit lost by Continuity. A reduction
in visual irregularity is not a success when baseline Hits regress.

## Commands

Run one policy with the standard self-contained HTML v0.3 viewer:

```powershell
.\.venv\Scripts\python.exe scripts\run_full_chart_fingering_replay_v02.py `
  --chart "data\DMDOD-v140-tuf-p11-p20\Train\04042 - bool - vo[ICE].adofai" `
  --selector continuity
```

Run the fixed Free/Continuity A/B (two independent Motor episodes):

```powershell
.\.venv\Scripts\python.exe scripts\run_fingering_continuity_comparison_v02.py `
  --chart "data\DMDOD-v140-tuf-p11-p20\Train\04042 - bool - vo[ICE].adofai" `
  --checkpoint checkpoints\fingering\fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts\dmdogo-continuity\bool-r1 `
  --status-file logs\dmdogo-continuity-bool-r1.status.json `
  --wall-timeout-s 1800
```

Both commands refuse to overwrite their output. The A/B HTML keeps the v0.3
16-contact counter and allows switching between the two independently executed
policies.
