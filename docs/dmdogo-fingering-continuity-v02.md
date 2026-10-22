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

## Fixed bool result

The fixed full-chart A/B used chart SHA256
`04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1`
and the unchanged `fingering_network_v01-4fa2ae1.pt` checkpoint. Both arms ran
the complete 1,122-event chart in independent Motor episodes.

| Metric | Free | Continuity |
|---|---:|---:|
| Hit | 1,122 | 1,122 |
| Miss / TooEarly / Overload | 0 / 0 / false | 0 / 0 / false |
| X accuracy | 94.059715% | 91.671123% |
| Timing MAE | 16.699232 ms | 17.255505 ms |
| left/right switches | 526 | 405 |
| thumb uses | 228 | 64 |
| side uses | 0 | 0 |
| fast-pattern stability | 0.615385 (13 comparisons) | 0.853659 (41 comparisons) |
| same-axis reinput <=35 ms | 0 | 0 |

The two independently executed arms chose different allocations at 944/1,122
events (84.14%). Within the Continuity arm, 440 decisions (39.22%) differed
from the Free argmin computed in that arm's own diverged Motor state. These are
different denominators and must not be conflated.

This result meets Hit non-regression and materially reduces hand switching and
auxiliary thumb use. It does **not** improve every metric: X accuracy fell by
2.3886 percentage points and Timing MAE increased by 0.5563 ms. The result is
therefore evidence for improved selection continuity under the frozen physical
success criterion, not evidence that the new selector dominates Free on timing
quality or resembles measured human fingering.

Focused Selector/runner/viewer regression: 33 passed. Full repository
regression: 2,113 passed with one unrelated existing PyTorch tensor-conversion
warning.
