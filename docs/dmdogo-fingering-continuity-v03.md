# DMDOGO v0.3 — Fingering Continuity & Timing Optimization

## Scope and frozen components

Fingering Network v0.1, checkpoint, AutoPlay timing, candidate generation,
physical eligibility, 12-axis/16-contact Motor constants, scorer, HUD and
Viewer v0.3 remain frozen. Continuity v0.2 remains available and unchanged.
No training, Teacher rollout, Validation20 or FINAL data is used.

## v0.2 cause audit

Source replay:
`artifacts/dmdogo-continuity/bool-full-r1/replay-v02.json`.
The audit is saved as
`reports/dmdogo-fingering-continuity-v03-cause-audit-r2.json`.

### Event/KeyDown correspondence

Both arms have 1,122 single-contact events, 1,122 physical KeyDowns, 1,122
Hits, zero Miss/TooEarly and no overload. Under the repository scorer's FIFO
contract, judgement order therefore gives a unique event/KeyDown mapping for
all events. A nearest-KeyDown heuristic is not used.

The selected axis and FIFO-corresponding physical KeyDown axis differ at six
Free events and two v0.2 events. This is valid evidence of in-flight Motor
ordering, and is why the audit keeps “intended allocation” separate from
“judged physical KeyDown”. Outside the complete-Hit FIFO condition, the tool
returns all axis/time-window candidates and labels multiple candidates as
ambiguous.

### Where accuracy was lost

| travel BPM | events | Free X | v0.2 X | Free MAE | v0.2 MAE |
|---:|---:|---:|---:|---:|---:|
| 480 | 737 | 97.17% | 96.01% | 14.41 ms | 14.58 ms |
| 1920 | 384 | 88.07% | 83.32% | 21.13 ms | 22.43 ms |

The 1920 BPM region is an exact 31.245 ms cadence. It contributes about 68%
of the full-chart X-Accuracy loss despite containing about 34% of events.
This is the region where v0.2's scale-free 65/35 rank blend can override a
Network candidate while several axes are still dynamically coupled.

Allocation-change categories provide causal candidates, not isolated causal
effects, because each changed selection alters later Motor state:

| category | events | mean MAE delta (v0.2 − Free) |
|---|---:|---:|
| thumb → ordinary | 217 | +1.28 ms |
| other changed | 674 | +0.91 ms |
| unchanged axis | 178 | +0.68 ms |
| ordinary → thumb | 53 | −7.30 ms |

The favorable ordinary-to-thumb cases and unfavorable thumb-to-ordinary cases
show that a blanket thumb penalty is physically unsound. The saved trace
contains position, velocity and pressed state at 10 ms, but not per-axis
activation. Activation is therefore not claimed as an observed cause.

## Continuity v0.3 rule

v0.3 makes one fixed change. When the gap from the previously committed event
is at most 35 ms, it restricts the v0.2 continuity ranking to the frozen
Network's top two currently eligible candidates. Within that guard it retains
the exact v0.2 continuity cost and rank blend. At larger gaps, it is identical
to v0.2.

This rule:

- preserves eligibility and causal history;
- allows a near-Network rolling alternative;
- prevents a style preference from selecting a low-ranked Motor-state option
  in the 31.245 ms region;
- does not hard-code BPM, chart identity, true event outcome, or future state;
- leaves v0.2 code paths and the default `free` policy intact.

The fixed top-two rule was selected from the causal diagnosis before the v0.3
full-chart run. No weight or threshold sweep is performed.

## Reproduction

Cause audit:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_fingering_continuity_v03.py `
  --replay artifacts\dmdogo-continuity\bool-full-r1\replay-v02.json `
  --output reports\dmdogo-fingering-continuity-v03-cause-audit-r2.json
```

Three independent Motor episodes and Viewer v0.3:

```powershell
.\.venv\Scripts\python.exe scripts\run_fingering_continuity_comparison_v03.py `
  --chart "data\DMDOD-v140-tuf-p11-p20\Train\04042 - bool - vo[ICE].adofai" `
  --checkpoint checkpoints\fingering\fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts\dmdogo-continuity\bool-v03-r1 `
  --status-file logs\dmdogo-continuity-bool-v03-r1.status.json `
  --wall-timeout-s 900
```

Both tools refuse to overwrite existing results.
