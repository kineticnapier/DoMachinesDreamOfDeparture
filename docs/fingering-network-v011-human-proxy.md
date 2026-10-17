# Fingering Network v0.1.1 — Human-Proxy comparison protocol

This is a fixed-checkpoint inference experiment.  Human-Proxy is an untrained,
ordinal style prior and must not be described as a model learned from human
demonstrations.

## Motor abstraction

Each hand has six physical axes and eight logical contacts:

- axes 1--4: independent ordinary-finger actuators;
- axis 5: one thumb actuator, with tip (`contact 5`) or flat
  (`contacts 5+6`) topology;
- axis 6: one coupled hand-side actuator (`contacts 7+8`);
- contacts 1--8 are simulator contact identifiers, not verified rows or OS
  keyboard assignments.

The Motor does not encode calibrated index/middle/ring/little anatomy.  For
this policy only, axis 1 is declared an index-finger proxy, axes 2--4 other
ordinary-finger proxies, axis 5 thumb, and axis 6 hand side.  Viewer aliases
such as `Tab` and `LShift` remain presentation labels and do not change this
topology.

## Fixed policy

Free is the existing frozen Network selection.  Human-Proxy converts eligible
Network scores and finite style costs to deterministic within-event percentile
ranks, then blends them:

`combined = (1 - alpha) * network_rank + alpha * style_rank`

`alpha=0` delegates exactly to Free.  `alpha=1` maximizes the finite style
preference but does not ban thumb/side candidates.  Physical eligibility still
rejects pressed, reserved or unsafe-topology axes.  The comparison alpha is
fixed at `0.5` before the full run; no validation/evaluation search is allowed.

The experiment uses the P16 `03640 - 東` Train chart, exact AutoPlay event
times, the read-only v0.1 checkpoint, 10ms control and 1ms physics.  Free and
Human-Proxy receive independent Motor environments.  Run a two-second smoke
test before one full comparison.  Miss, TooEarly, Overload and degradation are
reported without rerunning at a different alpha.

For long runs, use `--mode single --policy free` and `--mode single --policy
human_proxy` with separate create-once output/status paths.  Then use `--mode
assemble --free-result ... --human-result ...` to build the comparison without
rerunning either policy.  This keeps each physical process below the execution
session lifetime and leaves a resumable artifact after each policy.

```powershell
.\.venv\Scripts\python.exe scripts/run_fingering_network_v011_comparison.py `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/03640 - 東 - prisMKa _ Sprout.adofai" `
  --checkpoint checkpoints/fingering/fingering_network_v01-4fa2ae1.pt `
  --output-dir artifacts/fingering_network_v011/full-NEW `
  --status-file logs/fingering-network-v011-full-NEW.status.json `
  --wall-timeout-s 300
```
