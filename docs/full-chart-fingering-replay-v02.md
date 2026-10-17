# Full-Chart Fingering Replay v0.2 protocol

The selected Train80 chart is `03640 - 東 - prisMKa _ Sprout.adofai` (saved
difficulty metadata P16, SHA256
`e23c48c66e956a6e820fff2a5ed352718efb7f5e861094207cc49f6dcabdd622`).
It is 129.634147 seconds with 1,222 AutoPlay events.  Among the four existing
pilot charts it has the fewest events, while retaining 40 SetSpeed actions,
186 Twirls, 27 Midspins and 23 full rotations.  It has no Pause action, so
Pause remains covered by the synthetic viewer test and is reported as not
empirically exercised by this run.

The execution policy uses exact AutoPlay event times, current 12-axis Motor
state and static candidate eligibility.  It does not call Teacher v0.2
counterfactual rollout, read Teacher candidate scores, or continue after a
normal miss/overload termination.  Control and physics steps are fixed at
10ms and 1ms.  The checkpoint is read-only and no training occurs.

Run a bounded smoke test first, then the full run with new output paths:

```powershell
python scripts/run_full_chart_fingering_replay_v02.py --chart "data/DMDOD-v140-tuf-p11-p20/Train/03640 - 東 - prisMKa _ Sprout.adofai" --difficulty P16 --checkpoint checkpoints/fingering/fingering_network_v01-4fa2ae1.pt --output-dir artifacts/fingering_full_chart_replay_v02/smoke-NEW --status-file artifacts/fingering_full_chart_replay_v02/smoke-NEW.status.json --stop-after-s 2 --wall-timeout-s 60
python scripts/run_full_chart_fingering_replay_v02.py --chart "data/DMDOD-v140-tuf-p11-p20/Train/03640 - 東 - prisMKa _ Sprout.adofai" --difficulty P16 --checkpoint checkpoints/fingering/fingering_network_v01-4fa2ae1.pt --output-dir artifacts/fingering_full_chart_replay_v02/full-NEW --status-file artifacts/fingering_full_chart_replay_v02/full-NEW.status.json --wall-timeout-s 300
```

Each output is create-once and contains `summary.json`, the self-contained
`full-chart-replay.html`, and its complete JSON payload.  The HTML is directly
openable via `file://` and uses no CDN or fetch request.
