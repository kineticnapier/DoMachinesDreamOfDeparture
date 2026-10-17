# Replay Viewer v0.3 — key overlay and orbital motion

Viewer v0.3 is a render-only transformation of the saved P16 full-chart replay.
It does not load the Network checkpoint, run inference, run the Motor physics,
or alter the saved evaluation metrics.

The 16-key overlay counts `keydowns[].contact` entries at or before the current
seek time.  KPS is the number in `(time - 1s, time]`; Total and per-contact
counts are recomputed from the immutable log on every render.  Highlighted
contacts are reconstructed from the saved 12-axis pressed flags plus the saved
thumb and hand-edge topology.  The labels are aliases only:

| Display row | Logical contacts |
| --- | --- |
| `Tab 1 2 E P ^ | Enter` | `left_1` … `left_8` |
| `LShift LCtrl C 無変換 RAlt . RCtrl RShift` | `right_1` … `right_8` |

This table is configurable in `KEY_DISPLAY_MAP`.  It does not assert that the
logical contacts are wired to those OS keyboard keys.

Planet motion now uses the compiled orbit start angle, signed rotation amount,
Pause hold, exact arrival time and piecewise SetSpeed timing.  The primary
moving planet remains 1.5 geometry units from the fixed planet and reaches the
next floor at the compiled boundary.  Stable planet identities swap fixed and
moving roles without teleporting at ordinary two-planet boundaries.  Midspins
do not swap them.  Three-planet auxiliary rendering is diagnostic: the compiled
primary-orbit correction is used, but the decorative Unity presentation is not
claimed to be exact.

Generate a new create-once output directory:

```powershell
.\.venv\Scripts\python.exe scripts/build_fingering_replay_viewer_v03.py `
  --source-replay artifacts/fingering_full_chart_replay_v02/full-20261018-r1/full-chart-replay.json `
  --chart "data/DMDOD-v140-tuf-p11-p20/Train/03640 - 東 - prisMKa _ Sprout.adofai" `
  --output-dir artifacts/fingering_replay_v03/full-20261018-r1
```
