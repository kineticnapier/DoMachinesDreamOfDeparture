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

The recorded `full-20261018-r1` transformation reused all 1,222 events, 1,222
logical KeyDowns and 12,953 Motor frames.  All contacts were known and the
saved Hit/Miss/X-accuracy/Timing-MAE object remained byte-equivalent.  Across
the selected chart, the maximum calculated arrival-position delta was
`1.411039e-6` geometry units and the maximum two-planet role-swap continuity
delta was `1.235911e-6`.  Its 1,250 floors include 622 CW, 628 CCW, 40 full
turns and 27 Midspins.  This particular chart has no mid-arc SetSpeed action;
the piecewise-speed path is covered by a dedicated 90°/90° fixture.

Automated visual interaction is not marked PASS.  Browser Use blocks opening
new `file://` URLs, and the computer-use rules prohibit automating the Codex
desktop UI itself.  JavaScript syntax and self-containment are mechanically
verified; final human visual checks should cover play/pause, arbitrary seeking,
speed changes, event jumps, counters, KPS and planet continuity.

## User-confirmed 16-key overlay correction

The old `zip(KEY_LABELS, LOGICAL_KEYS)` paired the first eight displayed
keys with logical left contacts in reverse motor order. The actual user
layout is now explicitly mapped in `KEY_DISPLAY_MAP`:

- top: `Tab/1/2/E` = left little/ring/middle/index (`left_4/3/2/1`);
  `P/^/|/Enter` = right index/middle/ring/little (`right_1/2/3/4`)
- bottom: `LShift/LCtrl` = left coupled side (`left_7/8`);
  `C` = left flat-thumb-only (`left_6`);
  `無変換` = left thumb tip-or-flat (`left_5`);
  `RAlt` = right thumb tip-or-flat (`right_5`);
  `.` = right flat-thumb-only (`right_6`);
  `RCtrl/RShift` = right coupled side (`right_7/8`)

The side keys are coupled physical motions; individual side-contact ordering
is a viewer mapping, not independently actuated physical fingers. Saved Motor
and policy results are unchanged. To correct the existing Human-Proxy HTML
**without rerunning physics or loading a checkpoint**, after pulling:

```powershell
.\.venv\Scripts\python.exe scripts/remap_fingering_replay_key_overlay.py `
  --source-json artifacts/fingering_network_v011/full-20261018-r2-comparison/replay-v011.json `
  --output-html artifacts/fingering_network_v011/full-20261018-r2-comparison/replay-v011-keymap-fixed.html
start .\artifacts\fingering_network_v011\full-20261018-r2-comparison\replay-v011-keymap-fixed.html
```

The new output is write-once; use another filename if it already exists.
