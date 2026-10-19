# HUD v3 Phase 9 — Lifecycle v1.5 Motor A/B Smoke

## 判定

`MOTOR_INTEGRATION_PASS`

Train bool 冒頭5秒・9 targetについて、HUD v3 CompositeのLifecycleだけを
v1.4 / v1.5で切り替えた。Fingering Network checkpoint、12軸Motor、Scorer、
10 ms control、1 ms physics、44/69 ms Leadは固定した。両armの物理結果は
完全に一致し、Lifecycle v1.5による回帰はなかった。

これは限定的な安全性smokeであり、Hit改善や未知譜面への汎化を示さない。

## 固定入力

- Chart: `04042 - bool - vo[ICE].adofai`
- Chart SHA256: `04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1`
- Scope: episode 0–5 s、9 targets
- Checkpoint: `fingering_network_v01-4fa2ae1.pt`
- Checkpoint SHA256: `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`
- Selector / style alpha: `network` / `0.5`
- Control / physics: 10 ms / 1 ms
- Normal / decision Lead: 44 ms / 69 ms

## 物理結果

| Lifecycle | Hit / Target | Miss | TooEarly | Overload | X Accuracy | Timing MAE | physical KeyDown |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1.4 | 8 / 9 | 1 | 0 | false | 88.888889% | 2.625440 ms | 8 |
| v1.5 | 8 / 9 | 1 | 0 | false | 88.888889% | 2.625440 ms | 8 |

Hit/Miss、target別判定、KeyDown時刻、論理contact、物理axisは全て一致した。

| Target | Due (s) | 結果 | KeyDown (s) | Error (ms) | contact / axis |
|---:|---:|---|---:|---:|---|
| 0 | 3.999334 | Hit | 4.001000 | +1.666376 | left_2 / left_2 |
| 1 | 4.020163 | Miss | — | — | — |
| 2 | 4.249292 | Hit | 4.245000 | -4.291972 | left_1 / left_1 |
| 3 | 4.270122 | Hit | 4.273000 | +2.878162 | left_2 / left_2 |
| 4 | 4.499250 | Hit | 4.500000 | +0.749681 | left_1 / left_1 |
| 5 | 4.520080 | Hit | 4.520000 | -0.080185 | left_2 / left_2 |
| 6 | 4.749209 | Hit | 4.742000 | -7.208666 | left_1 / left_1 |
| 7 | 4.770039 | Hit | 4.773000 | +2.961468 | left_2 / left_2 |
| 8 | 4.999167 | Hit | 4.998000 | -1.167013 | left_1 / left_1 |

初回Countdown eventは両armともTiming ID `0`、`left_2`でHitした。未解決の
2打目は両armともstep 400で`insufficient_observable_lead`となり、物理KeyDownを
発行しなかった。これはPhase 3で分類済みの正当な棄権を維持した結果である。

## Lifecycle・予約契約

500 frameを比較した。

| Lifecycle | new | reused | canceled | elapsed | ID違反 | 予約違反 |
|---|---:|---:|---:|---:|---:|---:|
| v1.4 | 10 | 255 | 0 | 8 | 0 | 0 |
| v1.5 | 10 | 255 | 0 | 8 | 0 | 0 |

- pending ID集合の差分frame: 0
- due差分: 0
- uncertainty差分: 0
- physical / logical KeyDown差分: 0
- target結果差分: 0
- source conflict: 両arm 0
- unassigned / reservation collision / in-flight cancellation: 両arm 0
- lifecycle sequenceおよびadapter契約違反: 両arm 0
- 軸の二重予約、重複reservation ID、不明phase: 両arm 0

5秒境界ではtarget範囲外のlocal ID `9`（due 5.019997 s）が右打ち切りされ、
`armed_in_flight`のまま保存されている。またtarget 8は境界時点で
`keydown_observed`である。両armで同一であり、5秒scope外のイベントをHitとして
集計していない。

この譜面区間ではv1.4も対象イベントを全て`elapsed`として処理したため、Phase 7で
見られたcancel→elapsedの差自体は発火しなかった。従って本smokeの結論は
「v1.5を実Motor経路へ接続しても固定ケースを回帰させない」であり、物理性能改善ではない。

## Gate

全GateがPASSした。

- v1.4からHit数が減少しない
- TooEarly / Overloadが増加しない
- 初回Countdown Hitを維持
- forecast due / uncertaintyが不変
- pending ID集合が不変
- Event ID契約違反が0
- 予約状態違反が0
- 2打目の安全な棄権を維持

## 実行記録と再現

採用runはr4のみ。

- PID: `56860`
- Start: `2026-10-19T20:03:37.3275294+09:00`
- End: `2026-10-19T20:03:46.0112614+09:00`
- Exit code: `0`
- Runner内A/B wall time: `0.894428 s`
- Result: `reports/hud-timing-phase9-lifecycle-motor-ab-20261019-r4.json`
- Status: `logs/hud-timing-phase9-lifecycle-motor-ab-20261019-r4.status.json`
- stdout/stderr: `logs/hud-timing-phase9-lifecycle-motor-ab-20261019-r4.*.log`

r1/r2はWindowsの外側process launcherがPython/scriptを開始できず、r3はstandalone
import不備でMotor frame開始前に終了した。いずれも結果に採用していない。CLI起動を
回帰テストへ追加した後のr4だけが両armを完走した。

```powershell
.\scripts\run_hud_timing_phase9_lifecycle_motor_ab_logged.ps1 `
  -Output reports/hud-timing-phase9-lifecycle-motor-ab-<new-id>.json `
  -StdoutLog logs/hud-timing-phase9-lifecycle-motor-ab-<new-id>.stdout.log `
  -StderrLog logs/hud-timing-phase9-lifecycle-motor-ab-<new-id>.stderr.log `
  -StatusJson logs/hud-timing-phase9-lifecycle-motor-ab-<new-id>.status.json `
  -WallTimeoutS 180
```

Validation20、FINAL、追加譜面、学習、checkpoint更新、閾値探索は実施していない。
