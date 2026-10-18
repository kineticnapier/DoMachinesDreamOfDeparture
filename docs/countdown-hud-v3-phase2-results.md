# HUD v3 Phase 2 — Timing Integration Results

## Outcome

Countdown HUD v3から最初の打鍵時刻を推定する、学習なしの因果的Timing経路を
実装した。修正後の固定監査`reports/countdown-timing-phase2-20261018-r2.json`はPASSした。
Validation20、FINAL、checkpoint、Fingering、Motorは使用・変更していない。

## Implemented modules

- `countdown_phase_tracker_v01.py`: 実際に観測した表示遷移を区間化し、2つ以上の
  numeric onsetからtick周期区間を推定する。
- `countdown_first_event_v01.py`: 可視HUD263の現在床幾何、惑星移動方向、可視BPMと
  Countdown位相区間から初回due区間を計算する。
- `hud_timing_composite_v20.py`: HUD278を263+15へ分離し、263だけを凍結v1.4へ渡す。
  Countdown/v1.4の候補を独立状態で管理し、overlap時だけ単一local IDへ統合する。

Countdown表示中にv1.4が惑星周回を通常floor eventとして出すraw forecastは、診断用に
そのまま保存するが、`Ready!`または数字が可視の間はplayable候補として公開しない。
`Go!`後、Countdown due区間と一意に重なる単一のv1.4候補だけを統合する。余剰候補、
欠損/stale/invalid cue中の候補、区間矛盾はplayable候補へ漏らさず棄権またはcancelする。
通常playのhidden経路ではv1.4のraw `TimingUpdate`とstatusが完全一致する。

263DのTile/Real BPMは`mod-visible-tile-real-bpm-v2`契約で画面に数値表示された情報で
あり、stock-only観測ではない。Compositeは契約を明示的に要求し、stock-only入力を
fail closedする。凍結Timing v1.4そのものは変更していない。

## Predeclared gates

| Gate | Result |
|---|---:|
| N=3/N=5、120/240/480 BPM | PASS |
| pitch 200% | PASS |
| due midpoint error <= 10 ms | PASS |
| false Countdown candidate | 0 / PASS |
| unique first-event local ID | 1 per case / PASS |
| due下限による44 ms Lead | 全case PASS |
| 欠損・stale frameで棄権 | PASS |
| 矛盾digit列で棄権 | PASS |
| local ID reuse/cancel/conflict | PASS |
| unconfirmed pending ID reuse / lifecycle partition | PASS |
| invalid cue中のv1.4 playable漏洩 | 0 / PASS |
| 余剰v1.4候補をtruthなしで棄権 | PASS |
| Countdown due区間矛盾時のcancel | PASS |
| stock-only numeric BPM contract拒否 | PASS |
| 通常play v1.4 compatibility | PASS |
| policy truth-field schema/AST | PASS |
| Train bool first event | PASS |

10 ms精度Gateは、評価真値を見て候補を選ぶためには使っていない。推定due区間の半幅が
10 msを超える間は棄権し、追加の数字またはGo観測で区間が狭まった場合のみ候補を出す。

## Synthetic results

| Case | First observed | Midpoint error | Conservative Lead lower |
|---|---:|---:|---:|
| N3 / 120 BPM | 1.010 s | +5.000 ms | 485.000 ms |
| N3 / 240 BPM | 0.510 s | +5.000 ms | 235.000 ms |
| N3 / 480 BPM | 0.260 s | +5.000 ms | 110.000 ms |
| N5 / 120 BPM | 2.010 s | +5.000 ms | 485.000 ms |
| N5 / 240 BPM | 1.010 s | +5.000 ms | 235.000 ms |
| N5 / 480 BPM | 0.380 s | approximately 0 ms | 235.000 ms |
| N4 / 120 BPM / pitch 200% | 0.760 s | +5.000 ms | 235.000 ms |

全caseでCountdown candidate IDは1個、false candidateは0だった。

## Train bool Timing-only diagnostic

- Chart: `04042 - bool - vo[ICE].adofai`
- SHA256: `04981e92f74288b8aac7286e83420b933036d2caead9c157bc33b706779ad1b1`
- Scope: Train chartの冒頭5秒、501 HUD frames
- First AutoPlay due（評価器のみ）: 3.999333624 s
- First causal prediction time: 2.000 s
- Predicted interval: [3.979833385, 3.999833385] s
- Midpoint error: -9.500239 ms
- Conservative Lead lower: 1979.833385 ms
- Countdown local IDs: 1
- False candidate IDs: 0

Lead下限は`predicted due lower - observed_at`だけから計算した。chart truthはHUD
streamを最後まで収集した後にだけ照合した。collection loopは
`targets`、`target_time_s`、`episode_time_s`を参照しないことをASTで検査している。

## Known limitations

- synthetic rendererと保存Train chartのrenderer-side設定を使った診断であり、実ゲームの
  capture latency、OCR誤認識、frame dropは未評価。
- checkpoint、fast takeoff、practice/custom countdownは未対応。
- Pause、Midspin、MultiPlanet、未知marker、方向aliasでは意図的に棄権する。
- 10 msより高精度な保証は行わない。
- Timing-only評価であり、Fingering/MotorのHit/Miss改善は未測定。

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_countdown_hud_v3.py tests/test_countdown_phase_tracker_v01.py tests/test_countdown_first_event_v01.py tests/test_hud_timing_composite_v20.py tests/test_audit_countdown_timing_phase2_v01.py tests/test_hud_timing_tracker_transition_v14.py tests/test_contact_timing_hud_v01.py --basetemp artifacts/pytest-countdown-phase2-r2-final

.\.venv\Scripts\python.exe scripts/audit_countdown_timing_phase2_v01.py --output reports/countdown-timing-phase2-20261018-r2.json
```

上記テスト結果は95 passed。監査scriptは既存出力を上書きしない。
