# DMDOD Phase 13 — Future SetSpeed Causal Observability Audit

## 最終判定

**`FUTURE_SETSPEED_PREDEADLINE_NONIDENTIFIABLE`**

Phase 12で44msのMotor準備期限に間に合わなかった4件は、現在の
Mod-visible HUD263 + 可視Countdown契約では期限前に未来SetSpeed数値を識別できない。
4件とも、SetSpeedの存在とMidspin pairは十分早く見えている一方、変更後の数値BPMと
最初の安全なdue予測は期限後に到着した。

予測器、Timing v1.4、Lifecycle v1.5、Midspin拡張、Fingering、Motor、Scorerは
変更していない。SetSpeedゲートを緩和しない。

## 固定対象と情報到達時刻

| Chart / event | 44ms deadline | SetSpeed marker初見 | Midspin pair安全識別 | 数値BPM初識別 | 最初のsafe due | deadline超過 |
|---|---:|---:|---:|---:|---:|---:|
| ALL MY FELLAS 64 | 9.530999596s | 8.440s | 8.630s | 9.570s | 9.570s | 39.000ms |
| First town 1 | 0.676588255s | 0.000s | 0.000s | 0.710s | 0.710s | 33.412ms |
| First town 36 | 3.139823567s | 2.310s | 2.480s | 3.180s | 3.180s | 40.176ms |
| First town 146 | 8.786882432s | 7.960s | 8.120s | 8.830s | 8.830s | 43.118ms |

最初のsafe dueは4件ともuncertainty 0のpoint estimateだが、到着が期限後である。
期限前にpoint estimateや有限なsound intervalが存在したことを意味しない。

## 反事実監査

Phase 4と同じ事前固定介入`0.125x`を、各対象直前の既存SetSpeed actionの数値だけに
適用した。action数、floor、marker、geometry、Countdownは保存した。変更値やtarget
時刻はEvaluator側だけが保持し、captureはrendererが出した現在HUD263とCountdown
のみを記録した。

| Chart / event | 期限までのHUD263最大差 | Countdown | 正解due差 | 判定 |
|---|---:|---:|---:|---|
| ALL MY FELLAS 64 | 0.0 | 一致 | 87.500008ms | `PREDEADLINE_NONIDENTIFIABLE` |
| First town 1 | 0.0 | 一致 | 102.941195ms | `PREDEADLINE_NONIDENTIFIABLE` |
| First town 36 | 0.0 | 一致 | 51.470597ms | `PREDEADLINE_NONIDENTIFIABLE` |
| First town 146 | 0.0 | 一致 | 51.470597ms | `PREDEADLINE_NONIDENTIFIABLE` |

全件で10ms観測frameの期限以前prefixは全263次元が完全一致し、Countdown metadataも
一致した。それにもかかわらず正解dueは51–103ms分岐する。したがって、同じ因果policy
は期限以前に二世界へ異なるpoint dueを返せない。

二世界の正解を包含するEvaluator-only包絡区間は記録したが、Studentが導ける区間とは
扱わない。可視markerは未来の数値速度を符号化せず、HUD263契約上、未知の正の
SetSpeed値に有限なduration上限はない。このため期限前に導出可能な有限sound due区間は
`null`とした。未知速度の推測や訓練集合由来の範囲制約は使用していない。

## 各観測源

- **未来SetSpeed marker**: 全件で期限前に可視。ただし存在だけで数値を識別できない。
- **Mod-visible Tile/Real BPM**: SetSpeed floorがcurrentになった最初の10ms frameで差が出る。
  4件とも期限後だった。
- **Midspin + 次floor**: 全件で期限前にPhase 12の安全条件を満たした。今回の不足情報ではない。
- **Countdown**: 全反事実で期限まで一致。
- **Stock速度アイコン**: Phase 5/6の実pixel onset・capture latencyが未検証であり、
  HUD263契約にも含まれない。内部icon enumをStudent観測へ流用していない。
- **JRP数値表示**: Phase 5でcurrent-floor更新はHit後と確認済み。未来速度previewではない。
- **Audio**: 数値SetSpeedを符号化する実証がなく、今回の観測契約外。

## Triple Cross event 23（参考）

このイベント直前の特殊floorはSetSpeedではなく**Twirl**である。

- 44ms deadline: 5.081000148s
- Twirl marker初見: 3.010s
- Midspin pair安全識別: 3.310s
- 現行Candidateのsafe due: 5.110s（期限後約29ms）

Twirl markerは早期に見えるが、「可視Twirlを越えた方向・時刻推定」の反事実fixtureは
今回作成していない。SetSpeed非識別性へ混同せず、
`OBSERVABILITY_UNVERIFIED`とする。これは将来の限定的Twirl研究候補であり、
今回のSetSpeedゲート変更理由にはしない。

## Phase 12安全性

保存済みPhase 12 r3を再確認した。

- Midspin Lifecycle復元: **13/13維持**
- 44ms Motor-ready: **9/13維持**
- False Positive ID: **0維持**
- 既存成功イベント回帰: **0維持**
- Predictor変更: **なし**

本監査はTiming-onlyの観測可能性研究であり、Motor A/B、学習、checkpoint更新、
Validation20、FINALを実行していない。

## 成果物と再実行

- `reports/hud-timing-phase13-setspeed-counterfactual-20261020-r2.json`
- `reports/hud-timing-phase13-setspeed-causal-traces-20261020-r2.json`
- `scripts/audit_hud_timing_phase13_setspeed_v01.py`
- `tests/test_audit_hud_timing_phase13_setspeed_v01.py`

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  --basetemp artifacts/pytest-phase13-final-r1 `
  tests/test_audit_hud_timing_phase13_setspeed_v01.py `
  tests/test_audit_bool_second_event_counterfactual_v01.py `
  tests/test_audit_setspeed_observability_v01.py `
  tests/test_audit_hud_timing_phase12_midspin_v01.py `
  tests/test_hud_timing_tracker_midspin_v16.py

.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase13_setspeed_v01.py `
  --output reports/hud-timing-phase13-setspeed-counterfactual-20261020-r2.json `
  --trace-output reports/hud-timing-phase13-setspeed-causal-traces-20261020-r2.json
```

## 実装判断

主評価4件に`RECOVERY_IMPLEMENTATION_GAP`はなかった。従って実装案をTimingSystemへ
接続せず、**`DO_NOT_RELAX_SETSPEED_GATE`**とする。

将来変更を検討できるのは、数値速度またはMotorに有用な有限due区間が44ms期限前に
実際のpixel/audio/Mod表示として観測された場合だけである。カテゴリiconや作者傾向から
倍率を推測してはならない。
