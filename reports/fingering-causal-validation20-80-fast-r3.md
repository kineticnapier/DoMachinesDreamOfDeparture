# Fingering Causal Lead v0.2 — Validation20 80条件高速化

実行日: 2026-10-18 (Asia/Tokyo)  
コード: `3121135`  
対象: Validation20 × Free/Human-Proxy × Baseline/Causal Lead = 80 arm

## 結論

- Windows上の実測active wall timeは **452.240秒**。600秒目標を147.760秒下回った。
- 20譜面、80 armをすべて完了した。途中打切りや物理step省略はない。
- control 10 ms / physics 1 ms、Network checkpoint、AutoPlay Timing、Motor物理定数は不変。
- 4 CPU workerで独立armを並列化した。arm wall time合計は1,678.403秒。
- Final / Qualification / Reserveは使用せず、`Validation`フォルダの20譜面だけを事前固定した。

## Profileと高速化

最短Validation譜面のFree/Baselineは、最適化前39.451秒、最適化後8.379秒
（同一Hit/Miss/XAcc/MAE/KeyDown数）で、4.71倍になった。

30秒区間のcProfileでは、12軸fast path追加後も次が支配的だった。

- `NKeyMotorEnv._step_held_exact`: 3.595秒 cumulative
- `RealChartMotorEnv._expire_misses`: 2.697秒 cumulative
- `execute`全体: 7.497秒

実施した変更:

1. 12軸Motorの10 ms held commandを、同じ1 ms方程式・演算順序・接点判定で積分し、
   1 msごとの一時`FingerState`/tuple/list生成だけを除去。
2. 解決済みイベントとmiss expiryを先頭から毎frame再走査せず、単調cursorで追跡。
3. Network encoderが読む最大3 future eventより後ろのlookahead構築を省略。
4. 独立armを4 processで並列化。結果のpublishは親processだけが原子的に実施。

旧3譜面×4条件のarm wall合計は1,105.594秒から274.838499秒へ短縮し、
同一範囲で **4.023倍**。さらに4 worker並列化してValidation20全体を452.240秒で完了した。

## 数値結果

| Policy | Arm | Hit / Target | Miss | TooEarly | Overload | macro XAcc | macro MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| Free | Baseline | 28,274 / 28,274 | 0 | 0 | 0 | 98.9844% | 10.9597 ms |
| Free | Causal Lead | 28,271 / 28,274 | 3 | 1 | 0 | 99.8552% | 5.7283 ms |
| Human-Proxy | Baseline | 28,260 / 28,274 | 14 | 0 | 0 | 98.3838% | 11.5784 ms |
| Human-Proxy | Causal Lead | 28,274 / 28,274 | 0 | 0 | 0 | 99.9978% | 3.2348 ms |

高速化は成功したが、方式評価は無条件PASSではない。Free/Causal Leadは
`14483 - QuomodocunquizE`で3 Miss・1 TooEarlyとなり、Free/Baselineの全Hitから悪化した。
一方Human-Proxy/Causal Leadは、Baselineの計14 Missを解消し20/20譜面で全Hitだった。

## 同値性・完全性検証

- exact 12軸fast pathと旧公開1 kHz pathを2/4/12軸、100 control stepで比較し、
  全Motor state、diagnostics、KeyDown/KeyUp時刻・接点が一致。
- 同一譜面4 armの逐次版と4 worker版で、Stats、全Network decision、
  全4,104 KeyDown時刻・接点が一致。
- 最適化前の3譜面×4条件（12 arm）と、Hit/Target/Miss/XAcc/MAE/TooEarly/
  Overload/KeyDown/decision/unassigned/終了理由が全項目一致。
- 本番80 armは全て個別JSON化。113,080 KeyDownについて、保存した2表の件数と
  `physical_keydowns`が全armで一致し、不整合0。
- run manifestは20 chart SHA256、checkpoint SHA256、コードhash、control/physics設定を固定。
- create-only arm checkpoint、exclusive lock、atomic fsync+replace、Ctrl+C/resume testsを実施。
- 最終resumeは80/80をskipし、物理シミュレーションを再実行しなかった。

関連テスト: **69 passed**。

## 保護確認

- Fingering checkpoint: `cca1bfcc2d8e2988eeabf602a57211c8d8b66a54bfbbcbe0e3a9409ec167d2d8`
- Stage A report: `c4f71aadc5dc6cca49679a9902bcb7583a0f89b40f92b7fb283a50c3faad0558`
- MaleCNS4096: `a9d00a26c24ad19e815a0d768851f8040e00748e3a6474a31ce786576f53c710`
- 未コミット`configs/training/prestige1.toml`はstage/commitせず保護した。

## 成果物

- `artifacts/fingering_network_v02/validation20-80-fast-r3/multichart-summary.json`
- `artifacts/fingering_network_v02/validation20-80-fast-r3/multichart-summary.html`
- `artifacts/fingering_network_v02/validation20-80-fast-r3/progress.json`
- `artifacts/fingering_network_v02/validation20-80-fast-r3/runtime-sessions.json`
- `artifacts/fingering_network_v02/validation20-80-fast-r3/run-manifest.json`
- `artifacts/fingering_network_v02/validation20-80-fast-r3/arms/*.json` (80 files)
- `logs/fingering-causal-validation20-80-fast-r3.log`

## 再実行

```powershell
.venv\Scripts\python.exe scripts/audit_fingering_causal_multichart_goal_v02.py `
  --chart-root data/DMDOD-v140-tuf-p11-p20 `
  --max-charts 20 `
  --output-dir artifacts/fingering_network_v02/validation20-80-fast-r4 `
  --goal-seconds 600 `
  --wall-timeout-s 600 `
  --workers 4
```

中断後は同じ出力先に対して次を実行する。

```powershell
.venv\Scripts\python.exe scripts/audit_fingering_causal_multichart_goal_v02.py `
  --output-dir artifacts/fingering_network_v02/validation20-80-fast-r4 `
  --resume --goal-seconds 600 --wall-timeout-s 600 --workers 4
```
