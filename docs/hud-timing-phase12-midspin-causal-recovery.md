# DMDOD Phase 12 — Midspin Causal Timing Recovery

## 最終判定

**`MIDSPIN_PARTIAL_RECOVERY`**

固定13件はすべてClean Lifecycleへ復元できたが、44ms以上のMotor準備Leadを
確保できたのは9件である。残り4件をLifecycle成功だけでMotor-readyとは扱わない。
False Positive IDは固定Train16・既存5譜面とも0、既存成功イベントの回帰も0だった。

## 観測契約と因果境界

Studentが使うのは現在・過去のHUD263と可視Countdown cueだけである。新しい規則は
HUDに同時に見えている次の組だけを扱う。

1. 未来slotにMidspinフラグがあり、entry/exit角が同一である。
2. その直後のslotも可視で、通常の2惑星floorである。
3. Midspinと直後floorの双方にTwirl、SetSpeed、MultiPlanet、Pause、Checkpoint、
   SetFloorIconがない。
4. 連続Midspin、欠損lookahead、角度矛盾では従来どおり棄権する。

StockのMidspinは可視の0時間・非打鍵geometry transitionである。したがって、この
条件ではMidspin境界への到達時刻と直後の通常floorの打鍵時刻が同一になる。実装は
この境界を凍結v1.4の幾何計算に1回だけ通し、直後のzero arcで
`eligible_partial`として停止する。chart floor index、target時刻、将来action、
Evaluator associationは使用しない。

この規則はMidspinフラグを必要とするため、zero-angleまたは通常の360度floorとは
区別される。時刻が近いことによるID結合は行わず、Lifecycle v1.5が新しい独立IDを
通常どおり生成・継続・終了する。

## Stage A — 固定13件の結果

| Chart / event | Baseline | Candidate lead | 誤差 | 44ms |
|---|---:|---:|---:|---:|
| ALL MY FELLAS 64 | abstain | 5.000ms | -0.000004ms | FAIL |
| ALL MY FELLAS 69 | abstain | 80.000ms | -0.000005ms | PASS |
| ALL MY FELLAS 74 | abstain | 74.999ms | -0.000005ms | PASS |
| ALL MY FELLAS 78 | abstain | 76.249ms | -0.000005ms | PASS |
| Regular Dan 114 | abstain | 77.647ms | -0.000006ms | PASS |
| Crossing Notes 23 | abstain | 70.000ms | -0.000010ms | PASS |
| First town 1 | abstain | 10.588ms | -0.000003ms | FAIL |
| First town 36 | abstain | 3.824ms | -0.000001ms | FAIL |
| First town 41 | abstain | 70.294ms | -0.000002ms | PASS |
| First town 46 | abstain | 76.765ms | -0.000002ms | PASS |
| First town 146 | abstain | 0.882ms | -0.000001ms | FAIL |
| First town 151 | abstain | 77.353ms | -0.000002ms | PASS |
| First town 156 | abstain | 73.824ms | -0.000002ms | PASS |

13件すべてでmatching IDは1個、Lifecycle不整合は0だった。完全なHUD263、可視floor、
orbit phase、過去frameからのdirection、可視BPM、Countdown、両armのpending/new/
reused/canceled/elapsed、due区間、棄権理由は
`reports/hud-timing-phase12-midspin-causal-traces-20261020-r3.json`に保存した。
target時刻・floor ordinal・44ms deadlineはcapture完了後に追加したEvaluator-only欄である。

## Stage B — 観測可能性

- **Midspinと直後の通常floorの識別**: 上記の可視フラグ・zero arc・次slotにより成立。
- **同時刻性**: Stock Midspinの0時間・非打鍵規則から導出できる。target truthの照合で
  13件すべてのdue誤差が0.000010ms以下だった。
- **直前通常打鍵からの間隔**: 可視orbit phase、方向履歴、Mod-visible Tile BPM、床角から
  凍結v1.4と同じ計算を使う。数値BPMはStock-only観測ではなく既存Mod-visible契約である。
- **10ms / 80ms / 44ms**: 80ms窓では13件を復元したが、4件は44msに不足した。
- **zero-angle / full turn誤認**: Midspinフラグ必須なので対象外。
- **Twirl / Pause / MultiPlanet等**: Midspin pair自体または直後floorにmarkerがあれば棄権。
  安全条件を緩和していない。

44msに不足した4件では、期限以前のHUDに「通常floor → 可視SetSpeed floor → Midspin」
が見えていた。凍結v1.4は未来SetSpeed境界を越えて、その後の速度を推測しない。
SetSpeed floorがcurrentになり更新後の可視BPMを観測して初めてMidspin dueを出すため、
Leadは0.882–10.588msに留まった。これはMidspin同一性の失敗ではなく、Phase 4/5で
分離された未来SetSpeed観測問題である。今回その安全ゲートは変更していない。

反事実的にも、直後slotが欠損・別Midspin・MultiPlanet・marker付きなら、同じ現在の
Midspin表示だけから次の安全なplayable境界を確定しない。実装とunit testはこの場合を
棄権に固定している。

## Stage D — 回帰監査

| 対象 | Baseline | Candidate |
|---|---:|---:|
| 固定Train16 Clean | 1052/1068 | **1065/1068** |
| 固定Train16 False Positive ID | 0 | **0** |
| 既存5譜面 Clean | 199/200 | **200/200** |
| 既存5譜面 False Positive ID | 0 | **0** |
| 既存成功イベントの回帰 | — | **0** |

既存5譜面の追加1件はTriple Cross event 23で、同じ可視Midspin規則によりCleanへ
復元した。誤差は-0.000010ms、Leadは15.000msであり、Lifecycle上は正しいが
Motor-readyではない。この改善を理由にTiming MAEの母集団差を推定器精度向上とは
主張しない。

Candidateに残る未成功3件は、Phase 12対象外のFirst town Twirl直後2件とヒアソビ
Checkpoint 1件である。Phase 11の分類を維持した。

## 実装・再実行

Opt-inは`HUDV3CompositeForecastSource(..., lifecycle_version="v1.5",
midspin_recovery=True)`でのみ有効になる。既定値は`False`で、v1.4 Lifecycleとの
組合せは例外にする。

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp artifacts/pytest-phase12-final-r1 `
  tests/test_hud_timing_tracker_midspin_v16.py `
  tests/test_audit_hud_timing_phase12_midspin_v01.py `
  tests/test_hud_timing_tracker_transition_v14.py `
  tests/test_hud_timing_tracker_lifecycle_v15.py `
  tests/test_hud_timing_composite_v20.py `
  tests/test_hud_v3_composite_forecast_source_v30.py `
  tests/test_hud_timing_phase11_generalization.py

.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase12_midspin_v01.py `
  --manifest reports/hud-timing-phase11-selection-20261019-r1.json `
  --output reports/hud-timing-phase12-midspin-audit-20261020-r3.json `
  --trace-output reports/hud-timing-phase12-midspin-causal-traces-20261020-r3.json
```

Fingering、Motor、Scorer、checkpoint、学習、Validation20、FINALは変更・実行していない。
実Motor A/Bも実施していない。
