# HUD Timing Phase 14: Twirl Causal Recovery

## 結論

最終判定は **`TWIRL_CAUSAL_RECOVERY_PASS`** である。

可視HUD263のTwirl marker、直前10msの惑星位相差、可視床のentry/exit角、
Mod-visible Tile BPMだけから、固定3件のTwirl後イベントを一意に予測できた。
First townの2件は新たにClean Lifecycleへ復元し、Triple Crossの参考1件は
既存Cleanを維持したままForecast Leadを15.000148msから75.000148msへ延長した。
3件すべてが44ms Motor準備期限を満たした。

これはTiming-only監査であり、実Motor性能の改善を確認した結果ではない。

## 因果的な観測契約

Policyが受け取るのは現在・過去のHUD263と可視Countdown cueだけである。
target時刻、floor ordinal、将来action、compiled chartはEvaluatorだけがcapture完了後に
使用する。

成立条件は次のとおり。

1. 未来床にTwirl markerが実際に表示されている。
2. 同一の現在床を観測した直前10msとの惑星位相差が、可視Tile BPMから計算した
   1 tickの回転量と一意に一致する。
3. 1 tickの回転量が90度未満で、方向エイリアシングがない。
4. 対象Twirl床はTwirlだけを持つ通常2惑星床で、必要な次床も可視である。
5. 後続Twirlは消去せず、次の方向反転を越えないゲートとして残す。
6. SetSpeed、Pause、Checkpoint、MultiPlanetなどの既存v1.4ゲートは変更しない。

現在方向を固定したv1.4へ方向反転後の幾何を渡すため、可視Twirl markerを1件だけ
計算用コピーから除き、そのTwirl床から次のTwirl直前までentry/exit角を交換する。
固定方向 `s` に対し、この交換後の弧
`s * (entry - exit) mod 2pi` は、反転方向 `-s` で元の床を評価した弧と等しい。
表示HUDそのものは変更しない。

## 反事実と識別可能性

Twirl有り/無しの反事実は期限前からTwirl marker次元で異なる。したがって、異なる
markerを持つ2世界を同一観測とは扱っていない。Twirl有りではmarker到達時に方向が
反転し、無しでは反転しないため、現在方向と可視角から後続dueが一意になる。

一方、markerがない方向反転、90度以上/10msの高BPMエイリアシング、位相履歴欠損、
現在床変更直後、複合action Twirl、Twirl床のMidspin/MultiPlanet、lookahead欠損では
この証明が成立せず棄権する。連続Twirlでは最初のTwirlだけを復元し、次のTwirlを
越えない。未来SetSpeedの数値を推測せず、Phase 13の
`DO_NOT_RELAX_SETSPEED_GATE`を維持する。

## 固定3件

| 譜面 / Event | marker初見Lead | baseline | candidate初回Lead | error | 44ms |
|---|---:|---|---:|---:|---|
| First town 174 | 489.706ms | 棄権 | 79.705964ms | -0.000002ms | PASS |
| First town 201 | 488.529ms | 棄権 | 78.529498ms | -0.000002ms | PASS |
| Triple Cross 23 | 2115.000ms | Clean / 15.000148ms | 75.000148ms | -0.000017ms | PASS |

最初の安全な候補はそれぞれ9.810s（観測方向NEGATIVE）、10.870s
（POSITIVE）、5.050s（POSITIVE）で生成された。固定3件のdue uncertaintyは0ms。
First townの2件では7.352942ms間隔の2イベントを10ms制御区間内でも別IDとして
保持した。Triple CrossではTwirlとPhase 12 Midspin opt-inを独立に構成し、併用時も
同一Event IDを維持した。

## 回帰監査

| 集合 | Baseline | Candidate | False Positive ID |
|---|---:|---:|---:|
| 固定Train16 | 1065/1068 Clean | 1067/1068 Clean | 0 |
| 既存5譜面 | 200/200 Clean | 200/200 Clean | 0 |
| Phase 12固定13件 | 13/13 Clean, 9/13 Motor-ready | 13/13, 9/13 | 0 |

既存成功イベントの回帰は0件。Train16で残る1件は対象外のヒアソビEvent 54
（Checkpoint棄権）である。条件付きMAEの僅かな変化はClean集合が変わるため、
予測式の一般的な精度向上とは解釈しない。

開発中の初回全量監査ではALL MY FELLAS Event 48が
`full_turn_changed_floor`へ回帰した。原因は変換済み現在HUDを未変換の前フレームHUDと
比較したことであり、方向判定用の生履歴と、v1.4時間整合性用の因果変換済み履歴を
分離して解消した。この反例はユニットテストに固定した。

## 実装と成果物

- `src/dmdod/planning/hud_timing_tracker_twirl_v17.py`
- `src/dmdod/planning/hud_v3_composite_forecast_source_v30.py`
- `scripts/audit_hud_timing_phase14_twirl_v01.py`
- `tests/test_hud_timing_tracker_twirl_v17.py`
- `tests/test_audit_hud_timing_phase14_twirl_v01.py`
- `reports/hud-timing-phase14-twirl-audit-20261020-r3.json`
- `reports/hud-timing-phase14-twirl-causal-traces-20261020-r3.json`

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_hud_timing_tracker_twirl_v17.py `
  tests/test_audit_hud_timing_phase14_twirl_v01.py `
  tests/test_hud_timing_tracker_midspin_v16.py `
  tests/test_hud_v3_composite_forecast_source_v30.py -q `
  --basetemp artifacts/pytest-phase14-final-r1

.\.venv\Scripts\python.exe scripts/audit_hud_timing_phase14_twirl_v01.py `
  --manifest reports/hud-timing-phase11-selection-20261019-r1.json `
  --output reports/hud-timing-phase14-twirl-audit-20261020-r3.json `
  --trace-output reports/hud-timing-phase14-twirl-causal-traces-20261020-r3.json
```

監査スクリプトは既存出力を上書きしない。再実行時は新しい出力名を指定すること。
