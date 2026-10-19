# HUD v3 Phase 10.1 — Alias Identity Monotonicity

## 結論

Phase 10のCountdown/v1.4 aliasには、alias元が消えた後に別のv1.4 local IDを
再結合できる経路が存在した。合成反例で再現し、初回結合分岐をalias未確立時だけに
制限した。

修正後、確立済みaliasは単調である。同一Countdown tokenの生存中に別v1.4 local IDへ
付け替わらない。固定5譜面監査はPhase 10の`199/200 Clean`を維持した。

## 合成反例

次のcurrent/past観測列を固定した。

1. Countdown tokenを生成する。
2. 唯一重なるv1.4 ID `0`を観測し、初回aliasを確立する。
3. eligibleな空forecastによってv1.4 ID `0`を取り下げる。Countdown tokenとalias記憶は維持する。
4. 同じdueに新しいv1.4 ID `1`を観測する。

修正前はstep 5で新ID `1`を再び一意重なりとして採用し、statusは
`eligible_insufficient_lead`になった。これは確立済みidentityの暗黙の付け替えである。

修正後はstep 5を`source_conflict`として棄権し、`countdown:0`をcancelする。
新ID `1`をCountdown IDとして再利用しない。conflict処理後はalias状態も閉じる。

## 保証した状態遷移

- alias未確立時の、競合のない一意な初回結合：許可
- 同じv1.4 local IDの継続：許可
- 別v1.4 local IDへの再結合：禁止
- aliasと別候補が同時に見える場合：`source_conflict`
- aliasが一時的に見えず、その後に別IDが現れる場合：`source_conflict`
- `reset()`後の新しい初回結合：許可

変更は`HUDTimingComposite.observe()`の初回結合条件への1条件追加だけである。
Countdown推定式、v1.4予測式、due計算、不確実性計算は変更していない。

## 固定Train監査

Phase 7固定4譜面・12秒とTrain bool冒頭5秒、合計200イベントを再監査した。

| 指標 | legacy conflict | identity reconciled |
|---|---:|---:|
| Clean Lifecycle | 198 / 200 | 199 / 200 |
| Event ID不整合 | 1 | 0 |
| 正当な棄権 | 1 | 1 |
| False Positive ID | 0 | 0 |

全GateがPASSした。

- Triple Cross初回source conflict解消を維持
- Triple Cross重複ID生成なし
- False Positive増加なし
- reconciliation対象外のdue / uncertainty差分なし
- 既存成功イベント回帰0
- Train bool初回Countdown回帰なし
- Train boolの20.83 ms隣接イベントを別IDとして維持
- Adapter全frame PASS
- 合成Motor受信契約PASS
- Midspin直後の正当な棄権を維持

実Motor、Validation20、FINAL、学習、checkpoint更新は実行していない。Phase 9の物理結果、
Fingering、Motor、Scorerは変更していない。

## 検証

関連するTiming/adapter/Motor受信契約を含むテスト結果は`153 passed`。

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_audit_hud_timing_phase10_source_identity.py `
  tests/test_hud_timing_composite_v20.py `
  tests/test_hud_v3_composite_forecast_source_v30.py `
  tests/test_hud_timing_phase8_lifecycle_integration.py `
  tests/test_countdown_phase_tracker_v01.py `
  tests/test_countdown_first_event_v01.py `
  tests/test_countdown_motor_preflight_v30.py `
  tests/test_persistent_fingering_scheduler_v01.py `
  tests/test_timing_fingering_contract_v02.py `
  tests/test_timing_fingering_motor_integration_v02.py `
  tests/test_run_hud_timing_phase9_lifecycle_motor_ab.py

.\.venv\Scripts\python.exe -u `
  scripts/audit_hud_timing_phase10_source_identity_v01.py `
  --output reports/hud-timing-phase10-1-alias-monotonicity-<new-id>.json
```

採用結果：`reports/hud-timing-phase10-1-alias-monotonicity-20261019-r1.json`
