# Phase 23 — Predictive Timing Observability

## 判定

- `EARLY_DUE_NONIDENTIFIABLE`
- `LEAD_44MS_INFEASIBLE`
- `ORACLE_ONLY_INTERVENTION`
- `INSUFFICIENT_EVIDENCE`（実機Stockで安全な制御介入を主張するには不足）

production Timing、Fingering、Motor、Scorer、HUD、checkpointは変更していない。

## Event 37因果タイムライン

| 時刻 | 事象 | 根拠 |
|---:|---|---|
| 2.310 s | Floor 37の将来SetSpeed marker初見。倍率数値は非表示 | HUD263 capture |
| 2.480 s | Midspin pairを可視幾何から安全に識別 | HUD263 capture |
| 3.176588272 s | Event 37の44ms予約期限 | Oracle dueからの事後計算。Student入力ではない |
| 3.180 s | SetSpeed後の数値BPMが初めて反事実間で分岐 | HUD263 capture比較 |
| 3.190 s | Timing local ID 37を初生成、dueをpoint予測 | HUD Timing lifecycle |
| 3.190 s | `insufficient_observable_lead`で棄権 | policy履歴 |
| 3.220588272 s | Event 37 true due | Evaluator-only Oracle |
| 3.250 s | Event 36がNoFail参照モデルで自動遷移 | Phase 20C 10ms仮想frame |
| 3.261 s | ID 38の物理KeyDown | Motor履歴 |
| 3.270 s | ID 38がEvent 37へVeryLateとして消費 | Stock参照モデル、事後照合 |

現行TimingのEvent 37予測誤差は実質ゼロだが、観測Leadは30.588273msであり、
44ms条件を13.411727ms満たさない。正確さと予約の早さは別である。

3.180秒には数値BPMと必要な可視構造が揃う可能性があるが、そのframeからEvent 37の
exact dueを直接復元する独立式は今回確立していない。これは期限後なので、期限前の
非識別性の結論には影響しない。現在の実装が実際にpoint dueを生成するのは3.190秒である。

## 期限前の反事実証明

Floor 37の既存`bpmMultiplier=2.0`だけを、Phase 13と同じ事前固定係数0.125で
`0.25`へ変更した。marker、floor幾何、Countdown、その他actionは維持した。

| 世界 | Event 37 due | Stock Normal counted window |
|---|---:|---:|
| original, 1360 BPM | 3.220588272 s | 3.155588272–3.285588272 s |
| counterfactual, 170 BPM | 3.529411805 s | 3.411764747–3.647058864 s |

- true due差: 308.823533ms
- originalの44ms期限以前に比較したframe: 318（0.000–3.170秒）
- HUD263最大差: 0.0
- Countdown: 完全一致
- 初回差分: 3.180秒

二世界のcounted windowは非交差である。同一の期限前prefixへ同じ因果policyを適用した場合、
両世界で安全な単一KeyDown時刻は存在しない。original dueに合わせた打鍵はcounterfactual
ではTooEarlyとなる。このため、真のdueを44ms前から参照できるというOracle条件を、
HUDからの44ms予測可能性として扱えない。

## Stock参照介入

保存済み物理KeyDown列に研究用KeyDownを加え、production ScorerではなくPhase 20Cの
独立Stock参照モデルで評価した。

| 条件 | NoFail結果 | Event 89 |
|---|---|---|
| baseline | 50 KeyDownがintended IDと不一致 | 自動Miss |
| Event 37 true dueのみ追加 | 追加入力は未処理Event 36へVeryLateとして消費 | 自動Miss |
| Event 36＋37 true due追加 | 両方Perfect。ただし後続42・47で再度ずれる | 自動Miss |
| 3.251秒のHUD研究候補 | NoFailではEvent 36自動遷移後にEvent 37 VeryLate | 自動Miss |
| 36・37・42・47をOracle補完 | intended不一致0、ID 89がEvent 89へPerfect | Hit |

Event 37だけの回復はEvent 89防止の十分条件ではない。3.251秒候補は3.190秒から61msの
予約Leadを持つが、NoFail・10ms frame phase・Event 36自動遷移を前提とする。通常Failでは
同じ入力がEvent 36へのTooLateとなり、3.320秒に停止する。実Unity frame phase、FailMiss
表示、物理Motorを検証していないため、HUD-only safe interventionとは判定しない。

Oracleで36・37・42・47を全補完すればEvent 89は回復するが、これは因果的Student入力では
実行できない上限比較である。

## 保存済みログでの一般化監査

新規大規模評価はせず、First town 12秒とTrain bool 5秒の保存済みPhase 16結果だけを使用した。

- Lead不足: 14件
- 観測Lead範囲: 0.882430–42.353030ms
- 44ms達成: 0/14
- evaluator事後近傍にSetSpeed: 10件
- evaluator事後近傍にMidspin（SetSpeedなし）: 4件

これは二譜面だけのサンプルであり、一般的頻度を表さない。Phase 4/13および今回の反事実で
一部SetSpeed例の非識別性は証明済みだが、14件全てを反事実検証したわけではない。
個別のMidspin例やSetSpeed後続例には`INSUFFICIENT_EVIDENCE`を維持する。

## 安全介入が成立するための条件

Event 89をHUD-onlyで防ぐには、少なくとも次が必要である。

1. Event 36・37およびEvent 89以前の42・47について、期限前に有限なsound due区間を得る。
2. 互いに識別不能な速度仮説すべてで共通のKeyDown windowを持つ。
3. 44ms以上前に予約し、既存in-flight予約と衝突しない。
4. NoFail/通常Fail、Unity Update phase、自動Missの可視性を実機で確認する。
5. 追加KeyDownが古い未処理floorに消費されないことを集合値状態推定で保証する。

現在は1と2が反事実で否定され、4も未検証である。棄権条件は緩和しない。

## 成果物と再実行

- `src/dmdod/planning/predictive_timing_observability_v01.py`
- `scripts/audit_phase23_predictive_timing_observability_v01.py`
- `reports/phase23-predictive-timing-observability-20261022-r3.json`
- `tests/test_predictive_timing_observability_v01.py`
- `tests/test_audit_phase23_predictive_timing_observability_v01.py`

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_predictive_timing_observability_v01.py `
  tests/test_audit_phase23_predictive_timing_observability_v01.py

.\.venv\Scripts\python.exe scripts/audit_phase23_predictive_timing_observability_v01.py `
  --output reports/phase23-predictive-timing-observability-20261022-r4.json
```

検証結果：

- Phase 13・16・20C・21・22を含む関連回帰: 89 passed
- 全repository: 2032 passed, 1既存PyTorch warning
- 全体テストの隔離先: `artifacts/pytest-phase23-full-r1/`
