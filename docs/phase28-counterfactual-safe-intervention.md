# Phase 28 — Counterfactual Safe Intervention Search

## 判定

- `ORACLE_ONLY_INTERVENTION`
- `NO_SAFE_INTERVENTION_ESTABLISHED`

探索対象はPhase 23で事前宣言・保存された入力列だけに固定した。無条件の追加KeyDown探索や時刻調整は行っていない。

baselineは50入力の帰属ずれを持ち、Event 89は回復しない。Event 37 dueだけのOracle入力はEvent 36へ消費される。Event 36+37を補っても42・47でずれが再発し、Event 89は回復しない。NoFail auto-transition後3.251sのlate inputもEvent 37だけを埋め、Event 89は回復しない。

36・37・42・47を真のdueで補う上限比較だけがmismatch 0となりEvent 89を回復した。しかし4イベントの真のdue/floorはEvaluator-onlyで、既知のLead不足を含む。実軸Motor envelope、予約競合、baseline Hit保存、live Stock parityも成立していないため、Oracle-onlyである。

通常FailではEvent 36周辺で停止し、NoFailの継続介入を適用できない。run modeを跨いだ成功主張はしない。

状態依存LeadはEvent 37軸/snapshot不在、予約取消・順序変更・軸変更はcounterfactual物理trace不在のためINCONCLUSIVEとした。安全性未証明の候補をsafeへ昇格していない。
