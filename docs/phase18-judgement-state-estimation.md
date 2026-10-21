# Phase 18 — Judgement State Estimation

## 結論

最終判定は
**`SIMULATOR_MODEL_IDENTIFIES_BACKLOG_STOCK_IDENTITY_UNVERIFIED`**である。

Phase 17の可視累積カウンター差分に、HUD Timingのlocal ID/due区間/lifecycle、
自身のMotor予約とKeyDownを組み合わせる独立`JudgementStateEstimator`を実装した。
productionのTiming、Fingering、Motor、Scorerには接続していない。

Phase 16ログでは、repositoryの明示的な判定窓モデル
（Perfect 25 ms、Early/Late Perfect 30 ms、counted 65 ms）を仮定すると、
ID 38の打鍵後3.270 sに得る`very_late:+1`はID 37への対応だけと整合した。
同じ方法でID 89は5.260 sに「未処理である可能性」を検出し、ID 90は
5.320 sに再同期仮説を支持した。

これはinstalled Stockでの一意同定を証明しない。Stockの判定窓・カウンター分類・
非打鍵Miss表示が未検証であり、Estimatorは全結果に
`stock_identity_verified=False`を付ける。

## HUD-only入力契約

オンラインEstimatorが読む情報は次だけである。

- Timing: Composite local ID、観測から推定したdue区間、確認/棄権/取消/elapsed、
  観測時刻と棄権理由
- Motor: 自身の予約に含まれるintended local ID、物理/論理KeyDown時刻、
  その入力をAIが観測した時刻
- Judgement: Phase 17 `JudgementDelta`のlabel別個数、表示更新区間、
  baseline/reset/invalid状態

禁止した入力はScorerの`target_index`、内部cursor、`_used`、`_missed`、真のtarget
時刻である。Phase 16の`target_index`はオンライン再生が完了した後、監査結果と
照合するOracleとしてのみ使用した。

Motor KeyDownとTiming local IDの対応は、KeyDown発生frameに存在した自身の
`reservations_after_physics`から取得する。正解eventとのdue近傍照合ではない。

## 状態仮説

Estimatorは確率を生成しない。各可視deltaについて、次の有限集合を保持する。

- KeyDownがintended local IDへ対応する仮説
- 最大3件前までの未処理local IDへ対応する仮説
- due uncertaintyにより複数カテゴリが成立する仮説
- 同一10 ms区間に複数KeyDown/deltaがあり、順序のpermutationが残る仮説
- count不一致、reset、invalidにより対応付け不能な状態

各仮説は`input ID / intended ID / possibly judged ID / visible label`、支持証拠、
反証を保持する。確率分布は根拠がないため導入していない。

`MODEL_IDENTIFIED`は指定したwindow modelの下で仮説が1件という意味だけであり、
Stock identityの確定ではない。複数なら`AMBIGUOUS`、カウンター不連続なら
`OBSERVATION_GAP`とする。

## Phase 16 Event 37–90

### ID 37の棄権

3.190 sに30.588 ms Leadしかなく、Motor入力が予約されなかった。この時点で可視
判定差分はまだないため、ID 37が後に自動Missで解決する世界と、次入力時にも未処理
である世界を区別できない。

### ID 38

3.261 sのKeyDownに対する最初の可視更新は、理想10 ms captureで3.270 sである。

| 仮説 | 予測カテゴリ | 実測delta | repository model |
| --- | --- | --- | --- |
| KeyDown→ID 38 | `perfect` | `very_late:+1` | 不整合 |
| KeyDown→ID 37 | `very_late` | `very_late:+1` | 整合 |
| KeyDown→ID 36 | `too_late` | `very_late:+1` | 不整合 |

したがってrepository modelでは正常対応を排除し、ID 38が未処理である可能性を出力
した。Phase 17のカウンター単独では不可能だった原因候補の絞り込みが、Timing dueと
自身のKeyDownを加えることで成立した。

### ID 38–89の対応ずれ

対象範囲の51 KeyDown（ID 90を含む）を事後Oracleと比較し、実際の判定先は全51件
でEstimatorのmodel-compatible仮説集合に含まれた。これは同じ固定ログとrepository
model上の再現であり、未知譜面精度やStock parityを示さない。

ID 89の5.253 s KeyDown後、5.260 sの`very_late:+1`はID 88対応とだけ整合し、
local ID 89を未処理候補とした。repositoryの事後expiry 5.315 sより55 ms前である。
ただし「未処理可能性」は将来Missの確定ではない。非打鍵Missカウンター仕様とStock
cursor進行が不明なため、Event 89のMissを確定予測したとは報告しない。

### ID 90の再同期

5.319 s KeyDown後、5.320 sの`perfect:+1`はID 90との正常対応を支持し、未処理候補を
空にした。これはrepository model上の再同期認識であり、visible Missを見てcursorを
直接読んだ結果ではない。

## 識別不能性

次の反例をテストした。

1. Timing/Motor文脈を除けば、正常VeryLateとFIFO VeryLateは同じcounter deltaになる。
2. due区間が重なると、intendedとpredecessorの両方が同じlabelと整合する。
3. 1区間に複数KeyDownと複数deltaがある場合、累積値は順序を失う。
4. reset/invalid/欠損・遅延frameでは対応付けにgapが生じる。
5. Stockの非打鍵Miss表示が不明な限り、counter増分なしからcursor進行を断定できない。

従って「VeryLateだからFIFOずれ」とは判定しない。Timing due・KeyDown時刻・明示window
modelの全てが一意な場合にのみ`MODEL_IDENTIFIED`とする。

## 実装の影響

追加moduleと監査スクリプトはproductionからimportされない。既存263D/278D/368D、
44 ms gate、TimingSystem、Fingering Network、Motor物理、Scorer、checkpointは変更して
いない。推定結果を使う打鍵追加・遅延・取消・再同期操作も実装していない。

## 検証結果

- Phase 18 Estimator / 監査単体: 8 passed
- Phase 16–18と既存Timing / Fingering / Motor契約の限定回帰: 133 passed
- repository全体: 1962 passed、既知のPyTorch tensor変換warning 1件

テスト失敗は0件だった。

## 再実行

```powershell
.\.venv\Scripts\python.exe -m pytest -q `
  tests/test_judgement_state_estimator_v01.py `
  tests/test_audit_phase18_judgement_state_v01.py `
  --basetemp artifacts/pytest-phase18-unit

.\.venv\Scripts\python.exe scripts/audit_phase18_judgement_state_v01.py `
  --source reports/phase16-firsttown12-event89-causal-20261021-r2.json `
  --output reports/phase18-judgement-state-20261021-r1.json
```

監査出力は既存fileを上書きしない。
