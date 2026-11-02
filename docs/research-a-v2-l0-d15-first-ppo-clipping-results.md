# 研究A v2-L0-D15 — 最初のPPO勾配制限は発動せず（否定的結果）

**判定：`clipping_inactive_in_both`。** D12の保存済み凍結バッチを使用し、初回PPOの逆伝播をCPUの2条件で再計算。追加学習エピソード0、PPO更新0、Adam更新0、Motor実行0、新規物理KeyDown0。正式研究002は未承認。

## 実行前固定と出典
- [事前登録](research-a-v2-l0-d15-first-ppo-clipping-protocol.md) はコード・実行前のコミット [`0e0e1cd6`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/0e0e1cd61a772a152563402727625b88261768b1) に固定。判定基準を結果に合わせて変更していない。
- [D15 Draft PR #58](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/58) はD14 Draft PR #57のheadをbaseにした独立ブランチ。リファクタリング監査PR #34とは別。
- D12最終成功 [Actions #41424435292](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292) の [artifact #13130809065](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292/artifacts/13130809065) から、凍結バッチと4つの検査済みJSONを再利用。D8成功Actions #41397065687の初期重みを再利用。
- [D15 Actions #41448552410](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41448552410)：**SUCCESS**。専用テスト8件と補助3サブテスト合格、非致命的警告1件。[結果JSON・実行ログ artifact #13143730902](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41448552410/artifacts/13143730902)。

## 最初のPPO逆伝播・制限前後の測定

| 測定値 | native | default |
|---|---:|---:|
| `clip_grad_norm_` のしきい値 | 1.0 | 1.0 |
| **制限前の総勾配ノルム** | **0.29729172587394714** | **0.29729172587394714** |
| 有効な制限係数 | **1.0** | **1.0** |
| 制限前後で変化した勾配要素 | **0 / 1,509** | **0 / 1,509** |
| 制限前後の最大絶対差 | **0** | **0** |
| D12保存済み初回Adam.step直前の全勾配とビット単位一致 | **YES** | **YES** |

CPU間の勾配差は**制限前から292/1,509要素**（最大絶対差 `1.862645149230957e-9`）。制限後も**292/1,509要素**で、差のあるパラメータ配列は5、最初の差は `shared.0.weight[0]`。D12/D13の保存済み比較と厳密一致した。

**否定的結果：** 元PPOの `clip_grad_norm_(..., 1.0)` はこの最初のA0 seed0バッチでは勾配を縮小していない。したがって、D12で確認された**最初のAdam更新前のCPU勾配差は勾配ノルム制限によって生じたものではない**。ただし、後続の学習更新やA1/A3でも制限が発動しないとは言えない。

## 物理KeyDown・研究上の境界
D12で保存されたA0単指seed0の8エピソード、128目標は **Hit0/128、Extra9/128、実際の物理KeyDown9**。今回のMotor実行は0。A1二指、A3未学習対照、指への介入は実施していない。Hit改善も運指獲得も観測していない。左右交代率や使用率は成功条件にしていない。

CPU間の勾配差が最初に生じる演算は依然として未特定。次の候補は**同じ凍結バッチで最初の逆伝播の演算段階を計測し、差の初出位置を特定すること**。再学習を行う必要はない。D3/R2・D5/R1の歴史的再現不一致の原因、低速打鍵の学習失敗原因、人間同等性、単指の物理的不可能性、正式研究002の承認は未証明。

## 保全
既存研究PR #37〜#57、監査PR #34、身体・報酬・PPO本体・過去モデル・過去の結果を変更していない。元データは読み取り専用。**Draft維持、マージ・Draft解除・自動マージは永久禁止。**
