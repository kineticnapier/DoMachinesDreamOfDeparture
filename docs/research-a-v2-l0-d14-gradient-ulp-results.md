# 研究A v2-L0-D14 — CPU経路差の勾配は1 ULPだけでは説明できない

**判定：`multi_ulp_difference_observed`。事前登録を変更せず、D12の保存済みJSONのみを解析。追加学習0試行、PPO更新0回、Motor実行0、新規物理KeyDown0。正式研究002は未承認。**

## 事前固定と検証
- [事前登録](research-a-v2-l0-d14-gradient-ulp-protocol.md) は実装前コミット [`f3fe91f1`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/f3fe91f11847febdc5eb7d4b39db5c44bf145930) で固定。D13 Draft PR #56のheadをbaseに [D14 Draft PR #57](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/57) を作成した。
- D12の最終成功 [Actions #41424435292](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292)、artifact #13130809065 の `capture.json` / `native.json` / `default.json` / `comparison.json` を読み取り専用で使用。
- [D14 Actions #41440014416](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41440014416) は **SUCCESS**、専用検査15件合格。結果JSON・検査ログ・解析ログは [artifact #13139640418](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41440014416/artifacts/13139640418) に保存。Node関連の非致命的な警告あり。
- D13の入力検査・勾配比較を再利用し、元D12の初期重み/バッチSHA、CPU条件、物理打鍵記録、勾配差件数・最大絶対差・最初の差がすべて整合することを確認。元の身体・報酬・PPO・重みは変更していない。

## float32 ULP距離（生ビットから単調順序へ変換）

| 指標 | 1回目のAdam.step直前 | 2回目のAdam.step直前 |
|---|---:|---:|
| 勾配要素総数 | 1,509 | 1,509 |
| 差のある要素 | **292** | **407** |
| 1 ULP | 151 | 222 |
| 2 ULP | 84 | 116 |
| 3–4 ULP | 29 | 35 |
| 5–16 ULP | 18 | 29 |
| 17 ULP以上 | 10 | 5 |
| **1 ULP超の要素** | **141** | **185** |
| 最大ULP距離 | **128** | **48** |
| 最大絶対差 | 1.862645149230957e-9 | 3.725290298461914e-9 |
| 正負の反転 | 0 | 0 |
| ゼロ↔非ゼロ | 0 | 0 |
| 正負ゼロだけの違い | 0 | 0 |

最初の差は両段階とも `shared.0.weight[0]`。1回目は **2 ULP**、2回目は **3 ULP**。ULPはfloat32で隣接する表現可能値の間隔数であり、相対誤差や実際の打鍵差を表すものではない。

**否定的結果：** CPU経路による勾配差が「すべて隣接1 ULP以内に収まる」という説明は成立しない。一方、今回の全相違要素では正負反転・ゼロ遷移は観測されなかった。これもPPOの学習信号の向きが因果的に不変であることの証明ではない。

## パラメータごとの最大ULP距離
- 1回目：`shared.0.weight` 115要素・最大128、`shared.0.bias` 12・128、`shared.2.weight` 150・128、`shared.2.bias` 4・1、`actor.weight` 11・32。critic.weightの差は0。
- 2回目：`shared.0.weight` 138・48、`shared.0.bias` 14・4、`shared.2.weight` 217・32、`shared.2.bias` 1・2、`actor.weight` 35・32、`critic.weight` 2・2。

## 物理KeyDownと未解決事項
D12の保存済みA0単指・seed0・8エピソード、128目標は **Hit0/128、Extra9/128、実際の物理KeyDown9**。今回新たなMotor実行や評価は行っていない。A1二指・A3未学習対照・指への介入も追加していない。左右交代率・使用率は成功条件にしていない。

複数ULP差が観測されたが、**その最初の演算（softmax、log-prob、逆伝播など）は未特定**。D3/R2・D5/R1の歴史的な再現不一致への因果寄与、低速打鍵獲得の失敗原因、単指物理的不可能性、人間同等性、正式研究002の承認も未証明。結果を見て事前基準を変更していない。

**次の候補：** 同じ凍結バッチで、最初のPPO逆伝播の各演算を読み取り専用の計測付きで比較する。ただし新しい実験条件は別途事前登録する。重複した全256試行の再学習はしない。

元PR #34/#37〜#56、元の身体・報酬・PPO・モデル・過去結果は未変更。**Draft維持。マージ・Draft解除・自動マージは禁止。**
