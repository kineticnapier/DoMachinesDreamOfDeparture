# 研究A v2-L0-D13 — 保存済み勾配の差は最初のAdam更新より前に存在した

**判定：`gradient_difference_before_first_adam_step`。事前登録どおり、既存D12のJSONを読み取り専用で再解析。追加学習0・PPO更新0・Motor0・新規物理KeyDown0。正式研究002は未承認。**

## 実行前固定・証拠
- [事前登録](research-a-v2-l0-d13-frozen-gradient-stage-protocol.md) は実装前コミット [`7014573f`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/7014573fabc7f1f659ebc1926526e690588bbca6) に固定。変更していない。
- [Draft PR #56](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/56) はD12 Draft PR #55のheadをbaseにする独立研究PR。監査PR #34は未変更。
- [D12最終SUCCESS Actions #41424435292](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424435292) のartifact #13130809065から `capture.json`, `native.json`, `default.json`, `comparison.json` を取得。元モデル・PPO・Motorは実行せず。
- [D13 Actions #41431897268](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41431897268)：**SUCCESS**、13件の検査成功。 [検査ログ・解析ログ・結果JSON artifact #13133479891](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41431897268/artifacts/13133479891) を保存。

## 保存済みD12の事前照合
D12の同じ初期重みSHAと同じ凍結バッチSHA、CPU/OS/Python/PyTorch/1スレッド一致、異なるCPU capabilityを検証。D12の元比較の重み差10要素、Adam `exp_avg` 差572、`exp_avg_sq` 差739、返却PPO損失4項目の差0を確認した。物理打鍵の過去測定は**A0 seed0・8エピソード・128目標、Hit0、Extra9、物理KeyDown9**。A1/A3の追加測定なし。

## 2回のAdam更新直前の勾配（float32ビット比較）

| 項目 | Adam step 1直前 | Adam step 2直前 |
|---|---:|---:|
| 比較した勾配要素 | 1,509 | 1,509 |
| **異なる勾配要素** | **292** | **407** |
| 差があるパラメータ配列 | 5 | 6 |
| 最大絶対差 | 1.862645149230957e-9 | 3.725290298461914e-9 |
| `shared.0.weight`差 | 115 | 138 |
| `shared.0.bias`差 | 12 | 14 |
| `shared.2.weight`差 | 150 | 217 |
| `shared.2.bias`差 | 4 | 1 |
| `actor.weight`差 | 11 | 35 |
| `critic.weight`差 | 0 | 2 |

最初の差は両段階とも `shared.0.weight[0]`。
- step 1直前：native `-0.0003530441608745605`、default `-0.00035304410266689956`、差 `5.820766091346741e-11`。
- step 2直前：native `-0.00024932410451583564`、default `-0.00024932401720434427`、差 `8.731149137020111e-11`。

**結論：** D12のCPU経路差は、Adamの最初の更新後に初めて発生したものではない。**最初のAdam.step呼び出し直前に、すでに292要素の勾配差が存在した。** D12で返却された4つの損失値が完全一致していても、勾配は一致していない。

## 否定的結果・未解決
- 既存の実打鍵はHit **0/128**、Extra **9/128**。今回は追加打鍵・方策再評価なし。**低速打鍵獲得・Hit改善は確認していない**。
- 勾配の最初の差が**どの演算**（log-prob、softmax、逆伝播など）から生じたかは未特定。Adam内部の差が最終重みの差へどのように寄与したかも、これだけでは因果分解できない。
- D3のR2再現失敗、D5のR1再現失敗の原因は未確定。歴史的R2勾配を検証したことにはならない。A1/A3・指介入の追加検証はない。
- 人間同等性、単指物理的不可能性、正式研究002の承認は未証明。左右交代率・指使用率は成功条件にしていない。

**次の候補：** 追加学習ではなく、D12の凍結バッチに対する最初の逆伝播で、softmax・log-prob・勾配計算のどこでCPU差が初出するかを、別の事前登録で検証する。

既存研究PR #34/#37〜#55、身体、報酬、PPO、学習済み重み・過去結果は未変更。**Draft維持、マージ・Draft解除・自動マージは禁止。**
