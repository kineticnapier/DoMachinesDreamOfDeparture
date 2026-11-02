# 研究A v2-L0-D12 — 同じPPOバッチでもCPU演算経路で更新後の重みが異なる

**判定：`ppo_update_difference_observed`。実験条件を満たし、GitHub Actions #41424226871 はSUCCESS（専用テスト13件合格、警告1件）。正式研究002は未承認。**

## 実行前固定と記録
- [事前登録](research-a-v2-l0-d12-frozen-ppo-cpu-protocol.md) は実装前のコミット `3d7e946768cf4524e011317f8d3bc195aa8c5e90` で固定。後から判定基準を変更していない。
- [Draft PR #55](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/55)：D11 PR #54をbaseに独立。監査PR #34は変更していない。
- [初回Actions #41424098113](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424098113)：**FAILURE**。13テスト合格、D8元重み取得成功、バッチ採取のハッシュ処理で任意依存NumPyが存在せず停止。失敗ログと部分結果は[artifact #13130694031](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424098113/artifacts/13130694031)に保存。バッチ・PPO比較は未実行。
- [修正コミット `f862e57a`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/f862e57a78abadc5b6aeb953ef0420aabdc439f1)：float32/整数テンソルの生バイトのハッシュを、NumPyを必要としないPyTorchのuint8 viewに変更。モデル・Motor・報酬・PPO・事前登録は変更なし。
- [再実行Actions #41424226871](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424226871)：**SUCCESS**、13 passed / 1 warning。全記録は[artifact #13131775631](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41424226871/artifacts/13131775631)に保存（frozen-batch.pt、capture.json、native.json、default.json、comparison.json、テスト・実行ログ）。

## 8エピソードの実測（PPO更新前）
D8の初期重みをSHA検査して再利用し、A0単指・seed0の元`train_episode`で**最初の8エピソードのみ**を固定。観測・選択行動・old_log_probs・old_values・returnsと物理KeyDown時刻・左右・Hit/Extraを保存した。

| 指標 | 測定 |
|---|---:|
| 採取エピソード | 8 |
| 目標 | 128 |
| **Hit** | **0** |
| **Extra** | **9** |
| **実際の物理KeyDown** | **9** |
| PPO更新（各CPU条件） | 1回 |
| Adam step（各CPU条件） | 2回 |

元の学習の完全再現を主張しない。新規A1/A3評価なし、教師による運指指示なし。

## 同一バッチに対するCPU比較
native/defaultは**同じ初期モデル・同じ凍結PPOバッチ**を読み込み、元の`update_ppo`を1回ずつ呼んだ。両条件のCPUモデル、OS、Python、PyTorch、1スレッドは一致し、ATen CPU capabilityが異なることを比較処理で確認した。

| 比較項目 | native/defaultで異なる要素数 | 最大絶対差 |
|---|---:|---:|
| **PPO更新後の重み** | **10** | **1.4901161193847656×10⁻⁸** |
| Adam exp_avg | **572** | **9.313225746154785×10⁻¹⁰** |
| Adam exp_avg_sq | **739** | **4.547473508864641×10⁻¹³** |
| PPO返却loss / policy_loss / value_loss / entropy | **0（4項目すべて一致）** | **0** |

最初の重み差は`shared.0.weight[0]`。native `-0.0016586969140917063`、default `-0.0016586966812610626`、差 `2.3283064365386963e-10`。重み・Adam内部状態の差はfloat32のビット列で確認した。

**結論：** 凍結した同じ実打鍵バッチに対し、CPU命令経路のみの違いで**最初のPPO更新後のパラメータが10要素異なる**ことを確認した。一方、返却される4つの損失指標は同一であり、更新の微小差を損失表示だけから検出できない。

## 否定的結果・限界
- Hit 0/128、Extra 9/128で、**低速打鍵の学習獲得や改善は一切示されていない**。採取後の方策をMotorで再評価していないため、更新後の物理KeyDownやHit/Extraへの影響は不明。
- 今回は**凍結した1バッチ・2 Adam stepのみ**。過去R1/D3/D5の実CPU情報が欠落しているため、その再現失敗の原因が今回のCPU差だと証明したわけではない。R2の歴史的勾配も未検証。
- PPOの全256エピソードや他のseed、A1、A3は再学習していない。追加の多数回学習で数値差の増幅を検証したとも主張しない。
- 人間同等性、単指の物理的不可能性、正式研究002の承認は未証明。左右交代率・使用率は成功条件にしていない。

**次の検証候補：** 追加の全学習ではなく、凍結バッチの各PPO epochの勾配・重みの最初の差を特定し、同じ入力のどの演算で生じるかを分解する。別実験として事前登録が必要。

既存の研究PR #34/#37〜#54、身体、報酬、PPO、元モデル・過去結果は未変更。**PR #55はDraft維持、自動マージ・Draft解除・マージは禁止。**
