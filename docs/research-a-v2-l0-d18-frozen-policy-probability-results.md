# 研究A v2-L0-D18 — 凍結PPO更新前後の行動確率（実測結果）

**判定：`valid_frozen_policy_probability_comparison`。** 事前登録 `8dc4d90cbb0b3d4bebdd0c16016952533ae03be3` の条件・停止基準・評価基準を変更せず、A0単指seed0のD12凍結8エピソード・3,800観測・128目標を再利用。新規学習0、PPO更新0、Adam step0、Motor0、新規物理KeyDown0。D12に保存された**過去の2 Adam step後**の重みを、更新前の重みと同じ観測に適用しただけである。

## 実行前固定・成果物
- [D18実行前事前登録](research-a-v2-l0-d18-frozen-policy-probability-protocol.md)：コミット [`8dc4d90c`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/8dc4d90cbb0b3d4bebdd0c16016952533ae03be3)。実験コード・テスト・Actionsより前に固定。
- [Draft PR #61](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/61)：D17 PR #60のheadをbaseとする独立研究ブランチ。監査PR #34、基準点PR #44は変更なし。
- [成功Actions #41476670214](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476670214)：**SUCCESS、10テスト合格、NumPy任意依存の警告1件**。[結果JSON・検査ログ・実行ログ artifact #13157398200](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476670214/artifacts/13157398200)。
- 入力はD8成功run #41397065687の元初期重みと、D12最終成功run #41424435292の凍結バッチ・更新後native/default重み。SHA、各float32パラメータのビット列、3,800行、D16行動数 (1204/1170/783/643)、物理記録を照合した。
- 実行環境はPython 3.11.17、PyTorch 2.5.1+cpu、AMD EPYC 7763、AVX2、1スレッド。3モデルの**前向き評価は同じ現在のCPU経路**。D12のnative/defaultは過去の更新時に使用された経路であり、今回の比較ではCPU経路を変えていない。

## 3,800観測での行動確率（更新前→D12 native更新後）
| 行動 | 更新前平均確率 | 更新後平均確率 | 平均差 | 増加/減少/同値 |
|---|---:|---:|---:|---:|
| none | 0.308414369 | 0.306332916 | **−0.002081453** | 13 / 3787 / 0 |
| right | 0.307775591 | 0.309479539 | **+0.001703949** | 3709 / 91 / 0 |
| left | 0.216416900 | 0.217598892 | **+0.001181993** | 3800 / 0 / 0 |
| both | 0.167393138 | 0.166588656 | **−0.000804482** | 167 / 3633 / 0 |

- **argmaxが変化した観測：277/3,800**。観測ごとの確率分布L1距離は平均 **0.0058593961**、最大 **0.0063344240**。KL(before||after)は平均 **1.79434e-05**、最大 **2.44833e-05**。
- 凍結バッチで**実際に選ばれていた行動**の確率の平均差は **−5.38536e-06**。選択確率増加1,980行、減少1,820行。同じバッチの選択行動だけを見ても、確率が一律に増えたわけではない。
- D12 default更新後でもargmax変更 **277/3,800**、4行動の増減行数はnativeと一致。更新後nativeとdefaultを直接比較するとargmax変更 **0/3,800**、確率分布L1距離平均 **9.37205e-10**、最大 **1.04308e-07**。D12で更新後重み10要素に差があっても、今回の凍結観測ではargmaxの違いは確認できない。
- 更新前を2回評価した陰性対照は**確率テンソル完全一致**。モデル重みは全条件で不変。確率の有限性・正規化（行和の絶対誤差1e-6以下）を検査した。

## 否定的結果・限界
- **「D16でleftだけ平均優位度が正だから、学習後にleft以外の確率は上がらない」は否定**。rightも平均確率が上昇した。D17でnoneの勾配ノルムが最大でも、更新後noneの平均確率は低下した。平均優位度・勾配ノルムの大小だけでは更新後の行動確率を判断できない。
- 保存済みD12の実打鍵は**128目標、Hit0、Extra9、物理KeyDown9**。D18では更新後方策でMotorを動かしていない。**argmax 277行の変化も、left 3,800行の確率上昇も、Hit・Extra・物理KeyDownの改善を証明しない。**
- A1二指/A3未学習対照/介入は未実行。D3/D5再現失敗の因果関係、教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002の承認は未証明。左右交代率・指使用率は成功条件にしていない。

## 失敗記録と修正（事前登録は不変）
- [Actions #41476343280](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476343280) **FAILURE**（9テスト成功・警告1、D8/D12取得成功、KL計算が `KL unexpectedly negative` で停止）。[失敗artifact #13157377925](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476343280/artifacts/13157377925)に部分JSON・ログ保存。
- [Actions #41476416807](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476416807) **FAILURE**（同じKL計算問題）。[失敗artifact #13157587598](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476416807/artifacts/13157587598)保存。初回実行の状態反映前に記録コミットを追加したため、同じ検査がもう1回起動した。今後は実行状態確認後にのみ追記する。
- float32 softmaxの行和が厳密な1でない丸め誤差により、直接計算したKLがわずかに負になる問題を確認。**KLの計算に限りfloat64で各行を正規化**する数学的な修正を行い、確率そのもの・増減・argmax・事前の許容基準は変更しなかった。
- [Actions #41476630615](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476630615) **FAILURE**（修正ソースの改行が文字列 `\\n` になり、Pythonの構文検査で停止）。[失敗artifact #13158600026](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41476630615/artifacts/13158600026)保存。構文のみ修正したコミット [`00ec5314`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/00ec5314a5f40cb7aa62add5ef1cee2e01495560) のActions #41476670214がSUCCESS。
- 失敗を成功と読み替えず、全runの履歴と成果物を保存。過去の結果・事前登録・身体・報酬・PPO本体は未変更。

## 次の課題
確率の変化を実際の物理KeyDownに結びつけるには、**更新後方策のMotorでの同一条件評価**を別途事前登録し、A0/A1/A3・Hit・Extra・物理KeyDown・介入を比較する必要がある。D18だけで学習成功を主張しない。

**Draft維持。自動マージ・Draft解除・マージは永久禁止。**
