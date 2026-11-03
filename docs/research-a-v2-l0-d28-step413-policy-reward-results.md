# 研究A v2-L0-D28 — step413の行動確率・報酬：保存済み記録の測定結果

**判定：`valid_step413_archived_policy_reward_audit`（SUCCESS）。** D27で余分な物理KeyDownの局所的因果寄与を確認したstep413について、**PPO更新前Bと更新後A→noneでは8次元観測・Motor事前状態が完全一致する一方、4行動の確率と実際に抽選された指令が異なった**。ただしargmaxは両者とも`none`であり、学習改善は確認できない。

## 実行前固定・出典
- 解析・テスト・測定より前に固定した[事前登録コミット `fc67ec82`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/fc67ec8265748aa5185c663ccc62a0c7de921244)と[固定プロトコル](research-a-v2-l0-d28-step413-policy-reward-protocol.md)。結果を見て判定基準・対象・停止条件は変更していない。
- 独立[Draft PR #71](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/pull/71)、baseはD27 Draft PR #70のHEAD `0b40305a`。監査PR #34、基準点PR #44、過去の研究PRは変更していない。
- D22 [成功Actions #41515190995](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995) の [artifact #13175252296](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41515190995/artifacts/13175252296) から `result.json`・`steps.json` を再利用。D23の原本照合と陰性対照全step完全一致が成功。
- A0単指、seed320403、240ms間隔、開始1.075秒、16目標、480制御step。**新規学習0、PPO/Adam更新0、方策再推論0、Motor再生0、新規物理KeyDown0、Hit/Extra新規採点0**。

## step413（4.130秒）の観測と行動確率

| 指標 | 更新前B | 更新後A→none | 更新後−更新前 |
|---|---:|---:|---:|
| none（action0） | 0.3233119547 | 0.3235397637 | +0.0002278090 |
| right（action1） | 0.2952822149 | 0.2938378751 | −0.0014443398 |
| left（action2） | 0.2134876102 | 0.2145890146 | +0.0011014044 |
| both（action3） | 0.1679182351 | 0.1680332422 | +0.0001150072 |
| **左指押下指令の合計確率（left+both）** | **0.3814058453** | **0.3826222569** | **+0.0012164116** |
| **最大確率行動** | **none (0)** | **none (0)** | 変化なし |
| **実際の抽選行動** | **none (0)** | **left (2)** | 異なる |
| step413の報酬 | 0.0 | 0.0 | 0.0 |

- step413直前の**8次元観測、Motor状態、診断状態はすべて完全一致**。観測8成分の最大差0.0。step413より前の物理的に有効な左指指令も全件一致。
- 確率分布のL1距離は **0.0028885603**。更新後に左指押下指令の合計確率が増えたのは**約0.122パーセントポイント**。確率最大の行動は両方とも`none`で、実行指令は異なる。
- **否定的結果**：`argmax`が変化したためにstep413の左指指令が生じた、という説明は支持されない。また、わずかな確率変化が抽選行動を変えたことを厳密に証明するには、同一乱数抽選の内部値と反実仮想の評価が必要。今回の保存記録だけでは「PPOが正しい打鍵を学んだ」とは言えない。

## 4.327秒の余分な物理KeyDownと報酬

| 指標 | 更新前B | 更新後A→none |
|---|---:|---:|
| 4.327秒の物理KeyDown | なし | **あり**（source step432） |
| そのKeyDownの判定 | — | **Extra（Hit=false）** |
| 最近傍目標とのずれ | — | **−108ms** |
| そのKeyDownへの即時報酬 | — | **−0.05** |
| step432の総報酬 | 0.0 | **−0.05** |
| step413〜432の報酬合計 | **−0.10** | **−0.15** |

- 両系列ともstep421の報酬は−0.10。A→none側だけstep432でさらに−0.05が発生。step413の行動指令そのものに即時の加点・減点はない。
- **否定的結果**：この余分なKeyDownが「無罰だったために起きた」という説明は、保存された直接報酬 **−0.05** と一致しない。R1ではこのイベントは無罰ではない。ただし、この減点がPPO更新の行動別学習信号にどれだけ反映されたかは今回測定していない。
- 保存済み採点はB Hit1/Extra1/物理KeyDown2、A→none Hit1/Extra2/物理KeyDown3。新しい採点ではない。

## 否定的結果・限界
- D19の非介入sampled576目標では、初回PPO更新前後でHit **5→4**、Extra **44→46**、物理KeyDown **49→50**。argmaxでは両方とも0 KeyDown。**低速打鍵の獲得・改善は依然として確認できない**。
- 本D28は保存済み1 seed・A0単指の観測比較であり、抽選の因果分解、PPO優位度・勾配への寄与、閉ループの他条件、A1二指・A3未学習対照・他seedへの一般化を証明しない。人間同等性、単指物理的不可能性、正式研究002承認も未証明。
- 左右交代率・使用率は成功条件に使っていない。

## 検査・保存
- 新規コード：`scripts/a002v2_l0_d28_step413_policy_reward.py`、専用検査`tests/test_research_a_v2_l0_d28_step413_policy_reward.py`、専用Actions。
- [初回成功Actions #41562779612](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41562779612)：**SUCCESS、13テスト成功**。D22原本取得、全step原本検査、抽出、結果JSON・検査ログ・実行ログ保存成功。Node.js非推奨警告のみ。
- [結果JSON・tests.log・run.log artifact #13198221365](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41562779612/artifacts/13198221365)（30日保持）。ZIP SHA256 `4630a2cf3257a0bd069a8ba6aefb434e11e96e8cdf0ceec500c769948b29544d`。
- 事前登録の成立判定は**原本照合と抽出の成立**であり、学習・運指の成功ではない。

**Draft維持。自動マージ・Draft解除・既存PRマージは永久禁止。**
