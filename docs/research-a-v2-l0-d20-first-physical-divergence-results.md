# 研究A v2-L0-D20 — D19保存済み物理KeyDownの最初の分岐（実測結果）

**判定：`valid_archived_trace_comparison`。** 事前登録コミット [`499447b9`](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/commit/499447b9a513bdafed3b2745b966abb3bbd90af2) の比較・停止・判定基準は変更していない。**新規学習0、PPO更新0、Adam更新0、Motor評価0、新規物理KeyDown0**。D19の過去の保存記録のみを解析した。

## 入力・検査・保存
- 唯一の入力はD19の[最終成功Actions #41485667328](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41485667328)、[artifact #13161377329](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41485667328/artifacts/13161377329)の `result.json`。D19のHEAD `08d8de523ffae10513645dfe95566060b4e6bf0d`。元の学習結果・物理traceは上書きしていない。
- [D20成功Actions #41495915535](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41495915535)：**SUCCESS、10テスト合格**。元artifactの取得とSHA256照合、保存済み45対の検査・比較が成功。[artifact #13167040481](https://github.com/kineticnapier/DoMachinesDreamOfDeparture/actions/runs/41495915535/artifacts/13167040481)に全45対の最初の分岐・結果JSON・検査/実行ログを保存。
- D19の陰性対照一致、各行の物理KeyDown=Hit+Extra、trace長・時刻順・case/repetition/seed、D19集計値と対比較保存値を検証。A0 argmax 9対/144目標、sampled 36対/576目標。

## 物理KeyDownの差が生じた4対
| 評価条件（case / repetition） | 間隔 / 開始 | 最初の相違（before → after） | Hit差 / Extra差 / KeyDown差 |
|---|---|---|---|
| **2 / 1**（seed 320201） | 210ms / 1.290s | 第3打：1.823s → 1.857s、同じleft・Extra。目標3への偏差 −97ms → −63ms。即時報酬 −0.05 → 0.0 | 0 / 0 / 0 |
| **4 / 3**（seed 320403） | 240ms / 1.075s | **第1打：3.718s・left・Hit（目標11に+3ms）→ 3.689s・left・Extra（同目標に−26ms）**。この対だけHit減少・Extra増加 | **−1 / +2 / +1** |
| **5 / 1**（seed 320501） | 240ms / 1.290s | 第2打：4.831s → 4.834s、同じleft・Extra。目標15への偏差 −59ms → −56ms | 0 / 0 / 0 |
| **7 / 2**（seed 320702） | 250ms / 1.075s | 第1打：1.570s → 1.568s、同じleft・Hit。目標2への偏差 −5ms → −7ms。全対で最も早い分岐時刻1.568s | 0 / 0 / 0 |

- sampled **32/36対の物理traceが完全一致**、4/36対で相違。argmaxは**9/9対完全一致**（物理KeyDownなし）。
- Hit減少は**1/36対**、Extra増加は**1/36対**。**同一のcase4/repetition3**で発生し、別の対に分散したものではない。
- この対の全traceは、before **2打**（3.718s Hit、4.019s Extra）からafter **3打**（3.689s Extra、4.003s Extra、4.324s Extra）へ変化。第1打が**29ms早まり**、HitがExtraに変わった。追加のExtraが1件あるため、合計でExtraは**+2**。
- それ以外の3対は、時刻・偏差・場合によって即時報酬が変化しても、Hit・Extra・物理KeyDownの件数は変わらなかった。

## 集計と否定的結果
| 指標 | 更新前 | 更新後 | 差 |
|---|---:|---:|---:|
| sampled Hit / 576目標 | **5** | **4** | **−1** |
| sampled Extra / 576目標 | **44** | **46** | **+2** |
| sampled 物理KeyDown | **49** | **50** | **+1** |
| argmax Hit / 144目標 | 0 | 0 | 0 |
| argmax Extra / 144目標 | 0 | 0 | 0 |
| argmax 物理KeyDown | 0 | 0 | 0 |

**否定的結果：D19の初回PPO更新はA0のHit改善に結びつかなかった。** D20では、その減少とExtra増加が、同一評価対の物理打鍵時刻・判定の変化として観測されることを特定した。事前登録の `sampled_hit_increase_without_extra_increase` は**false**。

## 解釈の制限と次の検証候補
- この比較は**イベント列を同じインデックスで並べた記述的照合**。同じ位置のイベントが同一の行動指令から生じたとは限らない。保存済みD19には各物理KeyDown直前の**8次元観測、行動指令、内部Motor状態**がないため、なぜ3.689sに早まったかという因果的説明はできない。
- 次の独立研究候補は、case4/repetition3の**観測→行動指令→物理KeyDown**の対応を限定して取得し、発火までの遅延を比較すること。別の事前登録とDraft PRなしに実行しない。今回の結果から事後的に評価基準を変更しない。
- A1二指・A3未学習対照・身体介入は未実施。教師なし運指獲得、人間同等性、単指物理的不可能性、正式研究002の承認は未証明。左右交代率・使用率は成功条件にしていない。
- D19 PR #62、基準点PR #44、リファクタリング監査PR #34、既存研究PR、身体・報酬・PPO本体・元重みは変更していない。**Draft維持。自動マージ・Draft解除・マージは永久禁止。**
