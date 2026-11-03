# 研究A v2-L0-D32 — step413のPPO更新前後の確率差をパラメータ群交換で検証（事前登録）

**探索的研究。正式研究002ではない。** 本文のみを実装・テスト・解析より先にコミットし、事後に条件・評価基準・停止条件を変更しない。元の研究A基準点はDraft PR #44。監査Draft PR #34およびD31以前のPRは変更しない。

## 単一の問い
D27〜D31で確認したA0単指・seed320403・step413（4.130秒）の左指押下確率（left+both）0.3814058453→0.3826222569の増分は、保存済み最初のPPO更新（Adam 2 step）で変化した**共有特徴抽出器 `shared`** と **行動出力層 `actor`** のどちらのパラメータ群の変更に主に対応するか。既存の2組の重みを混成する「重み介入」であり、PPO勾配や報酬の因果寄与を直接測定するものではない。

## 凍結原本・実行環境
- baseはD31のDraft PR #74の研究ブランチ `research/a-v2-l0-d31-a0-local-extra-expectation`。新規ブランチ `research/a-v2-l0-d32-step413-parameter-group-swap` / 新規Draft PRのみを変更。
- D8成功Actions #41397065687、artifact #13119278298：`native-initial-state.pt` と `native.json`/`default.json`。
- D12成功Actions #41424435292、artifact #13130809065：`capture.json`/`native.json`/`default.json`/`comparison.json`、特に`native.weights_after`。D13の既存SHA・物理記録検査を再利用。
- D28最終成功Actions #41562899406、artifact #13197677066：`result.json` の `step413.before` と `step413.after_to_none` に保存された同一8次元観測と4行動確率。A0 seed320403、step413、元D22 run #41515190995、物理介入D27の因果結果は参照のみ。
- Python 3.11 / PyTorch 2.5.1+cpu / CPU native（ATEN_CPU_CAPABILITY未設定）/ torch 1スレッド。元の `ActorCritic` と `torch.softmax` を使用。新規学習・PPO/Adam更新・勾配計算・Motor再生・新規KeyDown・Hit/Extra採点は**全て0**。

## 条件・比較対象
1. D12の初期モデルとnative更新後モデルをSHA・float32ビット列で照合。元モデルの1509要素を `shared.*`, `actor.*`, `critic.*` の3群に分ける。元重みを変更しない。D28の両観測が完全一致し、保存済みの更新前/後確率と再推論の4行動確率が**各成分絶対差5e-6以下**でなければ失敗。
2. **同一の保存済みstep413観測**で、`shared` と `actor` の更新前(0)/後(1)を交差させた4条件 (00,10,01,11) を1回ずつ推論する。全条件の`critic`は更新前で固定。追加で`shared=1,actor=1,critic=1`を陰性対照として推論し、`11`とのlogits・確率がビット単位で完全一致するか検査する。
3. 各条件の4行動logits/確率と `P(left)+P(both)` を記録。全更新差 `delta=p11-p00`、共有群の対称平均寄与 `shared=0.5*((p10-p00)+(p11-p01))`、actor群 `actor=0.5*((p01-p00)+(p11-p10))`、交互作用 `p11-p10-p01+p00` を**float64で計算**。寄与和とdeltaの差は1e-12以下。
4. 陰性対照の完全一致、有限性、確率の和1（許容1e-6）、群の正確な切り分け、元重み不変、D28との一致を必須とする。寄与が正・負・0、どちらが大きいかに関係なく、全検査成立なら `valid_step413_parameter_group_swap`。欠落・不一致・非有限値・再構成誤差超過なら `invalid_step413_parameter_group_swap` として失敗ログを残す。

## 停止条件・解釈の制限
- 凍結済み原本3件を読み、固定観測への**5回の方策推論のみ**、専用単体テスト。追加seed、追加学習、追加物理試行は禁止。長時間実行中の重複ジョブは起動しない。
- この群交換は**非学習の混成モデル**であり、PPOの行動別優位度・勾配・Adam内部状態の寄与を証明しない。特にD17の行動別勾配ノルムから本結果を直接推論しない。局所1観測で、他seed・A1・A3・閉ループHit/Extra・人間同等性・単指物理的不可能性・教師なし運指獲得・正式研究002承認を主張しない。
- 否定的な既存結果D19：sampled 576目標でHit5→4、Extra44→46、物理KeyDown49→50。改善なし。結果の正負に関係なく維持。
- 成功・失敗の結果JSON、検査・実行ログを専用Actions artifactに保存し、数値と否定的結果を結果文書・Draft PRに永続保存する。**自動マージ、Draft解除、既存PRのマージは永久禁止。**
